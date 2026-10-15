"""Sanity check before GPJATK step 1 (2026-10-15): is SAM2's recall insufficient on high-density synthetic foliage?

Clips: 6 training-split GPJATK sequences (subjects p1–p6), cameras c1–c4 in turn, 40 walking frames each, occ_perlin
foliage (gpjatk_foliage.py) at densities 0.60 / 0.75 / 0.85, clump 12 px, static, same seed per (seq, cam).
Prompt at t0 = 0 exactly as the NFO benchmark: box = bbox(S_0) and point = distance-transform peak of V_0.
Methods: B0 (SAM2.1 b+, SAMURAI mode off) and T1 (SAMURAI b+), forward propagation, run_sam2_video settings.
Scored on frames 1…39 against V (exact visible person) and against T_8 = V ∪ (closing_8(V) ∧ S) (generous label).

Pre-registered verdict (written before running): SAM2 recall is "insufficient" on this proxy if, at density 0.85,
for T1 vs V: (i) R ≤ 0.85 while P ≥ 0.85 (recall-limited, as on NFO: T1 P 0.887 / R 0.819, seq2–4 void-ignored);
(ii) recall of sub-cell components (< 0.5 cell, cell = 16 px of SAM2's 1024 input = 15 px here) ≤ 0.25 (NFO 0.13).
Frames where the mask area is < 10 % of |V| count as track loss and are reported separately (a loss is a tracking
failure, not the recall deficit we target).

    ../samurai/.venv/bin/python -m benchmark.nfo_vos.synth.sanity_recall gen|b0|t1|score
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

from benchmark.nfo_vos.synth.gpjatk_foliage import OUT, make_clip, montage

SAN = f'{OUT}/sanity'
SEQS = [('p1s1', 'c1'), ('p2s1', 'c2'), ('p3s1', 'c3'), ('p4s1', 'c4'), ('p5s1', 'c1'), ('p6s1', 'c2')]
DENS = tuple(float(x) for x in os.environ.get('DENS', '0.60,0.75,0.85').split(','))
CELL = 16 * 960 / 1024
BINS = ((0, .5), (.5, 1), (1, 4), (4, np.inf))


def clips():
    return sorted(d for d in os.listdir(SAN) if os.path.exists(f'{SAN}/{d}/meta.json'))


def gen():
    for dens in DENS:
        made = [make_clip(s, c, dens, seed=i, out=SAN) for i, (s, c) in enumerate(SEQS)]
        montage(made, f'images/benchmark/gpjatk_foliage/sanity_d{int(dens * 100)}.png')
        for cid, *_ in made:
            m = json.load(open(f'{SAN}/{cid}/meta.json'))
            print(f"{cid}: person coverage {m['coverage_person']:.2f}, frame coverage {m['coverage_frame']:.2f}")


def run(method):
    point_only = method.endswith('pt')       # diagnostic: point prompt only (no box)
    import torch
    from benchmark.nfo_vos.run_sam2_video import build_predictor
    pred = build_predictor(method[:2])
    for cid in clips():
        p = f'{SAN}/{cid}/masks_{method}.npz'
        if os.path.exists(p):
            continue
        m = json.load(open(f'{SAN}/{cid}/meta.json'))
        fr = sorted(os.listdir(f'{SAN}/{cid}/frames'))
        stage = tempfile.mkdtemp(prefix='gpj_')
        for i, f in enumerate(fr):
            os.symlink(os.path.abspath(f'{SAN}/{cid}/frames/{f}'), f'{stage}/{i}.jpg')
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                st = pred.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
                pred.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=np.array([m['point']], np.float32),
                                           labels=np.array([1], np.int32), box=None if point_only else np.array(m['box'], np.float32))
                M = np.zeros((len(fr), st['video_height'], st['video_width']), bool)
                for fi, _, lg in pred.propagate_in_video(st):
                    M[fi] = (lg[0, 0] > 0).cpu().numpy()
        finally:
            shutil.rmtree(stage)
        np.savez_compressed(p, masks=M)
        print(cid, method, flush=True)


def st(a, h):
    tp = (a & h).sum()
    return np.array([tp / max((a | h).sum(), 1), tp / max(a.sum(), 1), tp / max(h.sum(), 1)])


def score():
    k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    R = {}
    for cid in clips():
        dens = json.load(open(f'{SAN}/{cid}/meta.json'))['density']
        V, S = np.load(f'{SAN}/{cid}/V.npy'), np.load(f'{SAN}/{cid}/S.npy')
        for meth in [x for x in ('b0', 't1', 't1pt') if os.path.exists(f'{SAN}/{cid}/masks_{x}.npz')]:
            M = np.load(f'{SAN}/{cid}/masks_{meth}.npz')['masks']
            r = R.setdefault((dens, meth), dict(v=[], t8=[], loss=0, n=0, bins=[]))
            for t in range(1, len(V)):
                r['n'] += 1
                if M[t].sum() < 0.1 * V[t].sum():
                    r['loss'] += 1; continue
                T8 = V[t] | (cv2.morphologyEx(V[t].astype(np.uint8), cv2.MORPH_CLOSE, k8).astype(bool) & S[t])
                r['v'].append(st(M[t], V[t])); r['t8'].append(st(M[t], T8))
                n, L, cs, _ = cv2.connectedComponentsWithStats(V[t].astype(np.uint8), connectivity=8)
                for c in range(1, n):
                    r['bins'].append((cs[c, cv2.CC_STAT_AREA] / CELL ** 2, (M[t] & (L == c)).sum() / cs[c, cv2.CC_STAT_AREA]))
    print('density method | vs V: J / P / R | vs T_8: J / P / R | loss | component recall by size (cells) [n]')
    for (dens, meth), r in sorted(R.items()):
        b = '  '.join(f"[{lo},{hi}) {np.mean(v):.2f} ({len(v)})" if (v := [x for z, x in r['bins'] if lo <= z < hi]) else
                      f'[{lo},{hi}) n/a' for lo, hi in BINS)
        print(f"{dens:.2f} {meth} | " + ' / '.join(f'{x:.3f}' for x in np.mean(r['v'], 0)) + ' | ' +
              ' / '.join(f'{x:.3f}' for x in np.mean(r['t8'], 0)) + f" | {r['loss']}/{r['n']} | {b}")


if __name__ == '__main__':
    {'gen': gen, 'score': score}.get(sys.argv[1], lambda: run(sys.argv[1]))()
