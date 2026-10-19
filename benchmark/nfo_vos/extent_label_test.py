"""Pre-registered test (docs/vos_benchmark/archive/nfo_vos_pilot_findings_full.md, 2026-10-13): automatic extent-leaning labels vs the
annotator's polygon labels.        python -m benchmark.nfo_vos.extent_label_test

Label = (blended exemplar silhouette minus empty-scene disparity > theta) u SAM2-ceiling; exemplar_gt.py v2 without G.
Tune (theta, v_min) on seg 4 by mean F2; freeze; score held-out seg 5 + seg 0 once; ablations on held-out.
"""
import json

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.exemplar_gt import (ASP, DEPTH, EXCL, N_DIFF, N_DISP, SEQ, SHIFTS, TOP, band_align, box,
                                           head_ok, place, sdf, seg_of)
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

TUNE = [1164, 1183, 1194, 1214, 1224, 1244, 1263, 1274, 1294]
TEST = [1490, 1520, 1550, 1580, 1610, 30, 70, 110]
THETAS, VMINS = (0.15, 0.2, 0.3, None), (0.6, 0.8, 'clear')
WARM = {0: (946, 1113), 4: (946, 1113), 5: (1349, 1435)}


def bank(vmin):
    E = []
    for s in TR.segments(SEQ):
        ks = TR.clear_frames(s) if vmin == 'clear' else \
            [k for k in range(s['start'], s['end'] + 1) if TR.gt_mask(k, SEQ) is not None and TR.visibility(k, s) >= vmin]
        for k in ks:
            m = gt_mask_native(SEQ, k)
            x0, y0, x1, y1 = box(k)
            if m is None or x0 < 0 or y0 < 0 or x1 > 800 or y1 > 600:
                continue
            c = m[y0:y1, x0:x1].astype(np.float32)
            c = c[:, ::-1] if s['dir'] == 'L' else c
            if head_ok(c):
                E.append((k, (x1 - x0) / (y1 - y0), c))
    return E


def silhouette(k, E, bg, d, align=True):
    b = box(k)
    a = (b[2] - b[0]) / (b[3] - b[1])
    left = seg_of(k)['dir'] == 'L'
    P = gt_mask_native(SEQ, k)
    N = (d < N_DISP) & (np.abs(V.native(k, SEQ).astype(np.float32) - bg) < N_DIFF)
    cand = [e for e in E if abs(e[0] - k) > EXCL]
    near = [e for e in cand if abs(e[1] - a) / a < ASP] or sorted(cand, key=lambda e: abs(e[1] - a))[:20]
    scored = []
    for _, _, c in near:
        c = c[:, ::-1] if left else c
        for dx in SHIFTS:
            S = place(c, b, dx)
            if S.any():
                scored.append(((S & P).sum() / max(P.sum(), 1) - (S & N).sum() / S.sum(), S))
    S = np.mean([sdf(S) for _, S in sorted(scored, key=lambda t: -t[0])[:TOP]], 0) > 0
    return (band_align(S, b, P, N) if align else S), P


def scores(L, H):
    tp = (L & H).sum()
    p, r = tp / max(L.sum(), 1), tp / max(H.sum(), 1)
    return dict(J=tp / max((L | H).sum(), 1), P=p, R=r, F2=5 * p * r / max(4 * p + r, 1e-9))


def main():
    d = np.load(DEPTH)
    bgs = {s: np.median(np.stack([V.native(k, SEQ) for k in range(a, b + 1)]), 0).astype(np.float32) for s, (a, b) in WARM.items()}
    H = {k: cv2.imread(f'data/nfo_processed/{SEQ}_gt/{k:05d}_handpoly.png', 0) == 1 for k in TUNE + TEST}
    banks = {v: bank(v) for v in VMINS}
    print('bank sizes:', {v: len(E) for v, E in banks.items()})
    lab = lambda S, P, th: ((S & (d <= th)) if th is not None else S) | P
    sil = {}
    def get(k, v, align=True):
        key = (k, v, align)
        if key not in sil:
            sil[key] = silhouette(k, banks[v], bgs[seg_of(k)['idx']], d, align)
        return sil[key]

    print('--- tuning (seg 4, 9 frames): mean F2 [J]')
    res = {}
    for v in VMINS:
        for th in THETAS:
            sc = [scores(lab(*get(k, v), th), H[k]) for k in TUNE]
            res[(th, v)] = np.mean([s['F2'] for s in sc])
            print(f'  theta {th}, v_min {v}: F2 {res[(th, v)]:.3f} [J {np.mean([s["J"] for s in sc]):.3f}]')
    th, v = max(res, key=res.get)
    print(f'FROZEN: theta {th}, v_min {v}; SAM2-ceiling tuning F2 {np.mean([scores(gt_mask_native(SEQ, k), H[k])["F2"] for k in TUNE]):.3f}')

    print('--- HELD-OUT (seg 5 + seg 0, 8 frames), per frame F2: label / SAM2-ceiling')
    arms = {'label (frozen)': lambda k: lab(*get(k, v), th), 'SAM2-ceiling': lambda k: gt_mask_native(SEQ, k),
            'ablation: no band alignment': lambda k: lab(*get(k, v, False), th),
            'ablation: no depth cleaning': lambda k: lab(*get(k, v), None),
            'ablation: v_min = clear': lambda k: lab(*get(k, 'clear'), th)}
    S = {a: [scores(f(k), H[k]) for k in TEST] for a, f in arms.items()}
    for i, k in enumerate(TEST):
        print(f'  {k}: {S["label (frozen)"][i]["F2"]:.3f} / {S["SAM2-ceiling"][i]["F2"]:.3f}')
    for a, sc in S.items():
        print(f'  {a:30s} F2 {np.mean([s["F2"] for s in sc]):.3f}  J {np.mean([s["J"] for s in sc]):.3f}  '
              f'P {np.mean([s["P"] for s in sc]):.3f}  R {np.mean([s["R"] for s in sc]):.3f}')
    dF = np.array([a['F2'] - b['F2'] for a, b in zip(S['label (frozen)'], S['SAM2-ceiling'])])
    rng = np.random.default_rng(0)
    bs = [dF[rng.integers(0, len(dF), len(dF))].mean() for _ in range(10000)]
    prec = np.mean([s['P'] for s in S['label (frozen)']])
    print(f'dF2 = {dF.mean():+.3f}  95% bootstrap CI over frames [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
    print(f'PASS rule (dF2 >= 0.05 AND precision >= 0.60): {"PASS" if dF.mean() >= 0.05 and prec >= 0.60 else "FAIL"}')


if __name__ == '__main__':
    main()
