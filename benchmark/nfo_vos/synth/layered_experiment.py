"""Layered synthetic NFO: calibrate on input statistics, then test SAM2 behaviour on held-out clips (2026-10-15).
Design: docs/layer_sam2/data/synthetic_nfo_design.md. Generator: gpjatk_foliage.make_layered_clip (occluder ≡ background).

Arms (one generator component changes between neighbouring arms):
  perlin        occ_perlin mask (clump 24), photo background                 (today's best, baseline)
  depth         depth-test mask from the photo's own disparity               (vs perlin: mask source)
  depth+twigs   depth + occ_branch twig layer drawn into the scene           (vs depth: fine foliage)
  stack+twigs   dead-leaves stack of K = 3 photo cut-outs + twigs            (vs depth+twigs: layering/diversity)
Phase A, calibration (clips of subjects p1–p6, no tracker): per arm, grid ρ ∈ {.35, .45, .55, .65} × twigs ∈
  {50, 100, 200, 400} (twig arms only); pick the setting minimising L = Σ_bins |share − NFO| + |void − 0.476|, with share
  = area of H's connected components per size bin (cells of 15 px) and void = |void| / |H ∪ void|.
  NFO targets (handAv, seq2–4, measured 2026-10-15): shares 0.013 / 0.009 / 0.108 / 0.870, void 0.476.
Phase B, held out (subjects p7–p18, s1, cameras c1–c4 in turn, 2 seeds = 24 clips per arm), B0 and T1, box + point
  prompt at t0, scored on frames 1–39 against H with void ignored (score_void.py convention).
PASS (pre-registered, design §6), per arm, T1:  P ∈ [0.83, 0.95], R ∈ [0.75, 0.89]; size-binned recall of H
  components: [0,.5) ≤ 0.25, [.5,1) ∈ [0.25, 0.55], [1,4) ∈ [0.55, 0.80], [4,∞) ≥ 0.75; |J(B0) − J(T1)| ≤ 0.03.
  Fitted-statistics check: every share within ±0.05 of NFO and void within 0.476 ± 0.10 (else the arm cannot pass).
Not modelled yet (stated so no one reads a pass as complete): sensor noise / JPEG / sway calibrated to NFO, person
scale (GPJATK 10–18 cells tall vs NFO ≈ 13), person contrast.

    ../samurai/.venv/bin/python -m benchmark.nfo_vos.synth.layered_experiment
"""
import itertools
import json
import os
import shutil
import tempfile

import cv2
import numpy as np

from benchmark.nfo_vos.synth.gpjatk_foliage import OUT, make_layered_clip

ROOT = f'{OUT}/layered'
RES = 'results/benchmark/synth'
CELL = 16 * 960 / 1024
BINS = ((0, .5), (.5, 1), (1, 4), (4, np.inf))
NFO_SHARE, NFO_VOID = np.array([0.013, 0.009, 0.108, 0.870]), 0.476
CAMS = ('c1', 'c2', 'c3', 'c4')
CAL = [(f'p{i}s1', CAMS[(i - 1) % 4]) for i in range(1, 7)]
HELD = [(f'p{i}s1', CAMS[(i - 1) % 4]) for i in range(7, 19)]
ARMS = {'perlin': ('perlin', False), 'depth': ('depth', False), 'depth+twigs': ('depth', True),
        'stack+twigs': ('stack', True)}
RHOS, TWIGS = (0.35, 0.45, 0.55, 0.65), (50, 100, 200, 400)
# Follow-up arm (declared 2026-10-15 after the first run, before its own result; same bands): perlin + twigs, finer
# twig grid around the void target (100 twigs gave void ≈ 0.34–0.37, 200 gave ≈ 0.65 in the first run).
if os.environ.get('ARMSET') == 'followup':
    ARMS = {'perlin+twigs': ('perlin', True)}
    RHOS, TWIGS = (0.35, 0.45, 0.55), (125, 150, 175)


def bin_of(z):
    return [lo <= z < hi for lo, hi in BINS].index(True)


def input_stats(Hs, voids):
    A, vs = np.zeros(4), []
    for H, v in zip(Hs, voids):
        for t in range(0, len(H), 5):
            n, L, cs, _ = cv2.connectedComponentsWithStats(H[t].astype(np.uint8), connectivity=8)
            for c in range(1, n):
                A[bin_of(cs[c, 4] / CELL ** 2)] += cs[c, 4]
            vs.append(v[t].sum() / max((H[t] | v[t]).sum(), 1))
    return A / max(A.sum(), 1), float(np.mean(vs))


def calibrate():
    best = {}
    for arm, (mode, tw) in ARMS.items():
        rows = []
        for rho, twigs in itertools.product(RHOS, TWIGS if tw else (0,)):
            Hs, vs = [], []
            for i, (s, c) in enumerate(CAL):
                *_, H, v = make_layered_clip(s, c, rho, 100 + i, mode, ROOT, twigs=twigs, save=False)
                Hs.append(H); vs.append(v)
            sh, vd = input_stats(Hs, vs)
            L = np.abs(sh - NFO_SHARE).sum() + abs(vd - NFO_VOID)
            rows.append(dict(rho=rho, twigs=twigs, share=sh.round(3).tolist(), void=round(vd, 3), L=round(float(L), 3)))
            print(f'  cal {arm:12s} ρ {rho:.2f} twigs {twigs:3d}: shares {np.round(sh, 3)} void {vd:.3f}  L {L:.3f}', flush=True)
        best[arm] = min(rows, key=lambda r: r['L']) | {'grid': rows}
    return best


def run(method, dirs):
    import torch
    from benchmark.nfo_vos.run_sam2_video import build_predictor
    pred = build_predictor(method)
    for cd in dirs:
        p = f'{cd}/masks_{method}.npz'
        if os.path.exists(p):
            continue
        m = json.load(open(f'{cd}/meta.json'))
        fr = sorted(os.listdir(f'{cd}/frames'))
        stage = tempfile.mkdtemp(prefix='gpj_')
        for i, f in enumerate(fr):
            os.symlink(os.path.abspath(f'{cd}/frames/{f}'), f'{stage}/{i}.jpg')
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                st = pred.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
                pred.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=np.array([m['point']], np.float32),
                                           labels=np.array([1], np.int32), box=np.array(m['box'], np.float32))
                M = np.zeros((len(fr), st['video_height'], st['video_width']), bool)
                for fi, _, lg in pred.propagate_in_video(st):
                    M[fi] = (lg[0, 0] > 0).cpu().numpy()
        finally:
            shutil.rmtree(stage)
        np.savez_compressed(p, masks=M)
    del pred; torch.cuda.empty_cache()


def score(dirs, method):
    S_, bins, loss, n = [], [[] for _ in BINS], 0, 0
    for cd in dirs:
        H, void, V = (np.load(f'{cd}/{k}.npy') for k in ('H', 'void', 'V'))
        M = np.load(f'{cd}/masks_{method}.npz')['masks']
        for t in range(1, len(H)):
            n += 1
            if M[t].sum() < 0.1 * max(V[t].sum(), 1):
                loss += 1
            keep = ~void[t]
            tp = (M[t] & H[t] & keep).sum()
            S_.append([tp / max(((M[t] | H[t]) & keep).sum(), 1), tp / max((M[t] & keep).sum(), 1),
                       tp / max((H[t] & keep).sum(), 1)])
            k, L, cs, _ = cv2.connectedComponentsWithStats(H[t].astype(np.uint8), connectivity=8)
            for c in range(1, k):
                bins[bin_of(cs[c, 4] / CELL ** 2)].append((M[t] & (L == c)).sum() / cs[c, 4])
    return dict(JPR=np.mean(S_, 0).round(3).tolist(), J_clip=None, bins=[round(float(np.mean(b)), 3) if b else None for b in bins],
                bin_n=[len(b) for b in bins], loss=f'{loss}/{n}')


def verdict(r, cal):
    J, P, R = r['t1']['JPR']; b = r['t1']['bins']
    checks = {'P': 0.83 <= P <= 0.95, 'R': 0.75 <= R <= 0.89,
              'bin0': b[0] is not None and b[0] <= 0.25, 'bin1': b[1] is not None and 0.25 <= b[1] <= 0.55,
              'bin2': b[2] is not None and 0.55 <= b[2] <= 0.80, 'bin3': b[3] is not None and b[3] >= 0.75,
              'B0~T1': abs(r['b0']['JPR'][0] - J) <= 0.03,
              'fit_shares': bool(np.all(np.abs(np.array(cal['share']) - NFO_SHARE) <= 0.05)),
              'fit_void': abs(cal['void'] - NFO_VOID) <= 0.10}
    return checks, all(checks.values())


def montage(dirs, path):
    rows = []
    for cd in dirs[:6]:
        g = cv2.imread(f'{cd}/frames/00020.jpg')
        H, void, S = (np.load(f'{cd}/{k}.npy')[20] for k in ('H', 'void', 'S'))
        M = np.load(f'{cd}/masks_t1.npz')['masks'][20]
        o = g.copy()
        o[H] = (0.5 * o[H] + [0, 127, 0]).astype(np.uint8)
        o[void] = (0.5 * o[void] + [0, 127, 127]).astype(np.uint8)
        o[S & ~H & ~void] = (0.5 * o[S & ~H & ~void] + [0, 0, 127]).astype(np.uint8)
        c = cv2.findContours(M.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(o, c, -1, (255, 0, 255), 1)
        r = np.concatenate([g, o], 1)
        cv2.putText(r, os.path.basename(cd), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        rows.append(cv2.resize(r, (960, 270)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.concatenate(rows, 0))


def main():
    os.makedirs(RES, exist_ok=True)
    print('Phase A: calibration on input statistics (no tracker)', flush=True)
    cal = calibrate()
    out = {}
    for arm, (mode, _) in ARMS.items():
        c = cal[arm]
        print(f'\n{arm}: chosen ρ {c["rho"]}, twigs {c["twigs"]}, shares {c["share"]}, void {c["void"]}, L {c["L"]}', flush=True)
        dirs = []
        for seed, (i, (s, cam)) in itertools.product((0, 1), enumerate(HELD)):
            cid = make_layered_clip(s, cam, c['rho'], 1000 + 100 * seed + i, mode, f'{ROOT}/{arm}', twigs=c['twigs'])[0]
            dirs.append(f'{ROOT}/{arm}/{cid}')
        for meth in ('b0', 't1'):
            run(meth, dirs)
        r = {m: score(dirs, m) for m in ('b0', 't1')}
        checks, ok = verdict(r, c)
        out[arm] = dict(calibration={k: v for k, v in c.items() if k != 'grid'}, grid=c['grid'], held=r, checks=checks, PASS=ok)
        montage(dirs, f'images/benchmark/gpjatk_foliage/layered_{arm.replace("+", "_")}.png')
        print(f'{arm}: T1 J/P/R {r["t1"]["JPR"]}  bins {r["t1"]["bins"]} (n {r["t1"]["bin_n"]})  loss {r["t1"]["loss"]}; '
              f'B0 J/P/R {r["b0"]["JPR"]}\n   checks {checks}\n   PASS = {ok}', flush=True)
        json.dump(out, open(f'{RES}/layered_experiment' + ('_followup' if os.environ.get('ARMSET') == 'followup' else '') + '.json', 'w'), indent=1)
    print('\nNFO reference: T1 J/P/R 0.750 / 0.887 / 0.819; bins 0.13 / 0.39 / 0.67 / 0.84; B0 − T1 +0.005')


if __name__ == '__main__':
    main()
