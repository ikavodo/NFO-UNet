"""Do benchmark conclusions depend on the label?  B0/T1(/-om/-pt) scored on seq1 against (a) pseudo-GT, (b) the
silhouette label L = (blended clear-frame silhouette minus empty-scene disparity > 0.2) u pseudo-GT
(extent_label_test.py, J-tuned setting), (c) hand frames (polygons, brush). 2026-10-13

    python -m benchmark.nfo_vos.label_robustness      # labels cached in results/benchmark/rr/labels_sil/

Scored frames: t0+1 .. t0+W-2 with a non-empty label (as in the benchmark); metric: region J at native res.
Window-level bootstrap (no twin clustering: seq1 only, 15 windows) -> CIs are indicative only.
"""
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.extent_label_test import DEPTH, SEQ, bank, silhouette
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

METHODS = ['b0', 't1', 'b0-om', 't1-om', 'b0-pt', 't1-pt']
LDIR = f'{TR.RES}/labels_sil'
THETA = 0.2
PAIRS = [('t1', 'b0'), ('t1-om', 't1'), ('b0-om', 'b0')]


def J(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def labels(t, E, d):
    path = f"{LDIR}/{t['id']}.npz"
    if os.path.exists(path):
        return np.load(path)['L']
    w0, w1 = t['warmup']
    bg = np.median(np.stack([V.native(k, SEQ) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
    L = []
    for k in t['frames']:
        P = gt_mask_native(SEQ, k)
        if P is None or not P.any():
            L.append(np.zeros((600, 800), bool)); continue
        S, P = silhouette(k, E, bg, d)
        L.append((S & (d <= THETA)) | P)
    L = np.stack(L)
    os.makedirs(LDIR, exist_ok=True); np.savez_compressed(path, L=L)
    return L


def boot(x, B=10000):
    x = np.asarray(x); r = np.random.default_rng(0)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    d, E = np.load(DEPTH), bank('clear')
    hand = {tag: {} for tag in ('handpoly', 'handmask_raw')}
    for tag in hand:
        for f in os.listdir(f'data/nfo_processed/{SEQ}_gt'):
            if f.endswith(f'_{tag}.png'):
                hand[tag][int(f[:5])] = cv2.imread(f'data/nfo_processed/{SEQ}_gt/{f}', 0) == 1
    win = {m: {'pseudo': [], 'sil': []} for m in METHODS}
    fr = {m: {tag: [] for tag in hand} for m in METHODS}
    nh = {tag: 0 for tag in hand}
    for i, t in enumerate(T):
        L = labels(t, E, d)
        P = [gt_mask_native(SEQ, k) for k in t['frames']]
        M = {m: np.load(f"{TR.RES}/masks/{m}/{t['id']}.npz")['masks'] for m in METHODS}
        ks = [j for j in range(1, len(t['frames']) - 1) if P[j] is not None and P[j].any()]
        for m in METHODS:
            win[m]['pseudo'].append(np.nanmean([J(M[m][j], P[j]) for j in ks]))
            win[m]['sil'].append(np.nanmean([J(M[m][j], L[j]) for j in ks]))
        for tag, H in hand.items():
            for j in range(1, len(t['frames']) - 1):
                if t['frames'][j] in H:
                    nh[tag] += 1
                    for m in METHODS:
                        fr[m][tag].append(J(M[m][j], H[t['frames'][j]]))
        print(f"{i + 1}/{len(T)} {t['id']}", flush=True)

    print(f'\nmean J over {len(T)} seq1 windows (pseudo-GT / silhouette label) and over hand frames '
          f'(polygon n={nh["handpoly"]} / brush n={nh["handmask_raw"]})')
    for m in METHODS:
        print(f"  {m:6s}  {np.mean(win[m]['pseudo']):.3f} / {np.mean(win[m]['sil']):.3f}   |  "
              f"{np.mean(fr[m]['handpoly']):.3f} / {np.mean(fr[m]['handmask_raw']):.3f}")
    print('\ndifferences (mean, 95% bootstrap CI over windows or frames)')
    for a, b in PAIRS:
        row = []
        for src, key in (('win', 'pseudo'), ('win', 'sil'), ('fr', 'handpoly'), ('fr', 'handmask_raw')):
            D = win if src == 'win' else fr
            x = np.array(D[a][key]) - np.array(D[b][key])
            lo, hi = boot(x)
            row.append(f'{x.mean():+.3f} [{lo:+.3f},{hi:+.3f}]')
        print(f'  {a} - {b}:  pseudo {row[0]}  sil {row[1]}  polygon {row[2]}  brush {row[3]}')


if __name__ == '__main__':
    main()
