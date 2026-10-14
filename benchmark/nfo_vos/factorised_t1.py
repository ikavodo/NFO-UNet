"""V̂ as a METHOD on top of a tracker (2026-10-14, fixed before running): replace the SAM2-ceiling M in
factorised_label.py by the tracker's own stored mask.

  V̂_X = (M_C ∧ pix_bg > τ) ∪ X,   X ∈ {B0, T1} (results/benchmark/rr/masks/<method>/<trial>.npz)
τ = 6.31, FIXED from factorised_label.py (chosen on the brush frames with the SAM2-ceiling base; not re-tuned).
M_C, composite alignment and SAM2 prompt still use GT boxes (oracle); tracker alignment is the next step.
Frames: seq1 hand-labelled frames (polygon, brush) that fall inside an admissible trial window.
PASS: J(V̂_T1) > J(T1) vs polygons, paired bootstrap CI > 0.
"""
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, OUT, SEQ, stats, window_for

TAU = 6.31


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    tag = lambda s: sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith(f'_{s}.png'))
    rng, bgs = np.random.default_rng(0), {}
    for ref, suf in (('polygon', 'handpoly'), ('brush', 'handmask_raw')):
        rows = []
        for k in tag(suf):
            win = [t for t in T if k in t['frames']]
            if not win or not os.path.exists(f'{OUT}/MC_{k:05d}.npy'):
                continue
            t = win[0]; j = t['frames'].index(k)
            if tuple(t['warmup']) not in bgs:
                w0, w1 = t['warmup']
                bgs[tuple(t['warmup'])] = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
            Rk = V.native(k, SEQ).astype(np.float32)
            P = np.load(f'{OUT}/MC_{k:05d}.npy') & (np.log(cv2.boxFilter((Rk - bgs[tuple(t['warmup'])]) ** 2, -1, (5, 5)) + 1) > TAU)
            H = cv2.imread(f'{GT}/{k:05d}_{suf}.png', 0) == 1
            r = {'k': k}
            for m in ('b0', 't1'):
                X = np.load(f"{TR.RES}/masks/{m}/{t['id']}.npz")['masks'][j].astype(bool)
                r[m], r[f'V̂_{m}'] = stats(X, H), stats(P | X, H)
            r['SAM2-ceiling'] = stats(cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127, H)
            rows.append(r)
        print(f'vs {ref} (n={len(rows)} frames inside trial windows): J / P / R')
        for n in ('b0', 'V̂_b0', 't1', 'V̂_t1', 'SAM2-ceiling'):
            print(f'  {n:13s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([r[n] for r in rows], 0)))
        for m in ('b0', 't1'):
            d = np.array([r[f'V̂_{m}'][0] - r[m][0] for r in rows])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f'  ΔJ (V̂_{m} − {m}) {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], better on {(d > 0).sum()}/{len(d)}'
                  + ((' -> PASS' if lo > 0 else ' -> FAIL') if (ref, m) == ('polygon', 't1') else ''))


if __name__ == '__main__':
    main()
