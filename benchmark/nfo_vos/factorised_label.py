"""V = S ∧ ¬O at fine scale, no learning (2026-10-14, fixed before running). Also the offline version of a cleaned S1
dense prompt (envelope from the composite, occluders removed by change vs the empty scene).

S = M_C (SAM2 on the GT-free composite, oracle alignment; cached by composite_envelope.py)
¬O = change vs the empty scene, at pixels (pix_bg = log 5x5 SSD to the empty scene) or stride 4 (z4)
  V̂_τ = (M_C ∧ score > τ) ∪ M,   and the prompt alone P_τ = M_C ∧ score > τ
τ: leave-one-frame-out on the brush frames (maximise mean J vs brush on the other frames); for polygon-only frames
τ = the value chosen on all brush frames. Reported vs polygons (17) and brush (18).
PASS: J(V̂) > J(M) vs polygons, paired bootstrap CI > 0.
"""
import json
import os

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, OUT, SEQ, stats, window_for
from benchmark.nfo_vos.diag_composite_diff import feats, static_z
from benchmark.nfo_vos.diag_fine_add import up

IMG = 'images/benchmark/rr_checks/composite_tests'


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    tag = lambda s: sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith(f'_{s}.png'))
    brush, poly = tag('handmask_raw'), tag('handpoly')
    bgs, D = {}, {}
    with torch.inference_mode():
        for k in sorted(set(brush) | set(poly)):
            if not os.path.exists(f'{OUT}/MC_{k:05d}.npy'):
                continue
            t = window_for(k, T)
            if tuple(t['warmup']) not in bgs:
                w0, w1 = t['warmup']
                bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
                bgs[tuple(t['warmup'])] = (bg, feats(bg))
            bg, Fbg = bgs[tuple(t['warmup'])]
            Rk = V.native(k, SEQ).astype(np.float32)
            D[k] = dict(MC=np.load(f'{OUT}/MC_{k:05d}.npy'), M=cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127,
                        pix_bg=np.log(cv2.boxFilter((Rk - bg) ** 2, -1, (5, 5)) + 1),
                        z4=up(static_z(feats(Rk)[4], Fbg[4])))
    H = {('brush', k): cv2.imread(f'{GT}/{k:05d}_handmask_raw.png', 0) == 1 for k in brush if k in D}
    H |= {('poly', k): cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1 for k in poly if k in D}
    rng = np.random.default_rng(0)
    for score in ('pix_bg', 'z4'):
        allv = np.concatenate([D[k][score][D[k]['MC']] for k in D])
        grid = np.percentile(allv, np.arange(0, 96, 5))
        Jb = lambda k, tau: stats((D[k]['MC'] & (D[k][score] > tau)) | D[k]['M'], H[('brush', k)])[0]
        bk = [k for (r, k) in H if r == 'brush']
        tau_all = grid[np.argmax([np.mean([Jb(k, g) for k in bk]) for g in grid])]
        tau = {k: (grid[np.argmax([np.mean([Jb(j, g) for j in bk if j != k]) for g in grid])] if k in bk else tau_all)
               for k in D}
        print(f'\n{score}: τ on all brush frames = {tau_all:.2f} (percentile {np.searchsorted(grid, tau_all) * 5} of '
              f'scores inside M_C); LOFO τ range {min(tau.values()):.2f}–{max(tau.values()):.2f}')
        for ref in ('poly', 'brush'):
            ks = [k for (r, k) in H if r == ref]
            rows = {n: [] for n in ('M', 'P = M_C ∧ change', 'V = P ∪ M')}
            for k in ks:
                P = D[k]['MC'] & (D[k][score] > tau[k])
                h = H[(ref, k)]
                rows['M'].append(stats(D[k]['M'], h)); rows['P = M_C ∧ change'].append(stats(P, h))
                rows['V = P ∪ M'].append(stats(P | D[k]['M'], h))
            print(f'  vs {ref} (n={len(ks)}): J / P / R')
            for n, v in rows.items():
                print(f'    {n:18s} ' + ' / '.join(f'{x:.3f}' for x in np.mean(v, 0)))
            d = np.array([a[0] - b[0] for a, b in zip(rows['V = P ∪ M'], rows['M'])])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f'    ΔJ (V − M) {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], better on {(d > 0).sum()}/{len(d)}'
                  + ((' -> PASS' if lo > 0 else ' -> FAIL') if ref == 'poly' else ''))
        if score == 'pix_bg':   # visual trace
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            ks = [k for k in poly if k in D]
            fig, ax = plt.subplots(2, (len(ks) + 1) // 2, figsize=(12, 6))
            for a_, k in zip(ax.ravel(), ks):
                P = D[k]['MC'] & (D[k][score] > tau[k]); M = D[k]['M']; h = H[('poly', k)]
                ys, xs = np.nonzero(h | D[k]['MC'])
                y0, y1, x0, x1 = max(0, ys.min() - 10), min(600, ys.max() + 10), max(0, xs.min() - 10), min(800, xs.max() + 10)
                a_.imshow(V.native(k, SEQ)[y0:y1, x0:x1], cmap='gray')
                ov = np.zeros((y1 - y0, x1 - x0, 4)); ov[(P & ~M)[y0:y1, x0:x1]] = (1, 0, 1, 0.6)
                a_.imshow(ov)
                for mm, col in ((h, 'cyan'), (M, 'lime')):
                    a_.contour(mm[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=col, linewidths=0.6)
                a_.set_title(str(k), fontsize=8); a_.axis('off')
            for a_ in ax.ravel()[len(ks):]:
                a_.axis('off')
            fig.suptitle('cyan = polygon, green = pseudo-GT, magenta = added by (composite envelope ∧ changed vs empty '
                         'scene, pixel level)', fontsize=9)
            fig.tight_layout(); fig.savefig(f'{IMG}/factorised_added.png', dpi=75)


if __name__ == '__main__':
    main()
