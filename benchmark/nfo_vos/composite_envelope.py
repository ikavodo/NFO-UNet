"""Test 2b (2026-10-14, fixed before running): the GT-free composite's SAM2 mask M_C as an EXTENT ENVELOPE, with
occluders removed by the empty-scene depth filter of the silhouette label (exemplar_gt.py:141: keep disparity ≤ 0.2).
Corrects test 2 (composite_tests.py), which scored M_C as if it were a modal mask.

References: hand polygons (17) and brush masks (18) only. Pseudo-GT M is a competitor row, not a reference.
  containment: R(M_C, hand) and share of frames with R ≥ 0.9
  V̂ = (M_C ∧ d ≤ 0.2) ∪ M          (same form as silhouette label v1/v2)
PASS (gate for S2): mean J(V̂) > mean J(M) vs polygons, paired bootstrap CI > 0.
Caveat: the 0.2 threshold was chosen on these seq1 frames (2026-10-13), not held out.
M_C cached to results/benchmark/rr/composite_tests/MC_<k>.npy.
"""
import json
import os

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, OUT, SEQ, box, stats, window_for
from benchmark.nfo_vos.diag_fine_add import composites

IMG = 'images/benchmark/rr_checks/composite_tests'
D = np.load('results/benchmark/rr/depth/seq1_bg_disp_1036.npy')
FAR = D <= 0.2


def mc(k, T, pred, bgs):
    p = f'{OUT}/MC_{k:05d}.npy'
    if os.path.exists(p):
        return np.load(p)
    t = window_for(k, T)
    if tuple(t['warmup']) not in bgs:
        w0, w1 = t['warmup']
        bgs[tuple(t['warmup'])] = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    C = composites(SEQ, k, bgs[tuple(t['warmup'])])
    if C is None:
        return None
    x0, y0, x1, y1 = box(k)
    with torch.inference_mode():
        pred.set_image(np.stack([np.clip(C['gtfree'], 0, 255).astype(np.uint8)] * 3, -1))
        m, _, _ = pred.predict(point_coords=np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), point_labels=np.array([1]),
                               box=np.array([x0, y0, x1, y1]), multimask_output=False)
    np.save(p, m[0] > 0)
    return m[0] > 0


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    tag = lambda s: sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith(f'_{s}.png'))
    pred, bgs, rng = R.image_predictor(), {}, np.random.default_rng(0)
    for ref, suf in (('polygon', 'handpoly'), ('brush', 'handmask_raw')):
        rows = []
        for k in tag(suf):
            MC = mc(k, T, pred, bgs)
            if MC is None:
                continue
            H = cv2.imread(f'{GT}/{k:05d}_{suf}.png', 0) == 1
            M = cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127
            rows.append({'k': k, 'M': stats(M, H), 'M_C': stats(MC, H), 'M_C & far': stats(MC & FAR, H),
                         'V = (M_C & far) | M': stats((MC & FAR) | M, H)})
        print(f'vs {ref} (n={len(rows)}): J / P / R')
        for n in ('M', 'M_C', 'M_C & far', 'V = (M_C & far) | M'):
            print(f'  {n:22s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([r[n] for r in rows], 0)))
        rc = np.array([r['M_C'][2] for r in rows])
        print(f'  containment: R(M_C, {ref}) mean {rc.mean():.3f}, frames with R ≥ 0.9: {(rc >= 0.9).sum()}/{len(rc)}')
        d = np.array([r['V = (M_C & far) | M'][0] - r['M'][0] for r in rows])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'  ΔJ (V − M) {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], better on {(d > 0).sum()}/{len(d)}'
              + ((' -> PASS' if lo > 0 else ' -> FAIL') if ref == 'polygon' else ''))

    # side-by-side filled masks for frames with both polygon and brush
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = [k for k in tag('handpoly') if k in tag('handmask_raw')][:4]
    cols = ['frame', 'polygon', 'brush', 'pseudo-GT M', 'M_C (composite extent)', '(M_C ∧ far) ∪ M']
    fig, ax = plt.subplots(len(ks), len(cols), figsize=(11, 2.6 * len(ks)))
    for i, k in enumerate(ks):
        P = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1
        B = cv2.imread(f'{GT}/{k:05d}_handmask_raw.png', 0) == 1
        M = cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127
        MC = np.load(f'{OUT}/MC_{k:05d}.npy')
        ys, xs = np.nonzero(P | MC | M)
        y0, y1, x0, x1 = max(0, ys.min() - 10), min(600, ys.max() + 10), max(0, xs.min() - 10), min(800, xs.max() + 10)
        fr = V.native(k, SEQ)[y0:y1, x0:x1]
        for j, m in enumerate((None, P, B, M, MC, (MC & FAR) | M)):
            ax[i, j].imshow(fr, cmap='gray', vmin=0, vmax=255)
            if m is not None:
                ov = np.zeros(fr.shape + (4,)); ov[m[y0:y1, x0:x1]] = (1, 0, 1, 0.45)
                ax[i, j].imshow(ov)
                ax[i, j].contour(P[y0:y1, x0:x1].astype(np.uint8), [0.5], colors='cyan', linewidths=0.6)
            ax[i, j].set_title(f'{k} {cols[j]}' if j == 0 else cols[j], fontsize=7); ax[i, j].axis('off')
    fig.suptitle('magenta fill = mask in that column; cyan outline = your polygon (for reference)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{IMG}/envelope_vs_hand.png', dpi=75)


if __name__ == '__main__':
    main()
