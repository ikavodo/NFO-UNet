"""Is the polygon-minus-SAM2-ceiling gap thin? Granulometry of G = P ∖ M on seq1 polygon frames (2026-10-14).

Hypothesis (user): the human polygon P ≈ S ∧ ¬γ_r(O) (annotator removes thick occluders, draws through thin ones),
SAM2 SAM2-ceiling M ≈ S ∧ ¬O, so G = P ∖ M ≈ S ∧ (O ∖ γ_r(O)) consists of thin structures (thin branches).
γ_r = morphological opening with a disk of radius r. Pattern spectrum (Maragos, IEEE TPAMI 1989):
    f(r) = 1 − |γ_r(G)| / |G|      (fraction of the gap removed by an opening of radius r)
Pre-registered: hypothesis supported if f(3) ≥ 0.5 (structures ≤ ~7 px wide, < half a stride-16 cell).
Reported separately for gap pixels inside the convex hull of M (enclosed by the person: candidate thin occluders)
and outside it (polygon overshoot along the outline). Reference spectrum: M itself.
"""
import os

import cv2
import numpy as np

GT = 'data/nfo_processed/seq1_gt'
IMG = 'images/benchmark/rr_checks/gap_granulometry'
RADII = (1, 2, 3, 4, 6, 8, 12)


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def removed(G, r):
    return G & ~(cv2.morphologyEx(G.astype(np.uint8), cv2.MORPH_OPEN, disk(r)) > 0)


def hull(M):
    pts = cv2.findNonZero(M.astype(np.uint8))
    out = np.zeros(M.shape, np.uint8)
    if pts is not None:
        cv2.fillPoly(out, [cv2.convexHull(pts)], 1)
    return out > 0


def main():
    ks = sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith('_handpoly.png'))
    os.makedirs(IMG, exist_ok=True)
    tot = {n: np.zeros(len(RADII)) for n in ('gap', 'gap_in', 'gap_out', 'M')}
    area = {n: 0 for n in tot}
    per = []
    for k in ks:
        P = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1
        M = cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127
        G, Hm = P & ~M, hull(M)
        parts = dict(gap=G, gap_in=G & Hm, gap_out=G & ~Hm, M=M)
        for n, X in parts.items():
            area[n] += X.sum()
            tot[n] += [removed(X, r).sum() for r in RADII]
        per.append((k, G.sum(), removed(G, 3).sum() / max(G.sum(), 1), (G & Hm).sum() / max(G.sum(), 1)))
    print('pooled fraction removed by opening of radius r (pattern spectrum, cumulative):')
    print('                 ' + '  '.join(f'r={r:<3d}' for r in RADII) + '   pixels')
    for n in tot:
        print(f'  {n:14s} ' + '  '.join(f'{v:.3f}' for v in tot[n] / max(area[n], 1)) + f'   {area[n]}')
    f3 = tot['gap'][RADII.index(3)] / area['gap']
    print(f'pre-registered: f(3) = {f3:.3f} -> {"SUPPORTED" if f3 >= 0.5 else "NOT SUPPORTED"} (bar 0.5)')
    print('per frame: k, |G| px, f(3), share of G inside hull(M)')
    for k, g, f, i in per:
        print(f'  {k:5d} {g:6d}  {f:.2f}  {i:.2f}')

    # visual: gap pixels coloured by the opening radius that removes them (red r<=3, yellow 4-8, blue survives 8)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from benchmark.nfo_vos import velocity as V
    fig, ax = plt.subplots(2, (len(ks) + 1) // 2, figsize=(12, 6))
    for a_, k in zip(ax.ravel(), ks):
        P = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1
        M = cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127
        G = P & ~M
        ys, xs = np.nonzero(P | M)
        y0, y1, x0, x1 = max(0, ys.min() - 10), min(600, ys.max() + 10), max(0, xs.min() - 10), min(800, xs.max() + 10)
        a_.imshow(V.native(k, 'seq1')[y0:y1, x0:x1], cmap='gray')
        ov = np.zeros((y1 - y0, x1 - x0, 4))
        thin, mid = removed(G, 3), removed(G, 8)
        ov[(G & ~mid)[y0:y1, x0:x1]] = (0.2, 0.4, 1, 0.6)
        ov[(mid & ~thin)[y0:y1, x0:x1]] = (1, 1, 0, 0.6)
        ov[thin[y0:y1, x0:x1]] = (1, 0, 0, 0.7)
        a_.imshow(ov)
        a_.contour(M[y0:y1, x0:x1].astype(np.uint8), [0.5], colors='lime', linewidths=0.6)
        a_.set_title(str(k), fontsize=8); a_.axis('off')
    for a_ in ax.ravel()[len(ks):]:
        a_.axis('off')
    fig.suptitle('gap = polygon minus SAM2-ceiling (green outline). red: thin (removed by opening r<=3), '
                 'yellow: r 4-8, blue: thick (survives r=8)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{IMG}/gap_by_thickness.png', dpi=75)


if __name__ == '__main__':
    main()
