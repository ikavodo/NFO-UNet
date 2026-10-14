"""H2 (docs/label_generators_design.md:58-66, pre-registered 2026-10-13): does SAM2 drop sub-cell visible fragments,
or does our SAM2-ceiling filter? 2026-10-14.

U = raw SAM2 union (box-clipped, thin-component filter OFF): *_sammask_nofilt_native.png
M = filter(U): the same run with the 14 px width filter applied (isolates the filter; the stored SAM2-ceiling differs
    from the re-run by SAM2 run-to-run variation, see check 0)
0. Reproducibility: filter(U) == M (same code path, gen_nfo_pseudo_masks.py:277-281).
1. H2 proper, vs brush (modal) masks, seq1 seg4: per brush connected component C (8-conn), recall |C ∧ X| / |C|
   for X in {U, M}, against area in stride-16 cells (1 cell = 12.5 x 9.4 ≈ 117 native px²). Main body = largest
   component, detached = the rest. KILL: slope of recall on log2(area) has a cluster-bootstrap CI containing 0,
   or detached sub-cell components have recall ≥ 0.5.
2. Polygons (17 frames): J/P/R of U vs M; share of the thin enclosed gap (polygon ∖ M, inside hull(M), removed by an
   opening of radius 3; gap_granulometry.py) that U recovers.
"""
import os

import cv2
import numpy as np

from benchmark.nfo_vos.gap_granulometry import hull, removed

GT = 'data/nfo_processed/seq1_gt'
CELL = 800 / 64 * 600 / 64   # 117.2 native px²
MINW = round(4 * 800 / 224)  # 14
IMG = 'images/benchmark/rr_checks/h2'


def load(k, tag):
    p = f'{GT}/{k:05d}_{tag}.png'
    return cv2.imread(p, 0) > 127 if os.path.exists(p) else None


def width_filter(U):
    n, lab, st, _ = cv2.connectedComponentsWithStats(U.astype(np.uint8))
    return np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_WIDTH] >= MINW])


def stats(a, h):
    tp = (a & h).sum()
    return tp / max((a | h).sum(), 1), tp / max(a.sum(), 1), tp / max(h.sum(), 1)


def main():
    rng = np.random.default_rng(0)
    frames = lambda tag: sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith(f'_{tag}.png'))
    # 0. reproducibility
    ks_all = [k for k in frames('sammask_nofilt_native') if load(k, 'sammask_native') is not None]
    diff = [(width_filter(load(k, 'sammask_nofilt_native')) ^ load(k, 'sammask_native')).sum() for k in ks_all]
    print(f'0. filter(U) vs stored M on {len(ks_all)} frames: {sum(d == 0 for d in diff)} identical, '
          f'mean |xor| {np.mean(diff):.1f} px, max {max(diff)} px')

    # 1. H2 vs brush
    rows = []
    for k in frames('handmask_raw'):
        B, U, M = cv2.imread(f'{GT}/{k:05d}_handmask_raw.png', 0) == 1, load(k, 'sammask_nofilt_native'), None
        if U is None:
            continue
        M = width_filter(U)
        n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
        main_c = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        for i in range(1, n):
            C = lab == i
            rows.append((k, st[i, cv2.CC_STAT_AREA] / CELL, i != main_c, (C & U).sum() / C.sum(), (C & M).sum() / C.sum()))
    R = np.array(rows, dtype=float)
    print(f'\n1. H2 vs brush: {len(np.unique(R[:, 0]))} frames, {len(R)} components '
          f'({int(R[:, 2].sum())} detached)')
    bins = [(0, 0.5), (0.5, 1), (1, 4), (4, 1e9)]
    print('   area (cells)   n    recall U (filter off)   recall M (SAM2-ceiling)   [detached only: n, U, M]')
    for lo, hi in bins:
        m = (R[:, 1] >= lo) & (R[:, 1] < hi); d = m & (R[:, 2] == 1)
        print(f'   [{lo:g}, {hi:g})'.ljust(16) + f'{m.sum():4d}    {R[m, 3].mean():.2f}                   {R[m, 4].mean():.2f}'
              f'                   [{d.sum()}, {R[d, 3].mean() if d.any() else np.nan:.2f}, {R[d, 4].mean() if d.any() else np.nan:.2f}]')
    ks = np.unique(R[:, 0])
    for col, name in ((3, 'U'), (4, 'M')):
        slope = lambda X: np.polyfit(np.log2(X[:, 1]), X[:, col], 1)[0]
        bs = []
        for _ in range(5000):
            X = np.vstack([R[R[:, 0] == k] for k in rng.choice(ks, len(ks))])
            bs.append(slope(X))
        sub = (R[:, 2] == 1) & (R[:, 1] < 1)
        print(f'   {name}: slope of recall per doubling of area {slope(R):+.3f} '
              f'[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]; detached sub-cell recall '
              f'{R[sub, col].mean():.2f} (n={sub.sum()})')

    # 2. polygons
    print('\n2. vs polygons')
    P_, thin_rec, tot = {'U': [], 'M': []}, [], [0, 0]
    for k in frames('handpoly'):
        P, U, M = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1, load(k, 'sammask_nofilt_native'), None
        if U is None:
            continue
        M = width_filter(U)
        P_['U'].append(stats(U, P)); P_['M'].append(stats(M, P))
        Ms = load(k, 'sammask_native')   # gap defined on the stored SAM2-ceiling, as in gap_granulometry.py
        thin = removed(P & ~Ms & hull(Ms), 3)
        tot[0] += (thin & U).sum(); tot[1] += thin.sum()
    for n in P_:
        print(f'   {n}: J/P/R ' + ' / '.join(f'{v:.3f}' for v in np.mean(P_[n], 0)) + f'  (n={len(P_[n])})')
    print(f'   thin enclosed gap recovered by U: {tot[0] / max(tot[1], 1):.2f} of {tot[1]} px')

    # visual: what the filter removed (U ∖ M, magenta) vs polygon (cyan), on polygon frames
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from benchmark.nfo_vos import velocity as V
    kp = [k for k in frames('handpoly') if load(k, 'sammask_nofilt_native') is not None]
    os.makedirs(IMG, exist_ok=True)
    fig, ax = plt.subplots(2, (len(kp) + 1) // 2, figsize=(12, 6))
    for a_, k in zip(ax.ravel(), kp):
        P, U = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1, load(k, 'sammask_nofilt_native')
        M = width_filter(U)
        ys, xs = np.nonzero(P | U)
        y0, y1, x0, x1 = max(0, ys.min() - 10), min(600, ys.max() + 10), max(0, xs.min() - 10), min(800, xs.max() + 10)
        a_.imshow(V.native(k, 'seq1')[y0:y1, x0:x1], cmap='gray')
        ov = np.zeros((y1 - y0, x1 - x0, 4)); ov[(U & ~M)[y0:y1, x0:x1]] = (1, 0, 1, 0.7)
        a_.imshow(ov)
        for mm, col in ((P, 'cyan'), (M, 'lime')):
            a_.contour(mm[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=col, linewidths=0.6)
        a_.set_title(str(k), fontsize=8); a_.axis('off')
    for a_ in ax.ravel()[len(kp):]:
        a_.axis('off')
    fig.suptitle('cyan = polygon, green = SAM2-ceiling (filter on), magenta = SAM2 pixels the 14 px width filter removed',
                 fontsize=9)
    fig.tight_layout(); fig.savefig(f'{IMG}/filter_removed.png', dpi=75)


if __name__ == '__main__':
    main()
