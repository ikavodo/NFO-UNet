"""Rolling walker polygon fitted to the GT box, minus "near" empty-scene depth, vs the hand masks of seq1
segment 4. 2026-10-13      python -m benchmark.nfo_vos.rolling_polygon

Articulated 2-D body ("cardboard people", Ju, Black & Yacoob, FG 1996), side view, in box units (H = box height):
head ellipse, torso trapezoid (shoulders -> hips), two leg strips hip -> foot, two arm strips shoulder -> hand.
Pose per frame from the box alone: feet at the box's left/right edges (the box widens with the stride), hands
swing half as far. Proportions (torso width, hip height, limb thickness) are grid-fitted to the SAM2-ceiling of
confirmed-clear frames of the OTHER seq1 segments (fully visible person). Visible = polygon AND disparity <= theta.
"""
import glob
import itertools
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

SEQ, HELD = 'seq1', 4
DEPTH = 'results/benchmark/rr/depth/seq1_bg_disp_1036.npy'
IMG = 'images/benchmark/rr_checks/hand'
GRID = dict(torso=(0.2, 0.24, 0.28, 0.32, 0.36), hip=(0.54, 0.58, 0.62, 0.66), leg=(0.07, 0.09, 0.11, 0.13, 0.15), arm=(0.03, 0.05, 0.07))


def box(k):
    return tuple(int(round(v)) for v in gt_to_native(TR.bbs(SEQ)[k][0], 800, 600))


def seg_of(k):
    return next(s for s in TR.segments(SEQ) if s['start'] <= k <= s['end'])


def walker(b, d, torso, hip, leg, arm):
    """Binary mask [600,800] of the walker in box b=(x0,y0,x1,y1); d = +1 walking right, -1 left."""
    x0, y0, x1, y1 = b
    H, cx = y1 - y0, (x0 + x1) / 2
    m = np.zeros((600, 800), np.uint8)
    P = lambda x, y: (int(round(x)), int(round(y)))
    wt, yh, ys = torso * H, y0 + hip * H, y0 + 0.15 * H
    cv2.ellipse(m, P(cx + d * 0.02 * H, y0 + 0.065 * H), P(0.05 * H, 0.065 * H), 0, 0, 360, 1, -1)
    cv2.fillPoly(m, [np.array([P(cx - wt / 2, ys), P(cx + wt / 2, ys), P(cx + 0.4 * wt, yh), P(cx - 0.4 * wt, yh)])], 1)
    lt, at = max(1, int(leg * H)), max(1, int(arm * H))
    for s in (-1, 1):
        cv2.line(m, P(cx + s * 0.2 * wt, yh), P((x0 + lt / 2) if s < 0 else (x1 - lt / 2), y1 - lt / 2), 1, lt)
        cv2.line(m, P(cx + s * 0.3 * wt, ys + 0.02 * H), P(cx + s * 0.5 * (x1 - x0) / 2, y0 + 0.5 * H), 1, at)
    return m.astype(bool)


def fit_proportions():
    J = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    F = [(k, s['dir']) for s in TR.segments(SEQ) if s['idx'] != HELD for k in TR.clear_frames(s)]
    F = [(k, d) for k, d in F if gt_mask_native(SEQ, k) is not None][::2]
    best = None
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID, vals))
        j = np.mean([J(walker(box(k), 1 if d == 'R' else -1, **p), gt_mask_native(SEQ, k)) for k, d in F])
        best = max(best or (j, p), (j, p), key=lambda t: t[0])
    print(f'proportions fitted on {len(F)} clear frames (other segments): {best[1]}, mean J vs SAM2-ceiling {best[0]:.3f}')
    return best[1], F


def main():
    p, F = fit_proportions()
    d = np.load(DEPTH)
    J = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    st = lambda a, h: (J(a, h), (a & h).sum() / max(a.sum(), 1), (a & h).sum() / max(h.sum(), 1))
    W = lambda k: walker(box(k), 1 if seg_of(k)['dir'] == 'R' else -1, **p)
    for name, tag in (('brush', '_handmask_raw'), ('polygon', '_handpoly')):
        H = {int(os.path.basename(f)[:5]): cv2.imread(f, 0) == 1 for f in sorted(glob.glob(f'data/nfo_processed/{SEQ}_gt/*{tag}.png'))}
        print(f'--- vs your {name} masks ({len(H)} frames): mean J / precision / recall')
        cands = {'SAM2-ceiling': lambda k: gt_mask_native(SEQ, k), 'walker': W}
        for th in (0.1, 0.2, 0.3):
            cands[f'walker - depth>{th}'] = lambda k, th=th: W(k) & (d <= th)
        for cn, f in cands.items():
            print(f'  {cn:20s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([st(f(k), H[k]) for k in H], 0)))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    clear = [F[i] for i in np.linspace(0, len(F) - 1, 3).astype(int)]
    hand = [1154, 1183, 1214, 1236, 1263, 1292]
    fig, ax = plt.subplots(2, 6, figsize=(12, 7))
    for a_, (k, dd) in itertools.zip_longest(ax[0], clear, fillvalue=(None, None)):
        a_.axis('off')
        if k is None:
            continue
        crop(a_, k)(gt_mask_native(SEQ, k), 'red')(walker(box(k), 1 if dd == 'R' else -1, **p), 'yellow')
        a_.set_title(f'clear {k} (fit set)')
    for a_, k in zip(ax[1], hand):
        a_.axis('off')
        c = crop(a_, k)
        c(cv2.imread(f'data/nfo_processed/{SEQ}_gt/{k:05d}_handmask_raw.png', 0) == 1, 'lime')
        poly = f'data/nfo_processed/{SEQ}_gt/{k:05d}_handpoly.png'
        if os.path.exists(poly):
            c(cv2.imread(poly, 0) == 1, 'cyan')
        c(W(k), 'yellow')(W(k) & (d <= 0.2), 'magenta')
        a_.set_title(str(k))
    fig.suptitle('top: clear frames, red = SAM2-ceiling, yellow = walker | bottom: green = your brush, cyan = your polygon,\n'
                 'yellow = walker polygon, magenta = walker minus depth>0.2', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{IMG}/rolling_polygon.png', dpi=80)
    print(f'-> {IMG}/rolling_polygon.png')


def crop(a_, k):
    x0, y0, x1, y1 = box(k)
    x0, y0, x1, y1 = max(0, x0 - 10), max(0, y0 - 10), min(800, x1 + 10), min(600, y1 + 10)
    a_.imshow(V.native(k, SEQ)[y0:y1, x0:x1], cmap='gray')

    def draw(m, col):
        a_.contour(m[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=col, linewidths=0.8)
        return draw
    return draw


if __name__ == '__main__':
    main()
