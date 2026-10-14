"""Shape-template labels: a mean person silhouette fitted into each GT box, optionally minus "near" empty-scene
depth; compared with the hand masks of seq1 segment 4. 2026-10-13

    python -m benchmark.nfo_vos.template_gt

Template for frame t: SAM2-ceiling masks of seq1's confirmed-clear frames (person fully visible), each cropped to its
GT box and resized to a canonical grid (mirrored for leftward segments); average the K exemplars whose box aspect
w/h is closest to t's (the box widens with the stride), resize into t's box, threshold 0.5.
Exemplars within +-EXCL frames of t are excluded.  Depth variant: drop pixels with empty-scene normalised disparity
> theta (results/benchmark/rr/depth/seq1_bg_disp_1036.npy, Depth Anything V2 L).
"""
import glob
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

SEQ, GW, GH, K, EXCL = 'seq1', 48, 128, 8, 10
DEPTH = 'results/benchmark/rr/depth/seq1_bg_disp_1036.npy'
IMG = 'images/benchmark/rr_checks/hand'


def box(k):
    return tuple(int(round(v)) for v in gt_to_native(TR.bbs(SEQ)[k][0], 800, 600))


def seg_of(k):
    return next(s for s in TR.segments(SEQ) if s['start'] <= k <= s['end'])


def exemplars():
    E = []
    for s in TR.segments(SEQ):
        for k in TR.clear_frames(s):
            m = gt_mask_native(SEQ, k)
            x0, y0, x1, y1 = box(k)
            if m is None or x0 < 0 or y0 < 0 or x1 > 800 or y1 > 600:
                continue
            c = cv2.resize(m[y0:y1, x0:x1].astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
            E.append((k, (x1 - x0) / (y1 - y0), c[:, ::-1] if s['dir'] == 'L' else c))
    return E


def fit(k, E):
    x0, y0, x1, y1 = box(k)
    a = (x1 - x0) / (y1 - y0)
    near = sorted((e for e in E if abs(e[0] - k) > EXCL), key=lambda e: abs(e[1] - a))[:K]
    T = np.mean([e[2] for e in near], 0)
    T = T[:, ::-1] if seg_of(k)['dir'] == 'L' else T
    out = np.zeros((600, 800), np.float32)
    big = cv2.resize(T, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(800, x1), min(600, y1)
    out[cy0:cy1, cx0:cx1] = big[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out > 0.5


def main():
    E = exemplars()
    print(f'{len(E)} clear exemplars in {SEQ}')
    d = np.load(DEPTH)
    J = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    st = lambda a, h: (J(a, h), (a & h).sum() / max(a.sum(), 1), (a & h).sum() / max(h.sum(), 1))
    labels = {'brush': '_handmask_raw', 'polygon': '_handpoly'}
    fits = {}
    for name, tag in labels.items():
        H = {int(os.path.basename(p)[:5]): cv2.imread(p, 0) == 1 for p in sorted(glob.glob(f'data/nfo_processed/{SEQ}_gt/*{tag}.png'))}
        print(f'--- vs your {name} masks ({len(H)} frames): mean J / precision / recall')
        cands = {'SAM2-ceiling': lambda k: gt_mask_native(SEQ, k), 'template': lambda k: fits[k]}
        for th in (0.1, 0.2, 0.3):
            cands[f'template - depth>{th}'] = lambda k, th=th: fits[k] & (d <= th)
        for k in H:
            fits.setdefault(k, fit(k, E))
        for cn, f in cands.items():
            print(f'  {cn:22s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([st(f(k), H[k]) for k in H], 0)))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = [1154, 1183, 1214, 1236, 1263, 1292]
    fig, ax = plt.subplots(1, len(ks), figsize=(12, 4))
    for a_, k in zip(ax, ks):
        x0, y0, x1, y1 = box(k)
        x0, y0, x1, y1 = max(0, x0 - 10), max(0, y0 - 10), min(800, x1 + 10), min(600, y1 + 10)
        c = lambda m: m[y0:y1, x0:x1].astype(np.uint8)
        a_.imshow(V.native(k, SEQ)[y0:y1, x0:x1], cmap='gray')
        for m, col in ((cv2.imread(f'data/nfo_processed/{SEQ}_gt/{k:05d}_handmask_raw.png', 0) == 1, 'lime'),
                       (gt_mask_native(SEQ, k), 'red'), (fits[k], 'yellow'), (fits[k] & (d <= 0.2), 'magenta')):
            a_.contour(c(m), [0.5], colors=col, linewidths=0.8)
        a_.set_title(str(k)); a_.axis('off')
    fig.suptitle('green = your brush mask, red = SAM2-ceiling, yellow = template fit, magenta = template minus depth>0.2')
    fig.tight_layout(); fig.savefig(f'{IMG}/template_fit.png', dpi=80)
    print(f'-> {IMG}/template_fit.png')


if __name__ == '__main__':
    main()
