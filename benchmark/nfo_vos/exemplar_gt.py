"""Exemplar silhouettes retrieved by partial evidence, blended by signed distance, minus near depth; vs the hand
masks of seq1 segment 4. 2026-10-13      python -m benchmark.nfo_vos.exemplar_gt

Database: pseudo-GT of seq1's confirmed-clear frames (fully visible, same person), cropped to the GT box (mirrored
for leftward segments); frames within +-EXCL of the target are excluded.
Evidence in frame t:  P = pseudo-GT (visible person);  N = "background seen through a gap": empty-scene disparity
< N_DISP (far) AND |frame - empty scene| < N_DIFF (unchanged) -> the person is not there.
Retrieval: exemplars with box aspect within ASP of t's (the box width follows the stride), resized into t's box,
shifted by dx in SHIFTS * box width; score = |S n P| / |P| - |S n N| / |S|. Blend the TOP best by averaging signed
distance maps (shape interpolation), threshold 0.  Labels: S, S u P, (S and disparity <= theta) u P.
"""
import glob
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

SEQ, WARM, EXCL, ASP, TOP = 'seq1', (946, 1113), 10, 0.15, 3
SHIFTS = np.linspace(-0.1, 0.1, 5)
N_DISP, N_DIFF = 0.1, 8
DEPTH = 'results/benchmark/rr/depth/seq1_bg_disp_1036.npy'
IMG = 'images/benchmark/rr_checks/hand'


def box(k):
    return tuple(int(round(v)) for v in gt_to_native(TR.bbs(SEQ)[k][0], 800, 600))


def seg_of(k):
    return next(s for s in TR.segments(SEQ) if s['start'] <= k <= s['end'])


def database():
    E = []
    for s in TR.segments(SEQ):
        for k in TR.clear_frames(s):
            m = gt_mask_native(SEQ, k)
            x0, y0, x1, y1 = box(k)
            if m is None or x0 < 0 or y0 < 0 or x1 > 800 or y1 > 600:
                continue
            c = m[y0:y1, x0:x1].astype(np.float32)
            E.append((k, (x1 - x0) / (y1 - y0), c[:, ::-1] if s['dir'] == 'L' else c))
    return E


def place(c, b, dx):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    big = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    out = np.zeros((600, 800), bool)
    x0 += int(round(dx * w))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(800, x0 + w), min(600, y1)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0:cy1, cx0:cx1] = big[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


def sdf(m):
    m = m.astype(np.uint8)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)


def silhouette(k, E, bg, d):
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
    best = sorted(scored, key=lambda t: -t[0])[:TOP]
    return np.mean([sdf(S) for _, S in best], 0) > 0, P, N


def main():
    E = database()
    d = np.load(DEPTH)
    bg = np.median(np.stack([V.native(k, SEQ) for k in range(WARM[0], WARM[1] + 1)]), 0).astype(np.float32)
    print(f'{len(E)} clear exemplars')
    J = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
    st = lambda a, h: (J(a, h), (a & h).sum() / max(a.sum(), 1), (a & h).sum() / max(h.sum(), 1))
    cache = {}
    for name, tag in (('brush', '_handmask_raw'), ('polygon', '_handpoly')):
        H = {int(os.path.basename(f)[:5]): cv2.imread(f, 0) == 1 for f in sorted(glob.glob(f'data/nfo_processed/{SEQ}_gt/*{tag}.png'))}
        for k in H:
            cache.setdefault(k, silhouette(k, E, bg, d))
        cands = {'pseudo-GT': lambda k: cache[k][1], 'silhouette S': lambda k: cache[k][0],
                 'S u P': lambda k: cache[k][0] | cache[k][1]}
        for th in (0.2, 0.3):
            cands[f'(S - depth>{th}) u P'] = lambda k, th=th: (cache[k][0] & (d <= th)) | cache[k][1]
        print(f'--- vs your {name} masks ({len(H)} frames): mean J / precision / recall')
        for cn, f in cands.items():
            print(f'  {cn:22s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([st(f(k), H[k]) for k in H], 0)))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = [1154, 1183, 1214, 1236, 1263, 1292]
    fig, ax = plt.subplots(2, len(ks), figsize=(12, 7))
    for j, k in enumerate(ks):
        S, P, N = cache[k]
        x0, y0, x1, y1 = box(k)
        x0, y0, x1, y1 = max(0, x0 - 10), max(0, y0 - 10), min(800, x1 + 10), min(600, y1 + 10)
        c = lambda m: m[y0:y1, x0:x1].astype(np.uint8)
        f = V.native(k, SEQ)[y0:y1, x0:x1]
        rgb = np.dstack([f] * 3).astype(float) / 255
        rgb[c(N) > 0] = 0.5 * rgb[c(N) > 0] + 0.5 * np.array([0, 0.3, 1]); rgb[c(P) > 0] = (1, 0, 0)
        ax[0, j].imshow(rgb); ax[0, j].contour(c(S), [0.5], colors='yellow', linewidths=0.8)
        ax[0, j].set_title(f'{k} evidence')
        ax[1, j].imshow(f, cmap='gray')
        ax[1, j].contour(c(cv2.imread(f'data/nfo_processed/{SEQ}_gt/{k:05d}_handmask_raw.png', 0) == 1), [0.5], colors='lime', linewidths=0.8)
        poly = f'data/nfo_processed/{SEQ}_gt/{k:05d}_handpoly.png'
        if os.path.exists(poly):
            ax[1, j].contour(c(cv2.imread(poly, 0) == 1), [0.5], colors='cyan', linewidths=0.8)
        ax[1, j].contour(c(S), [0.5], colors='yellow', linewidths=0.8)
        ax[1, j].set_title(str(k))
    for a_ in ax.flat:
        a_.axis('off')
    fig.suptitle('top: red = pseudo-GT (P), blue = background seen through a gap (N), yellow = retrieved silhouette\n'
                 'bottom: green = your brush, cyan = your polygon, yellow = retrieved silhouette', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{IMG}/exemplar_silhouette.png', dpi=80)
    print(f'-> {IMG}/exemplar_silhouette.png')


if __name__ == '__main__':
    main()
