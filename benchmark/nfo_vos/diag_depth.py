"""Quick feasibility check: does empty-scene depth mark where the person CAN be seen (foliage gaps)? 2026-10-13

    python -m benchmark.nfo_vos.diag_depth      # seq1 segment 4, every 5th frame; regions: GT box, and
                                                # convex hull of the SAM2-ceiling (oracle silhouette proxy) in the box;
                                                # depth input short side 518 (default) and 1036

Labels: native SAM2-ceiling (visible = 1, rest of the box = 0); hand masks replace them later.
Scores (AUC for visible vs not):
  far    -delta_bg(x): empty-scene disparity (Depth Anything V2 Large, larger = nearer); a gap shows far stuff
  change delta_t(x) - delta_bg(x), delta_t affine-aligned to delta_bg on pixels outside the box (label-free)
  z      empty-scene feature cue from static_gate (frames 1154-1273 only), and z + far (logistic,
         train even / test odd frames) -> does depth add to what we already have?
"""
import json
import os

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

MODEL = 'depth-anything/Depth-Anything-V2-Large-hf'
SEQ, WARM, K = 'seq1', (946, 1113), range(1154, 1309, 5)
IMG = 'images/benchmark/rr_checks/depth'


def depth_fn(short=518):
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = AutoModelForDepthEstimation.from_pretrained(MODEL).cuda().eval()

    @torch.inference_mode()
    def f(gray):
        x = proc(images=cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB), return_tensors='pt',
                 size={'height': short, 'width': short})
        d = net(**{k: v.cuda() for k, v in x.items()}).predicted_depth[None]
        return torch.nn.functional.interpolate(d, (600, 800), mode='bilinear')[0, 0].cpu().numpy()
    return f


def box(k):
    bb = TR.bbs(SEQ).get(k)
    if not bb or bb[0].x < 0:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in gt_to_native(bb[0], 800, 600))
    m = np.zeros((600, 800), bool); m[max(0, y0):y1, max(0, x0):x1] = True
    return m


def hull(g):
    h = np.zeros((600, 800), np.uint8)
    cv2.fillPoly(h, [cv2.convexHull(cv2.findNonZero(g.astype(np.uint8)))], 1)
    return h.astype(bool)


def main():
    os.makedirs(IMG, exist_ok=True)
    for short in (518, 1036):
        for region in ('box', 'hull'):
            print(f'--- depth input short side {short}, region {region}')
            run(depth_fn(short), region, fig=short == 1036 and region == 'hull')


def run(depth, region, fig):
    bg = np.median(np.stack([V.native(k, SEQ) for k in range(WARM[0], WARM[1] + 1)]), 0)
    dbg = depth(bg)
    trials = {t['frames'][0]: t for t in json.load(open(TR.OUT)) if t['seq'] == SEQ}
    rows, union, ex = [], np.zeros((600, 800), bool), None
    for k in K:
        b, g = box(k), gt_mask_native(SEQ, k)
        if b is None or g is None or not g.any():
            continue
        b = b & hull(g) if region == 'hull' else b
        dt = depth(V.native(k, SEQ))
        A = np.c_[dt[~b], np.ones((~b).sum())]
        a, c = np.linalg.lstsq(A, dbg[~b], rcond=None)[0]
        change = a * dt + c - dbg
        t0 = max([s for s in trials if s <= k <= trials[s]['frames'][-1]], default=None)
        z = None
        if t0 is not None:
            zz = np.load(f"{TR.RES}/static_z/{trials[t0]['id']}.npz")['z'][trials[t0]['frames'].index(k)]
            z = cv2.resize(zz.astype(np.float32), (800, 600), interpolation=cv2.INTER_LINEAR)
        rows.append(dict(k=k, y=g[b], far=-dbg[b], change=change[b], z=None if z is None else z[b]))
        union |= g
        if ex is None or g.sum() > ex[1].sum():
            ex = (k, g, change, b)

    cat = lambda key, rs: np.concatenate([r[key] for r in rs])
    y = cat('y', rows)
    print(f'{len(rows)} frames, {y.size} box pixels, visible {y.mean():.2f}')
    for key in ('far', 'change'):
        print(f'AUC {key:7s} {roc_auc_score(y, cat(key, rows)):.3f}')
    zr = [r for r in rows if r['z'] is not None]
    tr, te = zr[0::2], zr[1::2]
    X = lambda rs, keys: np.c_[tuple(cat(q, rs) for q in keys)]
    for keys in (('z',), ('z', 'far'), ('z', 'far', 'change')):
        lr = LogisticRegression(max_iter=1000).fit(X(tr, keys), cat('y', tr))
        print(f'held-out frames ({len(te)}), logistic {"+".join(keys):13s} AUC '
              f'{roc_auc_score(cat("y", te), lr.decision_function(X(te, keys))):.3f}')

    if not fig:
        return
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    k, g, change, b = ex
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    ax[0, 0].imshow(bg, cmap='gray'); ax[0, 0].set_title('empty scene (warm-up median)')
    im = ax[0, 1].imshow(dbg, cmap='magma'); ax[0, 1].set_title('empty-scene disparity δ_bg (bright = near)')
    fig.colorbar(im, ax=ax[0, 1], fraction=0.035)
    ax[1, 0].imshow(dbg, cmap='magma'); ax[1, 0].contour(union, [0.5], colors='cyan', linewidths=0.5)
    ax[1, 0].set_title('δ_bg + union of seg-4 SAM2-ceiling (cyan)')
    im = ax[1, 1].imshow(np.where(b, change, np.nan), cmap='coolwarm', vmin=-np.nanpercentile(abs(change[b]), 99),
                         vmax=np.nanpercentile(abs(change[b]), 99))
    ax[1, 1].contour(g, [0.5], colors='k', linewidths=0.5); fig.colorbar(im, ax=ax[1, 1], fraction=0.035)
    ax[1, 1].set_title(f'frame {k}: δ_t − δ_bg in hull (red = nearer now), black = SAM2-ceiling')
    for a_ in ax.flat:
        a_.axis('off')
    fig.tight_layout(); fig.savefig(f'{IMG}/seg4_depth_1036.png', dpi=80)
    print(f'-> {IMG}/seg4_depth.png')


if __name__ == '__main__':
    main()
