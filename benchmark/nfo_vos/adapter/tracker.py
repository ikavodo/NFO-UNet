"""Kernel-f16 person tracker, used by training (fusion_data.build), the NFO hand-frame inputs (fusion_nfo.items) and
inference (inference.fast_masks).

A fixed-size person kernel (w0 × h0 of the t0 box) is searched in a window around a Theil–Sen constant-velocity
prediction (x and y, last ≤ 8 estimates); score = mean of the foreground map inside the kernel (integral image), i.e. a
matched filter for a box-shaped person (mean-shift-style fixed-kernel tracking, Comaniciu et al., CVPR 2000). Map:
1 − cos(F_t, F_B) of SAM2.1 b+ stride-16 features of frame t and the empty scene B. Chosen over the MOG2 blob tracker
on 2026-10-16 (61 NFO windows; comparison script at tag archive/pre-cleanup-2026-10-21, layers/track_check.py).
"""
import os

import cv2
import numpy as np
import torch
from scipy.stats import theilslopes

from benchmark.nfo_vos import trials as TR
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

BATCH = int(os.environ.get('TC_BATCH', 1))


def gt_box(seq, k):
    bb = TR.bbs(seq).get(k)
    if not bb or bb[0].x < 0:
        return None
    return np.array(gt_to_native(bb[0], 800, 600), float)


def kernel_track(maps, c0, w0, h0):
    """maps: [T, H, W] float; c0 = t0 centre. Returns [T, 2] centres."""
    T, H, W = maps.shape
    out = [np.array(c0, float)]
    hw, hh = int(round(w0 / 2)), int(round(h0 / 2))
    for t in range(1, T):
        P = np.array(out[-8:])
        if len(P) >= 3:
            ts = np.arange(len(P))
            pred = np.array([theilslopes(P[:, i], ts)[0] * len(P) + theilslopes(P[:, i], ts)[1] for i in (0, 1)])
            v = np.abs(pred - P[-1])
        else:
            pred, v = P[-1], np.zeros(2)
        rx, ry = int(max(0.5 * w0, 3 * v[0] + 20)), int(max(0.25 * h0, 3 * v[1] + 10))
        I = cv2.integral(maps[t].astype(np.float64))
        best, bc = -np.inf, pred
        for cy in range(int(pred[1]) - ry, int(pred[1]) + ry + 1, 2):
            y0, y1 = np.clip([cy - hh, cy + hh], 0, H)
            if y1 <= y0:
                continue
            xs = np.arange(int(pred[0]) - rx, int(pred[0]) + rx + 1, 2)
            x0, x1 = np.clip(xs - hw, 0, W), np.clip(xs + hw, 0, W)
            area = np.maximum((x1 - x0) * (y1 - y0), 1)
            s = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / area
            i = int(np.argmax(s))
            if s[i] > best:
                best, bc = s[i], np.array([xs[i], cy], float)
        out.append(bc)
    return np.array(out)


def feat_maps(pred, frames, bg):
    """[T, H, W] foreground maps 1 − cos(F_t, F_B) on SAM2.1 b+ image_embed (stride 16), bilinear to native."""
    up = lambda c: torch.nn.functional.interpolate(c[None, None], size=frames.shape[1:3], mode='bilinear', align_corners=False)[0, 0]
    rgb = lambda x: np.stack([np.clip(x, 0, 255).astype(np.uint8)] * 3, -1)
    with torch.inference_mode():
        pred.set_image(rgb(bg))
        B16 = pred._features['image_embed'][0].float()
        m16 = []
        for i in range(0, len(frames), BATCH):                 # BATCH 1: shares the GPU with a training run
            pred.set_image_batch([rgb(x) for x in frames[i:i + BATCH]])
            for a in pred._features['image_embed'].float():
                m16.append(up(1 - torch.nn.functional.cosine_similarity(a, B16, dim=0)).cpu().numpy())
    return np.stack(m16)


def kernel16_track(t, pred):
    """Kernel-f16 track of one NFO window from its t0 GT box: ([T, 2] native centres, (w, h) of the t0 box)."""
    seq, fr = t['seq'], t['frames']
    g = gt_box(seq, fr[0])
    w0_, w1_ = t['warmup']
    bg = np.median(np.stack([TR.native(i, seq) for i in range(w0_, w1_ + 1)]), 0).astype(np.float32)
    m16 = feat_maps(pred, np.stack([TR.native(k, seq).astype(np.float32) for k in fr]), bg)
    c = kernel_track(m16, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), g[2] - g[0], g[3] - g[1])
    return c, (g[2] - g[0], g[3] - g[1])
