"""Static-layer gate on T1 (SAMURAI) masks: keep a mask pixel only where its stride-16 cell does NOT look
like the empty scene, in frozen SAM2 feature space (layered-memory plan, option C, NFO variant). 2026-10-12

    python -m benchmark.nfo_vos.static_gate cache    # GPU: per-frame z maps -> results/benchmark/rr/static_z/<trial>.npz
    python -m benchmark.nfo_vos.static_gate tune [feat|pix]          # CPU: theta on seq1 windows only (mean J)
    python -m benchmark.nfo_vos.static_gate write <theta> <theta_pix>  # masks/t1-gate, t1-gate-pix, t1-erode

Controls (same T1 masks, output-only):
  t1-gate-pix  the same gate on PIXELS: z of |frame - warm-up background| (3x3 mean), same per-frame
               median/MAD normalisation, own theta tuned on seq1 -> does feature space matter?
  t1-erode     area-matched erosion: per frame, peel T1's mask from its boundary (distance transform)
               until it has the area of the feature-gated mask -> is it WHICH pixels, or just fewer?

z_t(x) = (median_x c - c(x)) / (1.4826 MAD_x c), c = cos(f_t(x), S_bg(x)): f_t = SAM2.1 b+ image-encoder
stride-16 embedding of frame t (gray -> 3 channels, as diag_composite_diff), S_bg = the same for the median
of the window's person-free warm-up frames. Gate: M'_t = M_t AND (z_t upsampled to native > theta).
Output-only: T1's memory is unchanged (the gated mask is not fed back).
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import metrics, trials as TR, velocity as V
from benchmark.nfo_vos.diag_composite_diff import feats, static_z
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

ZDIR = f'{TR.RES}/static_z'
THETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


def trials():
    return [t for t in json.load(open(TR.OUT)) if t['admissible']]


def cache():
    os.makedirs(ZDIR, exist_ok=True)
    with torch.inference_mode():
        for i, t in enumerate(trials()):
            path = f"{ZDIR}/{t['id']}.npz"
            if os.path.exists(path):
                continue
            w0, w1 = t['warmup']
            bg = np.median(np.stack([V.native(k, t['seq']) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
            Sbg = feats(bg)[16]
            Z = torch.stack([static_z(feats(V.native(k, t['seq']).astype(np.float32))[16], Sbg) for k in t['frames']])
            np.savez_compressed(path, z=Z.cpu().numpy().astype(np.float16))
            print(f"{i + 1} {t['id']}", flush=True)


def gate(masks, z, theta):
    """masks [T,600,800] bool, z [T,64,64] (1024x1024 input stretched from 800x600) or [T,600,800] (pixels)."""
    up = z if z.shape[-2:] == masks.shape[-2:] else \
        np.stack([cv2.resize(zi.astype(np.float32), (800, 600), interpolation=cv2.INTER_LINEAR) for zi in z])
    return masks & (up > theta)


def pixel_z(t):
    """[T,600,800] z of |frame - warm-up median background| (3x3 mean), per-frame median/MAD."""
    w0, w1 = t['warmup']
    bg = np.median(np.stack([V.native(k, t['seq']) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
    out = []
    for k in t['frames']:
        d = cv2.blur(np.abs(V.native(k, t['seq']).astype(np.float32) - bg), (3, 3))
        med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med))
        out.append((d - med) / max(mad, 1e-6))
    return np.stack(out).astype(np.float32)


def erode_to(masks, target):
    """Peel each mask from its boundary until its area is that of target (same frame)."""
    out = np.zeros_like(masks)
    for i, (m, tg) in enumerate(zip(masks, target)):
        n = int(tg.sum())
        if n == 0 or not m.any():
            continue
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        thr = np.sort(dt[m])[::-1][min(n, int(m.sum())) - 1]
        out[i] = m & (dt >= thr)
    return out


def tune(kind='feat'):
    res = {th: [] for th in [np.inf] + THETAS}
    for t in [t for t in trials() if t['seq'] == 'seq1']:
        M = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks']
        z = np.load(f"{ZDIR}/{t['id']}.npz")['z'] if kind == 'feat' else pixel_z(t)
        G = [metrics.native_to_224(gt_mask_native(t['seq'], k)) for k in t['frames']]
        for th in res:
            Mg = M if th == np.inf else gate(M, z, th)
            res[th] += [metrics.frame_metrics(metrics.native_to_224(Mg[k]), G[k])['J']
                        for k in range(1, len(t['frames']) - 1) if G[k].any()]
    for th, J in res.items():
        print(f"theta {'none (T1)' if th == np.inf else th}: mean J {np.mean(J):.4f}  (n frames {len(J)})")
    best = max(THETAS, key=lambda th: np.mean(res[th]))
    print(f'best theta on seq1 ({kind}): {best}')


def write(theta, theta_pix):
    for name in ('t1-gate', 't1-gate-pix', 't1-erode'):
        os.makedirs(f'{TR.RES}/masks/{name}', exist_ok=True)
    for t in trials():
        src = dict(np.load(f"{TR.RES}/masks/t1/{t['id']}.npz"))
        M = src['masks']
        G = gate(M, np.load(f"{ZDIR}/{t['id']}.npz")['z'], theta)
        for name, masks, th in (('t1-gate', G, theta), ('t1-gate-pix', gate(M, pixel_z(t), theta_pix), theta_pix),
                                ('t1-erode', erode_to(M, G), theta)):
            np.savez_compressed(f"{TR.RES}/masks/{name}/{t['id']}.npz", **{**src, 'masks': masks, 'gate_theta': np.float32(th)})
        print(t['id'], flush=True)
    print(f'-> masks/t1-gate (theta {theta}), t1-gate-pix (theta {theta_pix}), t1-erode')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'cache':
        cache()
    elif cmd == 'tune':
        tune(*sys.argv[2:])
    else:
        write(float(sys.argv[2]), float(sys.argv[3]))
