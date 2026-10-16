"""Feature-space alignment of buffered frames onto frame t (2026-10-16): the swap-in alignment source for every composite
in the pipeline (V̂ on NFO via vhat_vs_silhouette NFO_TRACK=register, the adapter's training data, track_check).

Buffer: the last N frames' SAM2 backbone features at stride 16 (image_embed, 256 ch, ≈ 2 MB each at fp16), our own
cache, independent of SAM2's memory bank. Memory-encoder output is not used: it is conditioned on SAM2's predicted
mask (sam2_base.py:850), so registration on it would lock onto what SAM2 already segments, the opposite of the goal.

For each buffered frame i:  Δ_i = F_i − F_B, Δ_t = F_t − F_B (the empty scene cancels the static foliage).
Correlation volume over the person box around c_t (cells):
  V_i(dy, dx) = NCC( Δ_t[box],  Δ_i[box − (dy, dx)] ),  dx ∈ prior_i ± R,  dy ∈ ±R_y
(the local correlation layer of FlowNet, arXiv 1504.06852, normalised per window as in phase / cross-correlation
registration). Peak → shift; parabolic sub-cell refinement in x; confidence = peak NCC. Below `min_conf` the prior
shift is kept (prior = kernel tracker's x-shift, layers/track_check.kernel16_chain).
Cells are anisotropic: SAM2 resizes to 1024² without keeping the aspect ratio (native px per cell = W/64 × H/64).
"""
import numpy as np
import torch


def shifts(D_hist, D_t, c, wh, prior_px, cell, R=4, Ry=1, min_conf=0.2):
    """D_hist [N, C, H, W], D_t [C, H, W] (torch or numpy, any float dtype); c = (x, y) native centre at t; wh = (w, h)
    native person box; prior_px [N] x-shifts i → t (native px); cell = (px per cell in x, in y).
    Returns (dx_px [N], conf [N])."""
    Dh = torch.as_tensor(D_hist).float()
    Dt = torch.as_tensor(D_t).float()
    N, C, H, W = Dh.shape
    cx, cy = cell
    x0, x1 = int(round((c[0] - wh[0] / 2) / cx)), int(round((c[0] + wh[0] / 2) / cx)) + 1
    y0, y1 = int(round((c[1] - wh[1] / 2) / cy)), int(round((c[1] + wh[1] / 2) / cy)) + 1
    x0, x1, y0, y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
    prior_px = np.asarray(prior_px, float)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return prior_px.copy(), np.zeros(N)
    a = Dt[:, y0:y1, x0:x1].reshape(-1)
    a = (a - a.mean()) / (a.std() + 1e-6)
    hh, ww = y1 - y0, x1 - x0
    out_dx, out_c = prior_px.copy(), np.zeros(N)
    for i in range(N):
        p = int(round(prior_px[i] / cx))
        vol = np.full((2 * Ry + 1, 2 * R + 1), -np.inf)
        for iy, dy in enumerate(range(-Ry, Ry + 1)):
            for ix, dx in enumerate(range(p - R, p + R + 1)):
                ys, xs = y0 - dy, x0 - dx
                if ys < 0 or xs < 0 or ys + hh > H or xs + ww > W:
                    continue
                b = Dh[i, :, ys:ys + hh, xs:xs + ww].reshape(-1)
                vol[iy, ix] = float((a * (b - b.mean()) / (b.std() + 1e-6)).mean())
        if not np.isfinite(vol).any():
            continue
        iy, ix = np.unravel_index(np.argmax(vol), vol.shape)
        conf = vol[iy, ix]
        if conf < min_conf:
            out_c[i] = conf
            continue
        l = vol[iy, ix - 1] if ix > 0 else -np.inf
        r = vol[iy, ix + 1] if ix < vol.shape[1] - 1 else -np.inf
        den = l - 2 * conf + r
        off = 0.5 * (l - r) / den if np.isfinite(l) and np.isfinite(r) and den < 0 else 0.0
        out_dx[i], out_c[i] = (p - R + ix + off) * cx, conf
    return out_dx, out_c
