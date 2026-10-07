"""A/4 composite insertion, ported to NFO and gated by SAM2's confidence
(plan: docs/superpowers/plans/2026-10-07-nfo-gated-a4-port.md).

Composite C_t = recency-weighted mean of the N = 7 native frames ending at t, each shifted onto
frame t by a GT-free constant velocity; no mask, no cut-out. Ported from master_thesis
experiments/prototypes/gpjatk_em/fusion_location.py (`recency_fusion`, :493; `interleave`, :393;
weights exp(-age/3.5)). C_t is inserted just before raw frame R_t into ONE SAM2 memory bank:
  - fixed A/4: every k = 4th t;
  - gated:     only if the baseline's predicted IoU at t-1 < tau (t0: its prompted-frame
               confidence), never denser than one composite per k frames (density cap).
The gate is open-loop: it reads the baseline run's logged confidence, so with no trigger the
frame sequence, and hence the output, equals the baseline's.
"""
import numpy as np
import cv2

N = 7
TAU_AGE = 3.5          # fusion_location.py: weights exp(-age / 3.5)
K = 4


def recency_weights(n=N):
    """Oldest first, newest last; sums to 1."""
    w = np.exp(-np.arange(n - 1, -1, -1) / TAU_AGE)
    return w / w.sum()


def recency_fusion(frames, shifts_x, shifts_y=None):
    """frames: oldest..newest (newest = t); shifts: px moving each frame onto frame t."""
    shifts_y = shifts_y if shifts_y is not None else [0.0] * len(frames)
    w = recency_weights(len(frames))
    acc = np.zeros(frames[0].shape, np.float64)
    for f, dx, dy, wi in zip(frames, shifts_x, shifts_y, w):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        acc += wi * cv2.warpAffine(f.astype(np.float32), M, (f.shape[1], f.shape[0]),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(np.round(acc), 0, 255).astype(np.uint8)


def schedule_fixed(T, k=K):
    return list(range(0, T, k))


def schedule_gated(pred_iou, tau, k=K):
    """Insertion times t: confidence at t-1 (t = 0: the prompted frame's own) below tau, and at
    least k frames since the previous insertion."""
    out, last = [], -k
    for t in range(len(pred_iou)):
        c = pred_iou[0] if t == 0 else pred_iou[t - 1]
        if c < tau and t - last >= k:
            out.append(t); last = t
    return out
