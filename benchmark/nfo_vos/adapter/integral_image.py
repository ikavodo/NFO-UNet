"""Integral image of the past frames (aligned, change-gated, recency-weighted; not frame t itself): the input C and evidence mass a of the adapter.
Ancestor: Airborne Optical Sectioning (integrate frames before detection; arXiv 2111.06959, 2207.13344)."""
import cv2
import numpy as np

from utils.stats_utils import robust_z

TAU_AGE = 3.5          # recency weight scale: w ∝ exp(−age / 3.5), ported from master_thesis fusion_location.py


def integral_image_and_mass(imgs, Ms, bg, tau=None, z0=3.0):
    """Integral image C and evidence mass a of past frames imgs (each warped onto the target frame by its 2x3 affine Ms)
    against the empty scene bg. Recency weights w_i ∝ exp(−age_i / tau); change gate G_i = σ(z_i − z0), z = robust
    z-score of |img − bg| (z0 = 'adaptive': max(3, q90(z))); C = Σ w G A / Σ w G (plain weighted mean where the mass is
    ≈ 0); a = clip(Σ w G, 0, 1). tau and z0 are randomised in training (fusion_data); NFO uses tau = TAU_AGE."""
    tau = TAU_AGE if tau is None else tau
    w = np.exp(-np.arange(len(imgs) - 1, -1, -1) / tau); w = (w / w.sum())[:, None, None]
    wa = lambda x, M: cv2.warpAffine(x.astype(np.float32), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    A = np.stack([wa(img, M) for img, M in zip(imgs, Ms)])
    def gate(img):                        # z0 = 'adaptive': per-frame threshold max(3, q90(z)) → ≤ ~10% of pixels gated,
        z = robust_z(img, bg)                 # invariant to the scene's nuisance level (train / test parity, 2026-10-16)
        th = max(3.0, float(np.percentile(z, 90))) if z0 == 'adaptive' else z0
        return 1 / (1 + np.exp(-(z - th)))
    G = np.stack([wa(gate(img), M) for img, M in zip(imgs, Ms)])
    raw, mass = (w * A).sum(0), (w * G).sum(0)
    C = np.where(mass > 1e-6, (w * G * A).sum(0) / np.maximum(mass, 1e-6), raw)
    return C, np.clip(mass, 0, 1)
