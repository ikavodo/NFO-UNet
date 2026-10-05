"""Per-frame and per-trial metrics for the NFO VOS pilot (spec §5).

J and F are the official DAVIS-2017 implementations (vendored, davis_metrics.py), including their
empty-GT/empty-prediction conventions. p and r make the decomposition 1/J = 1/p + 1/r - 1 visible:
drift collapses p with P non-empty, fragmentation-induced under-segmentation collapses r.
"""
import numpy as np

from benchmark.nfo_vos.davis_metrics import db_eval_boundary, db_eval_iou, db_statistics
from gen_data.gen_kth_data.kth_utils import scale_and_pad_img_to_square
from utils.bb_utils import BoundingBox

OUT_SIZE = 224
PNORM_THRESHOLDS = np.linspace(0, 0.5, 51)   # LaSOT normalised-precision AUC range


def frame_metrics(P, G):
    P, G = P.astype(bool), G.astype(bool)
    inter, ap, ag = int((P & G).sum()), int(P.sum()), int(G.sum())
    return dict(J=float(db_eval_iou(G, P)), F=float(db_eval_boundary(G, P)),
                p=inter / ap if ap else float('nan'), r=inter / ag if ag else float('nan'),
                area_p=ap, area_g=ag)


def dre_nre(P, G):
    """VOTS2023 drift-rate / not-reported errors over GT-present frames of a [T,H,W] stack."""
    present = G.reshape(len(G), -1).any(1)
    nonempty = P.reshape(len(P), -1).any(1)
    hit = (P & G).reshape(len(P), -1).any(1)
    n = present.sum()
    if n == 0:
        return float('nan'), float('nan')
    return float((present & nonempty & ~hit).sum() / n), float((present & ~nonempty).sum() / n)


def centre_error_norm(P, box):
    """Centroid of the predicted mask vs centre of an (x0,y0,x1,y1) box, normalised by box w,h
    (LaSOT P_norm). inf for an empty prediction."""
    ys, xs = np.nonzero(P)
    if len(xs) == 0:
        return float('inf')
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5          # pixel i covers [i, i+1) in box coords
    return float(np.hypot((cx - (x0 + x1) / 2) / w, (cy - (y0 + y1) / 2) / h))


def pnorm(P, boxes):
    d = np.array([centre_error_norm(p, b) for p, b in zip(P, boxes)])
    return float(np.mean([(d <= t).mean() for t in PNORM_THRESHOLDS]))


def decay(values):
    """DAVIS decay: mean of the first quartile minus mean of the last (db_statistics)."""
    return float(db_statistics(np.asarray(values, dtype=float))[2])


def native_to_224(mask):
    """Exactly the GT's own downscale (gen_nfo_pseudo_masks.py:346), then > 127."""
    m224, _ = scale_and_pad_img_to_square((mask.astype(np.uint8) * 255), BoundingBox(0, 0, 0, 0), OUT_SIZE)
    return m224 > 127
