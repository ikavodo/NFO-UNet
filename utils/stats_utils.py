"""Small robust-statistics helpers shared by the NFO benchmark (composite gate, motion readout)."""
import numpy as np
from scipy.stats import theilslopes


def theil_at(ts, xs, t_eval):
    """Theil–Sen line through (ts, xs), evaluated at each t in t_eval: {t: x(t)}; last value (or NaN) if < 2 points."""
    if len(ts) < 2:
        return {t: (xs[-1] if xs else np.nan) for t in t_eval}
    s, b, _, _ = theilslopes(xs, ts)
    return {t: s * t + b for t in t_eval}


def robust_z(img, bg):
    """Robust z-score of |img − bg|: (d − median d) / (1.4826 · MAD d), per image."""
    d = np.abs(img - bg); med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med))
    return (d - med) / max(mad, 1e-6)
