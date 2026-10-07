"""GT-free person velocity from the causal blob tracker, used to align the frames of A/4 composites
(a4_build.py). Was composite.py (composite-prepend, removed 2026-10-07: did not replicate on the
round-robin pilot; see docs/nfo_failure_log.md and git history).

Estimator 'hybrid_x' (chosen on the old pilot: median 7.7 native px vs GT shifts, against 9.4 for
the chain alone): x-only constant velocity; the selected track's own OLS velocity when it has
>= 3 points in the horizon, else a track-ID-free chain (walk back from t0 along the predicted line,
merge the detections inside the gate, Theil-Sen refit), which crosses track-ID breaks.
"""
import cv2
import numpy as np
from scipy.stats import theilslopes

from benchmark.nfo_vos import run_t4 as R, trials as TR
from tracking.core.blob_tracker import merged_center

N, STRIDE = 7, 2                 # history horizon: 7 frames at stride 2, ending at t0
NATIVE = 'data/nfo_final/nfo_final'


def native(idx, seq=TR.SEQ):
    return cv2.imread(f'{NATIVE}/{seq}/{idx:05d}.jpg', 0)


def horizon_frames(t0, n, stride, keep):
    return [k for k in (t0 - stride * j for j in range(n - 1, -1, -1)) if k >= 0 and keep(k)]


def fit_velocity(pts, fit):
    """Constant velocity (vx, vy) per frame from {k: (x, y)}: 'ols' or 'theilsen'."""
    ks = np.array(sorted(pts), float)
    if len(ks) < 2:
        return np.zeros(2)
    P = np.array([pts[k] for k in sorted(pts)], float)
    if fit == 'theilsen':
        return np.array([theilslopes(P[:, d], ks)[0] for d in range(2)])
    return np.array([np.polyfit(ks, P[:, d], 1)[0] for d in range(2)])


def chain_positions(dets, t0, p0, v0, radius, fit='theilsen', iters=2):
    """Track-ID-free history: walk back from t0 along the constant-velocity line through p0; in
    each past frame merge the detections within `radius` of the prediction (merged_center); refit
    the velocity; repeat."""
    p0, v = np.asarray(p0, float), np.asarray(v0, float)
    for _ in range(iters):
        pts = {t0: tuple(p0)}
        for k in sorted((k for k in dets if k < t0), reverse=True):
            pred = p0 - v * (t0 - k)
            near = [d for d in dets[k] if np.hypot(d['x'] - pred[0], d['y'] - pred[1]) <= radius]
            if near:
                pts[k] = merged_center(near, pred[0], pred[1], radius)
        v = fit_velocity(pts, fit)
    return pts, v


def _history(trial, n, stride):
    """Tracker run on the contiguous frames t0-(n-1)*stride..t0; the track whose detection at t0
    lies in the prompt box (largest if several, else nearest to the box centre)."""
    t0 = trial['t0']
    c0 = max(0, t0 - stride * (n - 1))
    D = R.detect_and_track(list(range(c0, t0 + 1)), trial['warmup'], trial['box_224'][3] - trial['box_224'][1], trial['seq'])
    L = t0 - c0
    dets = {c0 + i: d for i, d in enumerate(D['dets'])}
    cands = [tr for tr in D['tracks'] if L in tr.history]
    x0, y0, x1, y1 = trial['box_224']
    bx, by = (x0 + x1) / 2, (y0 + y1) / 2
    if not cands:
        return dict(dets=dets, hist={}, kw=D['kw'], seed=(bx, by))
    inside = [tr for tr in cands if x0 <= tr.history[L][0] <= x1 and y0 <= tr.history[L][1] <= y1]
    tr = (max(inside, key=lambda r: r.history[L][2] * r.history[L][3]) if inside else
          min(cands, key=lambda r: np.hypot(r.history[L][0] - bx, r.history[L][1] - by)))
    return dict(dets=dets, hist={c0 + k: tr.history[k][:2] for k in tr.history if k <= L},
                kw=D['kw'], seed=tr.history[L][:2])


def estimate_shifts(trial, method='hybrid_x', n=N, stride=STRIDE):
    """GT-free shifts {k: (dx, 0)} (224 px) moving frame k's person onto t0's, for the horizon
    frames with person evidence (hybrid_x, see module docstring)."""
    assert method == 'hybrid_x', 'only the chosen estimator is kept (others: git history)'
    t0 = trial['t0']
    H = _history(trial, n, stride)
    v0 = fit_velocity(H['hist'], 'ols') if len(H['hist']) >= 2 else np.zeros(2)
    p0 = merged_center(H['dets'][t0], H['seed'][0], H['seed'][1], H['kw']['merge_radius'])
    pts, v = chain_positions(H['dets'], t0, p0, v0, H['kw']['merge_radius'], fit='theilsen')
    if len(H['hist']) >= 3:                                  # the track's own velocity when it has one
        v = fit_velocity(H['hist'], 'ols')
    ks = horizon_frames(t0, n, stride, lambda k: k in pts)
    return {k: (v[0] * (t0 - k), 0.0) for k in ks}
