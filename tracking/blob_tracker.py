import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def detect_blobs(masks: np.ndarray, min_area: float = 80):
    """[T, H, W] binary (0/255) mask stack -> list of per-frame detection lists.
    Each detection: {'x': cx, 'y': cy, 'area': area, 'bbox': (x1, y1, x2, y2)}."""
    detections = []
    for t in range(masks.shape[0]):
        frame = (masks[t] > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame, connectivity=8)
        dets = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            cx, cy = centroids[i]
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            dets.append({"x": cx, "y": cy, "area": area, "bbox": (x, y, x + w, y + h)})
        detections.append(dets)
    return detections


class _Track:
    """Constant-velocity Kalman filter, position/motion state only. Ported from
    master_thesis/experiments/prototypes/motion_via_blob_tracking.py (numpy/scipy only,
    no torch dependency in the original either)."""
    _next_id = 0

    def __init__(self, x, y, t0):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.state = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 50.0
        self.history = {t0: (x, y)}
        self.first_frame = t0
        self.last_frame = t0
        self.misses = 0

    def predict(self):
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        Q = np.eye(4) * 2.0
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return self.state[:2]

    def update(self, x, y, t):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * 9.0
        z = np.array([x, y])
        y_res = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.P = (np.eye(4) - K @ H) @ self.P
        self.history[t] = (x, y)
        self.last_frame = t
        self.misses = 0


def track_blobs(detections, max_dist: float, max_age: int = 6):
    """Kalman + Hungarian association across frames. Returns completed tracks."""
    T = len(detections)
    active, dead = [], []
    for t in range(T):
        preds = {tr.id: tr.predict() for tr in active}
        dets = detections[t]
        matched_tracks, matched_dets = set(), set()
        if active and dets:
            cost = np.zeros((len(active), len(dets)))
            for i, tr in enumerate(active):
                px, py = preds[tr.id]
                for j, d in enumerate(dets):
                    cost[i, j] = np.hypot(px - d["x"], py - d["y"])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= max_dist:
                    active[r].update(dets[c]["x"], dets[c]["y"], t)
                    matched_tracks.add(r)
                    matched_dets.add(c)
        for i, tr in enumerate(active):
            if i not in matched_tracks:
                tr.misses += 1
        for j, d in enumerate(dets):
            if j not in matched_dets:
                active.append(_Track(d["x"], d["y"], t))
        still_active = []
        for tr in active:
            (dead if tr.misses > max_age else still_active).append(tr)
        active = still_active
    dead.extend(active)
    return dead


def merged_center(detections_at_frame, anchor_x: float, anchor_y: float, merge_radius: float):
    """Merge all detections within merge_radius of (anchor_x, anchor_y) into one combined
    bbox, and return its center. A single frame's foreground mask commonly fragments a
    person into disconnected blobs (head/torso/legs) - track_blobs/score_and_fit track
    whichever individual fragment scores best, not the whole-person centroid a
    ground-truth box represents, so this merge step is needed to read off a position that
    corresponds to the whole person. Adapted from master_thesis's compute_merged_box,
    simplified: that function also picks the best-scoring frame across a whole track;
    here the target frame is already fixed (the window's center), so only the merge
    itself is needed.

    Falls back to (anchor_x, anchor_y) if no detections are within range.
    """
    nearby = [d for d in detections_at_frame if np.hypot(d['x'] - anchor_x, d['y'] - anchor_y) <= merge_radius]
    if not nearby:
        return anchor_x, anchor_y
    x1 = min(d['bbox'][0] for d in nearby)
    y1 = min(d['bbox'][1] for d in nearby)
    x2 = max(d['bbox'][2] for d in nearby)
    y2 = max(d['bbox'][3] for d in nearby)
    return (x1 + x2) / 2, (y1 + y2) / 2


def score_and_fit(tracks, min_track_length: int = 3):
    """Score completed tracks by persistence x drift-consistency (span * net_displacement
    / (1 + residual_std) of a linear fit to the x-centroid trajectory). Returns the winning
    track's info dict, or None if no track has at least min_track_length frames."""
    results = []
    for tr in tracks:
        frames = sorted(tr.history.keys())
        if len(frames) < min_track_length:
            continue
        span = tr.last_frame - tr.first_frame + 1
        xs = np.array([tr.history[f][0] for f in frames])
        A = np.vstack([frames, np.ones(len(frames))]).T
        coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
        resid_std = (xs - A @ coef).std()
        net_disp = np.hypot(xs[-1] - xs[0], tr.history[frames[-1]][1] - tr.history[frames[0]][1])
        score = span * net_disp / (1.0 + resid_std)
        results.append(dict(id=tr.id, span=span, score=score, vx=coef[0],
                             resid_std=resid_std, net_disp=net_disp, frames=frames, history=tr.history))
    if not results:
        return None
    results.sort(key=lambda r: -r["score"])
    return results[0]
