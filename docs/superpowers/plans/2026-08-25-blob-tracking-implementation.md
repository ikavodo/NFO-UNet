# Blob Tracking Over Sliding Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a classical (non-learned) blob-tracking baseline — background subtraction + morphological filtering + Kalman/Hungarian (SORT-style) tracking — that estimates a person's position for one `seq_size`-length window of KTH frames, at the same granularity the U-Net operates at.

**Architecture:** A new `tracking/` package with three modules: `preprocess.py` (MOG2 bg-sub, morphological refine, shape filter — all plain numpy/`cv2`), `blob_tracker.py` (connected-component detection + vendored Kalman/Hungarian tracker + trajectory scoring), and `track_window.py` (wires the above into one `track_window(frames) -> dict | None` call). No persistence, no batch runner, no NFO wiring yet — those are explicit follow-ups per the spec.

**Tech Stack:** numpy, opencv-python (already in `requirements.txt`), `scipy.optimize.linear_sum_assignment` (not yet in `requirements.txt` — added in Task 1).

**Spec:** `docs/superpowers/specs/2026-08-25-blob-tracking-design.md`

## Global Constraints

- No torch dependency anywhere in `tracking/` — plain numpy/`cv2` end-to-end, matching the rest of this repo's dataset/generation code (spec: "Reuse strategy").
- Do not import from `master_thesis` — reimplement preprocessing, vendor the tracker math (spec: "Reuse strategy"). `master_thesis` installs as a package literally named `src`, which is the same class of naming collision that broke `utils.bb_utils` on the remote conda env earlier in this project.
- `track_window` operates on exactly one `seq_size`-length window (not a full sequence) and returns an estimate for the window's **center frame** (spec: "Why per-window, not per-sequence").
- This repo has no test framework (no pytest anywhere) and no CI — self-checks are plain `assert`-based scripts run via `python3 -m`, matching the codebase's existing convention (e.g. `gen_data/*/main.py`'s `if __name__ == '__main__':` pattern), not pytest.
- No `__init__.py` files — this repo's other packages (`dataset/`, `eval/`, `utils/`) are plain namespace packages; `tracking/` follows the same convention.

---

### Task 1: Preprocessing + blob detection + tracker core

**Files:**
- Create: `tracking/preprocess.py`
- Create: `tracking/blob_tracker.py`
- Create: `tracking/sanity_check_core.py`
- Modify: `requirements.txt` (add `scipy`)

**Interfaces:**
- Produces: `preprocess.foreground_mask(frames: np.ndarray[T,H,W] uint8, bg_frames: int = 5, var_threshold: float = 16.0) -> np.ndarray[T,H,W] uint8` (0/255 mask)
- Produces: `preprocess.refine_mask(masks: np.ndarray[T,H,W] uint8, close_kernel_size: int = 6, open_kernel_size: int = 4) -> np.ndarray[T,H,W] uint8`
- Produces: `preprocess.filter_by_shape(masks: np.ndarray[T,H,W] uint8, min_area: float = 50, min_solidity: float = 0.1) -> np.ndarray[T,H,W] uint8`
- Produces: `blob_tracker.detect_blobs(masks: np.ndarray[T,H,W] uint8, min_area: float = 80) -> list[list[dict]]` — each inner dict has keys `x`, `y`, `area`, `bbox` (a `(x1,y1,x2,y2)` tuple)
- Produces: `blob_tracker.track_blobs(detections: list[list[dict]], max_dist: float, max_age: int = 6) -> list[_Track]`
- Produces: `blob_tracker.score_and_fit(tracks: list[_Track], min_track_length: int = 3) -> dict | None` — keys `id`, `span`, `score`, `vx`, `resid_std`, `net_disp`, `frames`, `history` (`history` is `{frame_idx: (x, y)}`)

- [ ] **Step 1: Add scipy to requirements.txt**

Add this line to `requirements.txt` (it's already installed in this dev environment, so no install step needed here, but the remote/any fresh env needs it declared):

```
scipy==1.15.3
```

- [ ] **Step 2: Write `tracking/preprocess.py`**

```python
import numpy as np
import cv2


def foreground_mask(frames: np.ndarray, bg_frames: int = 5, var_threshold: float = 16.0) -> np.ndarray:
    """MOG2 background subtraction over a [T, H, W] uint8 grayscale stack.
    Returns a [T, H, W] uint8 binary mask (0/255)."""
    subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_frames, varThreshold=var_threshold,
                                                     detectShadows=False)
    masks = np.zeros_like(frames, dtype=np.uint8)
    for t in range(frames.shape[0]):
        masks[t] = subtractor.apply(frames[t])
    return masks


def refine_mask(masks: np.ndarray, close_kernel_size: int = 6, open_kernel_size: int = 4) -> np.ndarray:
    """Morphological close then open, per frame, on a [T, H, W] binary mask stack."""
    close_k = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    open_k = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    refined = np.zeros_like(masks, dtype=np.uint8)
    for t in range(masks.shape[0]):
        m = (masks[t] > 0).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        refined[t] = m
    return refined


def filter_by_shape(masks: np.ndarray, min_area: float = 50, min_solidity: float = 0.1) -> np.ndarray:
    """Keep only contours passing area/solidity thresholds, per frame, on a [T, H, W] mask stack."""
    clean = np.zeros_like(masks, dtype=np.uint8)
    for t in range(masks.shape[0]):
        frame = (masks[t] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(frame)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity >= min_solidity:
                cv2.drawContours(out, [cnt], -1, 1, -1)
        clean[t] = out * 255
    return clean
```

- [ ] **Step 3: Write `tracking/blob_tracker.py`**

```python
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
```

- [ ] **Step 4: Write the foundational sanity check (synthetic data, no KTH needed yet)**

`tracking/sanity_check_core.py`:

```python
import numpy as np

from tracking.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.blob_tracker import detect_blobs


def make_synthetic_window():
    """A dark background with a bright square moving right at 10px/frame."""
    frames = np.full((5, 60, 80), 30, dtype=np.uint8)
    for t in range(5):
        x = 10 + t * 10
        frames[t, 20:40, x:x + 15] = 220
    return frames


def main():
    frames = make_synthetic_window()
    masks = foreground_mask(frames)
    masks = refine_mask(masks)
    masks = filter_by_shape(masks)
    assert masks.shape == frames.shape, f"shape changed: {masks.shape} vs {frames.shape}"
    assert masks.dtype == np.uint8

    detections = detect_blobs(masks)
    assert len(detections) == 5
    found = sum(1 for d in detections if len(d) > 0)
    # MOG2's first frame typically has no learned background yet, so allow one miss
    assert found >= 4, f"expected blobs detected in most frames, got {found}/5"
    print(f"OK - detected blobs in {found}/5 synthetic frames")


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Run the sanity check**

Run: `cd /home/akovi/git_projects/NFO-UNet && python3 -m tracking.sanity_check_core`
Expected: prints `OK - detected blobs in N/5 synthetic frames` with N >= 4, exit code 0. If it fails with an `AssertionError` on `found >= 4`, check the mask output shapes with a quick `print(masks.sum())` inline — a common cause is `min_area`/`min_solidity` thresholds rejecting the synthetic square (it's a solid rectangle, solidity should be ~1.0, so this would indicate an `area` unit mismatch, not a real algorithm bug).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tracking/preprocess.py tracking/blob_tracker.py tracking/sanity_check_core.py
git commit -m "$(cat <<'EOF'
Add blob-tracking preprocessing + Kalman/Hungarian tracker core

Ports the validated tracker math from master_thesis (numpy/scipy, no torch)
and reimplements its preprocessing natively rather than importing across
repos. See docs/superpowers/specs/2026-08-25-blob-tracking-design.md.
EOF
)"
```

---

### Task 2: `track_window` + validation against real KTH ground truth

**Files:**
- Create: `tracking/track_window.py`
- Create: `tracking/sanity_check_kth.py`

**Interfaces:**
- Consumes: `preprocess.foreground_mask`, `preprocess.refine_mask`, `preprocess.filter_by_shape`, `blob_tracker.detect_blobs`, `blob_tracker.track_blobs`, `blob_tracker.score_and_fit` (all from Task 1, exact signatures above)
- Consumes (existing repo code): `gen_data.gen_kth_data.main.extract_bbs(file: utils.fs_utils.File) -> list[utils.bb_utils.BoundingBox]` (parses the *raw*, KTH-native `groundtruth.txt` format — one line per frame, `x,y,w,h` **normalized to `[0,1]`**, confirmed via `data/kth_staged/person01_walking_d1_uncomp/groundtruth.txt`); `utils.bb_utils.BoundingBox.scale((sx, sy))` and `.center()`; `utils.fs_utils.File` (a `namedtuple('File', 'name path')`)
- Produces: `track_window.track_window(frames: np.ndarray[seq_size,H,W] uint8, bg_frames=5, var_threshold=16.0, close_kernel_size=6, open_kernel_size=4, min_area=50, min_solidity=0.1, max_dist=12.5, max_age=6, min_track_length=3) -> dict | None` — keys `x`, `y`, `vx`, `score`, `resid_std` (all `float`), estimate is for the window's center frame (`frames.shape[0] // 2`), or `None` if no track of sufficient length was found

- [ ] **Step 1: Write `tracking/track_window.py`**

```python
import numpy as np

from tracking.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.blob_tracker import detect_blobs, track_blobs, score_and_fit


def track_window(frames: np.ndarray, bg_frames: int = 5, var_threshold: float = 16.0,
                 close_kernel_size: int = 6, open_kernel_size: int = 4,
                 min_area: float = 50, min_solidity: float = 0.1,
                 max_dist: float = 12.5, max_age: int = 6, min_track_length: int = 3):
    """Run bg-sub -> morph refine -> shape filter -> Kalman/Hungarian tracking over one
    [T, H, W] uint8 grayscale window, and return the winning track's estimate for the
    window's center frame.

    max_dist/max_dist default (12.5px) is rescaled from master_thesis's 1024x1024-tuned
    value (80px) to KTH's native 160x120 resolution by width ratio - see
    docs/superpowers/specs/2026-08-25-blob-tracking-design.md, "Hyperparameters". Retune
    empirically per dataset resolution, don't assume this transfers as-is.

    Returns a dict with keys 'x', 'y', 'vx', 'score', 'resid_std', or None if no track of
    at least min_track_length frames was found in the window.
    """
    masks = foreground_mask(frames, bg_frames=bg_frames, var_threshold=var_threshold)
    masks = refine_mask(masks, close_kernel_size=close_kernel_size, open_kernel_size=open_kernel_size)
    masks = filter_by_shape(masks, min_area=min_area, min_solidity=min_solidity)
    detections = detect_blobs(masks, min_area=min_area)
    tracks = track_blobs(detections, max_dist=max_dist, max_age=max_age)
    winner = score_and_fit(tracks, min_track_length=min_track_length)
    if winner is None:
        return None

    center_t = frames.shape[0] // 2
    if center_t in winner["history"]:
        cx, cy = winner["history"][center_t]
    else:
        # center frame had no detection in the winning track -> extrapolate x from the
        # fitted line, and use the mean y (this tracker's motion model is horizontal-only)
        frames_arr = np.array(winner["frames"])
        xs = np.array([winner["history"][f][0] for f in winner["frames"]])
        A = np.vstack([frames_arr, np.ones(len(frames_arr))]).T
        coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
        cx = coef[0] * center_t + coef[1]
        ys = np.array([winner["history"][f][1] for f in winner["frames"]])
        cy = ys.mean()

    return dict(x=float(cx), y=float(cy), vx=float(winner["vx"]),
               score=float(winner["score"]), resid_std=float(winner["resid_std"]))
```

- [ ] **Step 2: Write the KTH ground-truth validation self-check**

`tracking/sanity_check_kth.py`:

```python
import numpy as np
import cv2

from gen_data.gen_kth_data.main import extract_bbs
from tracking.track_window import track_window
from utils.fs_utils import File

SEQ_DIR = 'data/kth_staged/person01_walking_d1_uncomp'
SEQ_SIZE = 5
PIXEL_TOLERANCE = 20  # native KTH resolution is 160x120 - generous tolerance for a first pass


def find_valid_window(bbs, seq_size):
    """First window whose center frame has a real (non-sentinel) ground-truth detection."""
    margin = seq_size // 2
    for start in range(len(bbs) - seq_size + 1):
        center = start + margin
        if bbs[center].x >= 0:
            return start
    raise RuntimeError(f"no window with a valid center-frame detection found in {SEQ_DIR}")


def main():
    bbs = extract_bbs(File('groundtruth.txt', f'{SEQ_DIR}/groundtruth.txt'))
    start = find_valid_window(bbs, SEQ_SIZE)
    center_idx = start + SEQ_SIZE // 2

    frame_paths = [f'{SEQ_DIR}/{str(i).zfill(5)}.jpg' for i in range(start, start + SEQ_SIZE)]
    frames = np.stack([cv2.imread(p, 0) for p in frame_paths], axis=0)
    assert frames.shape[0] == SEQ_SIZE and frames.ndim == 3, f"unexpected frames shape: {frames.shape}"
    h, w = frames.shape[1], frames.shape[2]

    # raw KTH groundtruth.txt is normalized [0,1] - scale to this video's actual pixel size
    gt_bb = bbs[center_idx].scale((w, h))
    gt_cx, gt_cy = gt_bb.center()

    result = track_window(frames)
    assert result is not None, f"track_window found no track in window starting at frame {start}"

    dist = np.hypot(result['x'] - gt_cx, result['y'] - gt_cy)
    print(f"window start={start} center_frame={center_idx}")
    print(f"estimated=({result['x']:.1f}, {result['y']:.1f}) vx={result['vx']:.2f}px/frame "
          f"score={result['score']:.1f} resid_std={result['resid_std']:.2f}")
    print(f"ground truth=({gt_cx:.1f}, {gt_cy:.1f})")
    print(f"distance={dist:.1f}px (tolerance={PIXEL_TOLERANCE}px, frame size={w}x{h})")
    assert dist < PIXEL_TOLERANCE, f"estimated center {dist:.1f}px from ground truth, expected < {PIXEL_TOLERANCE}px"
    print("OK")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run the KTH validation check**

Run: `cd /home/akovi/git_projects/NFO-UNet && python3 -m tracking.sanity_check_kth`
Expected: prints the window/estimate/ground-truth/distance lines followed by `OK`, exit code 0.

If it fails on `result is not None` (no track found at all): check `max_dist=12.5` isn't gating out real matches — this is a rescaled *guess* per the spec, not verified against KTH yet. Try loosening it (e.g. `track_window(frames, max_dist=25)`) as a diagnostic, not a permanent fix — if that resolves it, the tolerance needs updating in `track_window`'s default and the spec's hyperparameter section, with a note on what value actually worked.

If it fails on the `dist < PIXEL_TOLERANCE` assertion but a track *was* found: print `result` and `gt_cx, gt_cy` to see whether the error looks like noise (a few px) or a systematic issue (e.g. tracking the wrong blob) — this determines whether to loosen `PIXEL_TOLERANCE` or investigate `min_area`/`min_solidity` rejecting the real person blob while accepting noise elsewhere.

- [ ] **Step 4: Commit**

```bash
git add tracking/track_window.py tracking/sanity_check_kth.py
git commit -m "$(cat <<'EOF'
Add track_window() and validate against real KTH ground truth

Wires preprocessing + detection + tracking into one per-window call
matching the U-Net's own per-window granularity, and validates the
estimated center against real KTH box ground truth rather than an
indirect visual/sharpness proxy.
EOF
)"
```
