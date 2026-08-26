# Integrated Image Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a window's winning track (position + fitted velocity), produce a single "integrated" image by aligning frames to the track's motion and fusing pixel intensities across them — for feeding to an off-the-shelf person detector (YOLO etc.) as a denoised, occlusion-robust reconstruction.

**Architecture:** One new function in `tracking/`, built on the alignment/anchor logic already validated in `tracking/show_aligned_masks.py`: given a window's frames + winning track, warp each frame to the center frame's reference point (via the fitted `vx`, matching the tracker's own horizontal-only motion model), then fuse the aligned stack with a configurable statistic. No new subsystem — reuses `track_window`/`track_windows_in_sequence`'s existing output.

**Tech Stack:** numpy, opencv (already in use throughout `tracking/`).

**Spec:** No separate spec doc — design was worked out in conversation (see the "aligned masks" and "statistics" discussion immediately preceding this plan); this plan documents the agreed decisions directly.

## Global Constraints

- Default fusion statistic is **median**, not mean — robustness to a minority-of-frames occlusion at a given aligned pixel is the actual point (mean blends the occluder's intensity in; median picks the majority/true value outright).
- Default background mode is **full-frame** (not masked/blank) — an off-the-shelf detector's backbone expects natural scene statistics; a cutout-on-blank-background is a severe, likely confidence-degrading distribution shift, whereas full-frame motion-compensated integration naturally blurs the (now-non-stationary-in-the-warped-frame) background while keeping the aligned subject sharp - a composition much closer to what such detectors are actually trained on.
- Both dimensions (fusion statistic: mean/median/gaussian-weighted; background: full-frame/masked) must be independently selectable, not hardcoded to the defaults — this is explicitly a comparison/exploration tool, not a single fixed pipeline.
- Follows this repo's established `tracking/` conventions: plain functions (not classes) taking/returning numpy arrays, no framework, no `__init__.py` (namespace packages throughout this repo).
- Self-check: one assert-based script (no pytest, matching every other `tracking/*sanity_check*` script in this repo), not a persistent test suite.

---

### Task 1: Core alignment + fusion function

**Files:**
- Create: `tracking/integrate_image.py`
- Test: `tracking/sanity_check_integrate.py`

**Interfaces:**
- Consumes: a window's `frames: np.ndarray[T, H, W]` (uint8 grayscale) and the `winner` dict `track_window`/`track_windows_in_sequence` already produce internally (specifically needs `winner['vx']` and `winner['history']` - NOTE: `track_window`'s *public* return dict does NOT currently expose `history` or the per-frame anchor positions, only the final `x`/`y`/`vx`/`score`/`resid_std` for the center frame. This task's function takes the raw `winner` dict from `score_and_fit()` directly (as `show_aligned_masks.py` already does), not `track_window()`'s return value - it is called at the same point in the pipeline `show_aligned_masks.py` already demonstrates, not downstream of `track_window`.
- Consumes: optionally, per-frame detections (`list[list[dict]]` from `blob_tracker.detect_blobs`) and `merge_radius: float`, for masked-background mode's mask restriction (same `restrict_to_nearby` logic already written in `tracking/show_aligned_masks.py` - this task moves/reuses it, not reimplements it).
- Produces: `integrate_image.align_frames(frames: np.ndarray, winner: dict, crop_size: int) -> np.ndarray[T, crop_size, crop_size]` - per-frame crops centered on the track's anchor at each frame, shifted by `-vx*dt` to compensate motion (identical alignment convention to `show_aligned_masks.py`'s `aligned` list).
- Produces: `integrate_image.fuse(aligned: np.ndarray, method: str, gaussian_sigma: float = None) -> np.ndarray[crop_size, crop_size]` - `method` one of `'mean'`, `'median'`, `'gaussian'` (temporal-distance-weighted sum; `gaussian_sigma` required when `method='gaussian'`, in units of frame-index distance from center).
- Produces: `integrate_image.integrate(frames, winner, detections=None, merge_radius=None, crop_size=220, method='median', gaussian_sigma=None, mask_background=False) -> np.ndarray[crop_size, crop_size]` - the end-to-end entry point combining the above; when `mask_background=True`, `detections`/`merge_radius` are required (raises `ValueError` if missing) and each frame is restricted to nearby-detection pixels (via `restrict_to_nearby`, moved into this module) before alignment/fusion; when `False`, the full crop (not just mask pixels) is used.

- [ ] **Step 1: Move `crop_at`/`anchor_for_frame`/`restrict_to_nearby` out of `show_aligned_masks.py` into `tracking/integrate_image.py`**

These three helpers are already written and validated in `tracking/show_aligned_masks.py`. Cut them from that file and paste into the new `tracking/integrate_image.py`, unchanged:

```python
import numpy as np
import cv2


def crop_at(img, cx, cy, size):
    h, w = img.shape
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    x1, y1 = x0 + size, y0 + size
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = img[y0:y1, x0:x1]
    return cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)


def anchor_for_frame(winner, t):
    if t in winner['history']:
        return winner['history'][t][0], winner['history'][t][1]
    frames_arr = np.array(winner['frames'])
    xs = np.array([winner['history'][f][0] for f in winner['frames']])
    A = np.vstack([frames_arr, np.ones(len(frames_arr))]).T
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    ys = np.array([winner['history'][f][1] for f in winner['frames']])
    return coef[0] * t + coef[1], ys.mean()


def restrict_to_nearby(mask, detections_at_frame, ax, ay, merge_radius):
    nearby = [d for d in detections_at_frame if np.hypot(d['x'] - ax, d['y'] - ay) <= merge_radius]
    keep = np.zeros_like(mask)
    for d in nearby:
        x1, y1, x2, y2 = d['bbox']
        keep[y1:y2, x1:x2] = 1
    return mask * keep
```

Update `tracking/show_aligned_masks.py` to `from tracking.integrate_image import crop_at, anchor_for_frame, restrict_to_nearby` instead of defining them locally, and delete its now-duplicate local definitions. Note the signature drops the unused `T` parameter from `anchor_for_frame` (it was never used in the original) - update `show_aligned_masks.py`'s one call site (`anchor_for_frame(winner, t, T)` -> `anchor_for_frame(winner, t)`) accordingly.

- [ ] **Step 2: Run the existing visualization to confirm the move didn't break it**

Run: `cd /home/akovi/git_projects/NFO-UNet && python3 -m tracking.show_aligned_masks`
Expected: identical output to before (same `vx=7.91px/frame, saved to tracking/aligned_masks.png` line, same image content) - this step only moved code, changed nothing behaviorally.

- [ ] **Step 3: Write `align_frames`**

Append to `tracking/integrate_image.py`:

```python
def align_frames(frames: np.ndarray, winner: dict, crop_size: int = 220) -> np.ndarray:
    """Crop every frame to crop_size x crop_size, centered on the winning track's anchor
    point at that frame and shifted by -vx*dt so every frame samples the same real-world
    point as the center frame (this tracker's motion model is horizontal-only, matching
    its constant-velocity Kalman assumption)."""
    T = frames.shape[0]
    center_t = T // 2
    aligned = np.zeros((T, crop_size, crop_size), dtype=frames.dtype)
    for t in range(T):
        ax, ay = anchor_for_frame(winner, t)
        dt = t - center_t
        aligned[t] = crop_at(frames[t], ax - winner['vx'] * dt, ay, crop_size)
    return aligned
```

- [ ] **Step 4: Write `fuse`**

```python
def fuse(aligned: np.ndarray, method: str = 'median', gaussian_sigma: float = None) -> np.ndarray:
    """Fuse an aligned [T, H, W] stack into one [H, W] image.

    'median': robust to a minority of occluded frames at a given pixel (the occluder's
    intensity gets outvoted by the majority true value) - default, since occlusion
    robustness is the actual point, not just noise averaging.
    'mean': simple baseline: blends occluder and true value together, ghosting rather
    than reconstructing - kept for comparison, not the default.
    'gaussian': weights frames by a Gaussian in temporal distance from the center frame
    (gaussian_sigma, in frame-index units) - trades occlusion-robustness (which benefits
    from using all frames equally) for pose fidelity (limb articulation across a gait
    cycle changes shape frame to frame; weighting toward center reduces smearing a moving
    limb across positions). Requires gaussian_sigma.
    """
    if method == 'median':
        return np.median(aligned, axis=0).astype(aligned.dtype)
    if method == 'mean':
        return aligned.mean(axis=0).astype(aligned.dtype)
    if method == 'gaussian':
        if gaussian_sigma is None:
            raise ValueError("gaussian_sigma is required when method='gaussian'")
        T = aligned.shape[0]
        center_t = T // 2
        dt = np.arange(T) - center_t
        weights = np.exp(-(dt ** 2) / (2 * gaussian_sigma ** 2))
        weights /= weights.sum()
        return np.tensordot(weights, aligned.astype(np.float64), axes=(0, 0)).astype(aligned.dtype)
    raise ValueError(f"method must be 'median', 'mean', or 'gaussian', got {method!r}")
```

- [ ] **Step 5: Write the end-to-end `integrate` entry point**

```python
def integrate(frames: np.ndarray, winner: dict, detections=None, merge_radius: float = None,
             crop_size: int = 220, method: str = 'median', gaussian_sigma: float = None,
             mask_background: bool = False) -> np.ndarray:
    """End-to-end: align frames to the winning track's motion, optionally restrict each
    frame to only the person's own nearby detection(s) (mask_background=True - blanks
    everything else, e.g. background clutter, before fusion), then fuse into one image.

    mask_background=False (default): fuses full-frame crops. Recommended for feeding an
    off-the-shelf detector (YOLO etc.) - full-frame integration naturally blurs the
    (non-stationary-in-the-aligned-frame) background while keeping the aligned subject
    sharp, a composition much closer to what such detectors are trained on than a
    cutout-on-blank-background, which is a severe distribution shift from natural photos.
    mask_background=True requires detections and merge_radius.
    """
    if mask_background:
        if detections is None or merge_radius is None:
            raise ValueError("mask_background=True requires detections and merge_radius")
        frames = frames.copy()
        for t in range(frames.shape[0]):
            ax, ay = anchor_for_frame(winner, t)
            frames[t] = restrict_to_nearby(frames[t], detections[t], ax, ay, merge_radius)

    aligned = align_frames(frames, winner, crop_size)
    return fuse(aligned, method=method, gaussian_sigma=gaussian_sigma)
```

- [ ] **Step 6: Write the self-check**

`tracking/sanity_check_integrate.py`:

```python
import os

import cv2
import numpy as np

from tracking.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval_nfo import BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN
from tracking.integrate_image import integrate
from tracking.preprocess import filter_by_shape, foreground_mask, refine_mask


def load_sequence_prefix(seq, up_to):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    return np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(up_to)], axis=0)


def main():
    seq, center = 'seq1', 17
    frames_all = load_sequence_prefix(seq, center + SPAN + 1)
    masks_all = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))
    window_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = frames_all[window_indices]
    masks = masks_all[window_indices]
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT)
    assert winner is not None, "expected a track on this known-good window"

    for method in ['median', 'mean']:
        for mask_bg in [False, True]:
            img = integrate(frames, winner, detections=detections, merge_radius=MERGE_RADIUS,
                            method=method, mask_background=mask_bg)
            assert img.shape == (220, 220)
            assert img.dtype == frames.dtype
            print(f"method={method} mask_background={mask_bg}: "
                  f"min={img.min()} max={img.max()} mean={img.mean():.1f}")

    img_gauss = integrate(frames, winner, method='gaussian', gaussian_sigma=1.5)
    assert img_gauss.shape == (220, 220)
    print(f"method=gaussian sigma=1.5: min={img_gauss.min()} max={img_gauss.max()} "
          f"mean={img_gauss.mean():.1f}")
    print("OK")


if __name__ == '__main__':
    main()
```

- [ ] **Step 7: Run the self-check**

Run: `cd /home/akovi/git_projects/NFO-UNet && python3 -m tracking.sanity_check_integrate`
Expected: prints one stats line per `(method, mask_background)` combination plus the gaussian line, then `OK`, exit code 0. `mask_background=True` runs should show a lower `mean` than their `False` counterparts (blanked background pulls the average down) - if not, that's a sign `restrict_to_nearby` isn't actually zeroing anything for this window, worth checking before trusting the run.

- [ ] **Step 8: Commit**

```bash
git add tracking/integrate_image.py tracking/sanity_check_integrate.py tracking/show_aligned_masks.py
git commit -m "$(cat <<'EOF'
Add integrated-image generation (align + fuse) for feeding a detector

align_frames/fuse/integrate in tracking/integrate_image.py: warp a
window's frames to the winning track's motion (reusing the alignment
logic already validated in show_aligned_masks.py), then fuse with a
configurable statistic (median default - robust to a minority of
occluded frames at a given pixel, unlike mean which blends the
occluder in; mean and gait-aware gaussian-temporal-weighting also
available). mask_background option supports both full-frame (default -
natural scene statistics, closer to what an off-the-shelf detector like
YOLO was trained on) and blank-background (person cutout only) modes
for comparison.
EOF
)"
```

---

### Task 2: Visual comparison across fusion/background combinations

**Files:**
- Create: `tracking/show_integrated_image.py`

**Interfaces:**
- Consumes: `integrate_image.integrate` (Task 1).
- Produces: a saved PNG (`tracking/integrated_image_comparison.png`) grid comparing all `(method, mask_background)` combinations side by side on the same known-good window (`seq1`, `center=17`), so the full-frame-vs-masked and median-vs-mean-vs-gaussian questions can be inspected visually, not just numerically.

- [ ] **Step 1: Write the comparison script**

```python
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracking.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval_nfo import BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN
from tracking.integrate_image import integrate
from tracking.preprocess import filter_by_shape, foreground_mask, refine_mask


def load_sequence_prefix(seq, up_to):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    return np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(up_to)], axis=0)


def main(seq='seq1', center=17):
    frames_all = load_sequence_prefix(seq, center + SPAN + 1)
    masks_all = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))
    window_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = frames_all[window_indices]
    masks = masks_all[window_indices]
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT)

    configs = [
        ('median', False), ('mean', False), ('gaussian', False),
        ('median', True), ('mean', True), ('gaussian', True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for (method, mask_bg), ax in zip(configs, axes.flat):
        img = integrate(frames, winner, detections=detections, merge_radius=MERGE_RADIUS,
                        method=method, gaussian_sigma=1.5, mask_background=mask_bg)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"{method}, {'masked' if mask_bg else 'full-frame'}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    out_path = 'tracking/integrated_image_comparison.png'
    plt.savefig(out_path, dpi=110)
    print(f"saved to {out_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it and visually inspect**

Run: `cd /home/akovi/git_projects/NFO-UNet && python3 -m tracking.show_integrated_image`
Expected: `tracking/integrated_image_comparison.png` shows a 2x3 grid. Visually confirm: the two `masked` columns show a blank/black background outside the person silhouette; the `full-frame` columns show real (if slightly blurred) scene content; `mean` should look slightly softer/more ghosted than `median` wherever any frame had partial occlusion near the person.

- [ ] **Step 3: Commit**

```bash
git add tracking/show_integrated_image.py tracking/integrated_image_comparison.png
git commit -m "Add visual comparison of integrated-image fusion/background options"
```
