# Blob tracking over sliding windows — design

## Context

NFO-UNet trains a spatio-temporal U-Net to localize a person from a window of
`seq_size` stacked frames. We want a second, classical (non-learned) baseline
for the same task: background-subtraction + morphological blob filtering,
followed by a constant-velocity Kalman filter + Hungarian assignment (SORT-style
tracking), scoring completed tracks by persistence × drift-consistency, and
reading off the winning track's position/velocity.

An implementation of exactly this already exists and is validated, in the
sibling repo `master_thesis` (`experiments/prototypes/motion_via_blob_tracking.py`,
`src/preprocess.py`). This spec covers porting it into NFO-UNet at the right
granularity to be comparable to the U-Net baseline, not a from-scratch design.

## Scope

In scope:
- Preprocessing: MOG2 background subtraction → morphological close/open →
  contour area/solidity filtering → connected-component blob detection.
- Tracking: constant-velocity Kalman filter + Hungarian assignment across a
  window's frames; score completed tracks; return the winning track's info.
- One callable entry point, `track_window(frames) -> dict | None`, operating on
  a single `seq_size`-length window of grayscale frames (native KTH resolution,
  read from `data/kth_staged`), not a full sequence.
- One assert-based self-check against real KTH ground truth.

Out of scope (deferred, per earlier discussion):
- The blob→heatmap localization NN.
- NFO integration.
- Any persisted/batch-runner output — this is a callable module for now.

## Why per-window, not per-sequence

Two independent reasons converge on the same answer:
1. A KTH sequence contains multiple back-and-forth passes at different
   velocities and directions. A single linear `vx` fit across a whole sequence
   is invalid — `score_and_fit`'s residual-vs-line assumption only holds within
   one directional run.
2. The eval pipeline (and the U-Net itself) operates per-window: one estimate
   per `seq_size`-frame window, centered on a target frame, matching
   `AbstractDataSet.__getitem__`'s `margin = seq_size // 2` convention. For an
   apples-to-apples comparison, the tracker must produce estimates at the same
   granularity, not one trajectory per video.

## Reuse strategy

Split by component:

- **Vendor near-verbatim**: `_Track`, `track_blobs`, `score_and_fit` from
  `motion_via_blob_tracking.py`. Pure numpy + `scipy.optimize.linear_sum_assignment`,
  no torch dependency, already validated — re-deriving this math from scratch
  would only risk reintroducing bugs that repo already worked through.
- **Reimplement, don't import**: `fgs` (mog2 branch), `kernel_filter`,
  `filter_by_shape` from `master_thesis/src/preprocess.py`. These are thin
  torch-tensor-in/out wrappers around a handful of `cv2` calls — low complexity,
  low duplication risk. Reimplementing as plain numpy/`cv2` (a) matches how the
  rest of NFO-UNet's pipeline already works (numpy end-to-end, torch only at
  the final batch step) and (b) avoids adding `master_thesis` as an installable
  dependency, whose package is named `src` — a dangerously generic top-level
  name, the same class of collision that broke `utils.bb_utils` on the remote
  conda `base` env earlier in this project.

## Module layout

New top-level package `tracking/`, mirroring the existing `eval/`/`dataset/`/`network/` convention:

- `tracking/preprocess.py` — `foreground_mask()`, `refine_mask()`, `filter_by_shape()`.
  All numpy arrays in/out.
- `tracking/blob_tracker.py` — `detect_blobs()` (connected components per frame),
  the vendored `_Track` / `track_blobs` / `score_and_fit`, and `merged_center()`
  (merges detections near a tracked point into one combined bbox before reading
  off a position — added during implementation once validation showed the
  tracker locking onto individual body-part fragments rather than a
  whole-person centroid; see "Hyperparameters" and "Validation" below).
- `tracking/track_window.py` (or a top-level function re-exported from
  `tracking/__init__.py`) — `track_window(frames: np.ndarray[seq_size, H, W]) -> dict | None`,
  wiring preprocessing → detection → tracking → scoring, and returning the
  winning track's estimate for the window's **center frame**.

## Hyperparameters

Carried over from `motion_via_blob_tracking.py` as starting points, exposed as
function arguments (not hardcoded), since they need dataset-specific retuning:

- MOG2: `bg_frames`, `var_threshold` — same defaults as `master_thesis`.
- Morphology: `close_kernel_size`, `open_kernel_size` — same defaults.
- Shape filter: `min_area`, `min_solidity` — same defaults.
- Kalman gating (`MAX_DIST`): **rescaled**, not copied as-is. `master_thesis`'s
  original (`MAX_DIST=80`) was tuned at 1024×1024. `data/kth_staged` frames are
  native KTH resolution, 160×120. Scaling by width (matching the
  horizontal-motion convention throughout): `160/1024 ≈ 0.156` → `MAX_DIST ≈
  12.5`. Starting point to verify empirically, not a final value.
- Blob-merge radius (`MERGE_RADIUS`): **do not** resolution-rescale this one —
  tried during implementation and empirically wrong. It needs to track how
  large a person actually is in pixels (a framing/zoom property), not raw
  frame resolution: `master_thesis`'s people apparently occupy a much smaller
  fraction of their 1024×1024 frame than KTH's do of 160×120 (measured
  directly from KTH ground truth: person height ≈ 90-95px in a 120px-tall
  frame, ~78% of frame height). Resolution-ratio scaling gave `MERGE_RADIUS ≈
  23`, which was too small to bridge KTH's fragmented body parts (verified:
  tracker locked onto an isolated head/shoulder fragment ~35px from the
  ground-truth whole-person centroid). Deriving it instead from measured
  person height (`≈ half of ~93px`) gives `MERGE_RADIUS ≈ 50`, which passes
  the validation check below (11.3px from ground truth). General lesson: when
  porting a tuned geometric threshold across datasets, rescale by the
  *quantity the threshold is actually about* (object size here), not by
  whatever's most convenient to compute (frame resolution) — they only agree
  when camera framing/zoom is comparable across datasets, which it wasn't
  here.
- `min_track_length` may need lowering from `master_thesis`'s default of 3,
  since a `seq_size`-length window (e.g. 7 frames, matching `config/train_config.py`'s
  current `seq_size=7`) leaves limited headroom.

## Validation

One assert-based self-check (no framework, no fixtures), run against real data:
pick a real KTH sequence + window from `data/kth_staged`, run `track_window`,
and assert the winning track's estimated center for the window's target frame
is within a pixel tolerance of that frame's ground-truth bbox center (from
`groundtruth.txt`, via the existing `parse_bbs`). This is preferred over an
integration-image/sharpness proxy (used in `master_thesis`'s own validation)
because KTH ground truth is directly available here — a direct numeric
comparison is strictly more rigorous than an indirect visual/sharpness signal.
The integration-image approach remains a reasonable follow-up for NFO windows
later, as a second independent signal, but isn't needed now.

## Follow-ups (explicitly not now)

- Wiring `track_window` over NFO windows.
- Persisting tracking results / a batch runner.
- The blob→heatmap localization NN consuming `track_window`'s output.
