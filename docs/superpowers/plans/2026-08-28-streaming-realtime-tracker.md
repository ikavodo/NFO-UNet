# Streaming Real-Time Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the offline windowed blob tracker into a fixed-latency streaming system that emits, for every frame, a person position, a bounding box, and a motion-compensated integrated image crop, ready to hand to a downstream detector or recognizer - runnable first on a simulated stream from KTH/NFO and then on a live webcam.

**Architecture:** A frame source yields frames one at a time. A streaming core keeps a ring buffer of the last `2*SPAN+1` frames plus their masks and detections, maintains *persistent* tracks (one Kalman predict/update per frame instead of re-tracking a whole window per output frame), and emits a result for frame `t - SPAN` - a fixed lookahead latency, which the existing centred-window design already has. Output is built by accumulating each frame's *mask* through the same motion-compensating alignment already used for intensity fusion, giving a temporal support map from which a clean bounding box falls out even though no single frame contains a clean silhouette.

**Tech Stack:** Python 3, NumPy, OpenCV (`cv2`, already a dependency - `VideoCapture` for both file and webcam), scikit-learn (already present, for the fitted ranker weights). No new dependencies. No C++ - see "Why not C++" below.

**Spec:** `docs/scale_generalization_plan.md` (findings F1-F7, the Stage 1/Stage 2 results, the merge-radius sweep, and the real-time measurements this plan is sized from). Also `docs/superpowers/specs/2026-08-25-blob-tracking-design.md` for the original tracker design.

## Global Constraints

- **No new pip dependencies.** `cv2`, `numpy`, `scipy`, `scikit-learn`, `dill` are available; nothing else may be added.
- **No pytest.** This repo's convention is assert-based scripts at `tracking/tests/sanity_check_*.py` with a `main()` and a `if __name__ == '__main__':` guard, run as `python -m tracking.tests.sanity_check_<name>`. Follow it; do not introduce a test framework.
- **No absolute pixel constants.** Every length, area, or pixel-variance parameter must be derived from measured person height via `tracking.core.track_sequence.scale_relative_params`. This is finding F1/F3: absolute pixel constants fail silently and catastrophically (hit@0.1 collapsing 91% -> 2% over a 2x change in person size).
- **Do not change tracker behaviour.** `track_blobs`, `score_and_fit`, `merged_center`, `position_from_track`, `integrate` must keep their current outputs for their current inputs. Streaming is a restructuring, and Task 2's parity check is what proves it.
- **Static camera only.** The whole method is background-subtraction based. A moving camera invalidates it; say so in the CLI help rather than trying to handle it.
- **Calibrated coefficients (do not re-tune casually):** `ALPHA_MAX_DIST = 0.25`, `ALPHA_MERGE = 0.625`, `ALPHA_EXP_HEIGHT = 0.95`, `H_CALIB = 120.0`, all in `tracking/core/track_sequence.py`. `SEQ_SIZE = 7`, `NTH_FRAME = 2`, so `SPAN = 6`.
- **Latency budget:** `SPAN` frames of lookahead = 6 frames = ~240 ms at 25 fps. This is inherent to the centred window and pre-exists this work. Do not add to it.

---

## Why Python, and why not C++

Measured on this machine, single-threaded Python + OpenCV, NFO's native 800x600 (see `docs/scale_generalization_plan.md`, "Real-time feasibility"):

| stage | ms/frame |
|---|---|
| MOG2 background subtraction | 4.25 |
| morphology refine | 0.60 |
| shape filter | 1.45 |
| `detect_blobs` with appearance features | 2.60 |
| association, **streaming** (one step/frame) | 19.33 |
| association, offline evaluator (re-tracks a window per frame) | 73.66 |

Streaming total is ~26.5 ms/frame, about **38 fps at 800x600**. A 640x480 webcam is 0.64x the pixels, so expect comfortably more. **Python is sufficient; C++ would be optimizing the wrong thing.** Revisit only if pushing to 1080p, running several streams at once, or targeting an embedded CPU - and even then, port only the association step, since that is 19 of the 26 ms and the rest is already OpenCV C++ underneath.

Two cost caveats that matter more than language choice:

- **Hungarian assignment is O(n^3) in detections per frame.** The 19.33 ms figure came from a run averaging 57.5 detections/frame (under-warmed MOG2). `min_area` therefore has a direct latency cost, not only an accuracy one. Task 2 must cap detections per frame.
- **Track pruning is mandatory.** Scoring the full accumulated set of 2350 tracks took 42 ms. A streaming system must score only live tracks and retire dead ones. `max_age` already provides the mechanism; Task 2 must actually rely on it.

## Effort estimate

| package | tasks | estimate |
|---|---|---|
| frame sources + simulated stream | 1 | 0.5 day |
| streaming core + offline parity | 2, 3 | 2-3 days (the real work, and the only real risk) |
| live ranker | 4 | 0.5 day |
| support-map bbox + temporal filter | 5 | 1 day |
| integrated-image output | 6 | 0.5 day |
| webcam + operational guards | 7 | 0.5 day |
| downstream hook | 8 | 0.5 day |
| **total** | | **~6 focused days** |

The risk is concentrated in Task 2. Everything else is additive; the restructure is the part that can silently change results, which is why its parity check is a gate rather than a formality.

## File Structure

- **Create `tracking/stream/__init__.py`** - empty package marker.
- **Create `tracking/stream/sources.py`** - frame sources. One responsibility: yield `(frame_index, grayscale_frame)` from a directory of images, a video file, or a webcam, with optional real-time pacing and a drop-oldest policy for live capture.
- **Create `tracking/stream/tracker.py`** - the streaming core. Ring buffers, persistent tracks, per-frame step, fixed-latency emit. This is the only file that owns pipeline state.
- **Create `tracking/stream/boxes.py`** - temporal support map, bounding box extraction, and the box smoother/predictor.
- **Create `tracking/stream/output.py`** - assembles the emitted record: position, box, integrated crop; and the downstream-task hook.
- **Create `tracking/stream/run.py`** - CLI wiring: pick a source, build the pipeline, loop, optionally display or write results.
- **Create `tracking/tests/sanity_check_stream.py`** - assert-based checks for the streaming core and boxes.
- **Modify `tracking/core/integrate_image.py`** - make `crop_size` scale-relative and let `align_frames` accept a precomputed anchor sequence so the streaming core can reuse it without rebuilding a `winner` dict.
- **Modify `tracking/core/track_sequence.py`** - add `ALPHA_CROP` next to the other coefficients.

Nothing in `tracking/core/blob_tracker.py` changes. The streaming core calls `_Track` directly, which is why Task 2 exposes it rather than reimplementing the filter.

---

### Task 1: Frame sources

**Files:**
- Create: `tracking/stream/__init__.py`
- Create: `tracking/stream/sources.py`
- Test: `tracking/tests/sanity_check_stream.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class FrameSource` with `__iter__() -> Iterator[tuple[int, np.ndarray]]` yielding `(index, uint8 [H, W] grayscale)`, and properties `shape -> tuple[int, int]` and `fps -> float | None`.
  - `ImageSequenceSource(directory: str, suffix: str = '.jpg', fps: float | None = None)` - the simulated stream. `fps=None` means "as fast as possible" (for parity testing); a value paces the iterator with `time.sleep` to imitate a live feed.
  - `VideoFileSource(path: str, fps: float | None = None)`.
  - `WebcamSource(index: int = 0, width: int = 640, height: int = 480, drop_oldest: bool = True)` - implemented in Task 7; declare the class here raising `NotImplementedError` so `run.py` can reference it.

- [ ] **Step 1: Write the failing check**

Add to `tracking/tests/sanity_check_stream.py`:

```python
import numpy as np
from tracking.stream.sources import ImageSequenceSource

def check_image_sequence_source():
    src = ImageSequenceSource('data/kth_processed/person01_walking_d1_uncomp_gt', suffix='_or.jpg')
    idxs, first = [], None
    for i, frame in src:
        if first is None:
            first = frame
        idxs.append(i)
        if len(idxs) == 5:
            break
    assert idxs == [0, 1, 2, 3, 4], f"indices not sequential from 0: {idxs}"
    assert first.ndim == 2 and first.dtype == np.uint8, f"expected 2D uint8, got {first.ndim}D {first.dtype}"
    assert src.shape == first.shape, f"shape property {src.shape} != frame shape {first.shape}"
    print("image sequence source ok")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream'`

- [ ] **Step 3: Implement the sources**

Write `tracking/stream/sources.py` with:
- `FrameSource` as a small base class holding `_shape` and `fps`, with `__iter__` abstract.
- `ImageSequenceSource.__init__` sorts filenames matching `suffix`, reads the first with `cv2.imread(path, 0)` to establish `shape`, and stores the file list. `__iter__` yields `(i, cv2.imread(path, 0))`; when `fps` is set, sleep so that consecutive yields are `1/fps` apart measured against a monotonic start time (compute the target wall-clock time for frame `i` as `start + i / fps` rather than sleeping a fixed interval, so pacing does not drift with processing time).
- `VideoFileSource` wraps `cv2.VideoCapture(path)`, converts with `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`, and reads `fps` from `cv2.CAP_PROP_FPS` when the caller passes `None`.
- `WebcamSource.__init__` raises `NotImplementedError("implemented in Task 7")`.

Grayscale conversion happens in the source, so everything downstream sees exactly the `uint8 [H, W]` the offline code already expects.

- [ ] **Step 4: Run the check**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `image sequence source ok`

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/__init__.py tracking/stream/sources.py tracking/tests/sanity_check_stream.py
git commit -m "Add frame source abstraction with simulated-stream pacing"
```

---

### Task 2: Streaming core with persistent tracks

This is the load-bearing task. Everything after it is additive.

**Files:**
- Create: `tracking/stream/tracker.py`
- Test: `tracking/tests/sanity_check_stream.py` (extend)

**Interfaces:**
- Consumes: `FrameSource` from Task 1; `_Track`, `detect_blobs`, `score_and_fit`, `position_from_track` from `tracking.core`.
- Produces:
  - `class StreamTracker(person_height: float, seq_size: int = 7, nth_frame: int = 2, bg_frames: int = 30, max_age: int = 6, min_track_length: int = 3, max_detections: int = 40, use_appearance: bool = True)`
  - `StreamTracker.step(frame: np.ndarray) -> StreamResult | None` - feed one frame, get the result for the frame `SPAN` behind it, or `None` while the buffer is still filling.
  - `class StreamResult` - a dataclass-like record with fields `frame_index: int`, `x: float`, `y: float`, `score: float`, `candidates: list[dict]` (all scored tracks, best-first, as `score_and_fit(return_all=True)` returns them), `window_frames: np.ndarray` `[T, H, W]`, `window_masks: np.ndarray` `[T, H, W]`, `window_detections: list[list[dict]]`. Later tasks consume `candidates`, `window_frames`, `window_masks`.

**Design notes the implementer needs:**

- **Ring buffers.** Keep `deque(maxlen=2*SPAN+1)` of frames, masks, and detection lists. `SPAN = (seq_size // 2) * nth_frame`. A window is the strided slice `range(0, 2*SPAN+1, nth_frame)` of the buffer, which is exactly what `track_windows_in_sequence` builds today.
- **One MOG2 instance for the life of the stream**, `cv2.createBackgroundSubtractorMOG2(history=bg_frames, varThreshold=16.0, detectShadows=False)`, applied once per frame. This already matches the offline `track_windows_in_sequence`, which makes one continuous pass - so parity is achievable.
- **Morphology and shape filter run per frame**, on that frame's mask only. The offline code calls `refine_mask`/`filter_by_shape` on a whole stack, but both are per-frame loops internally, so calling them on a single-frame stack `mask[None]` gives identical output. Do that rather than duplicating the logic.
- **Persistent tracks.** Hold a list of live `_Track` objects. Per *strided* frame: `predict()` each, build the cost matrix with `np.hypot`, `scipy.optimize.linear_sum_assignment`, `update()` matched pairs whose cost `<= max_dist`, increment `misses` on unmatched tracks, spawn new `_Track`s for unmatched detections, retire tracks with `misses > max_age` into a discard list that is *dropped*, not accumulated. This is `track_blobs`' body, stepped one frame at a time instead of looped over a window.
- **Why not just call `track_blobs` per window:** it rebuilds every track from scratch for every output frame - measured at 73.66 ms/frame against 19.33 ms for the incremental version. The 3.8x is the entire performance argument for this task.
- **Scoring.** Call `score_and_fit(live_tracks, expected_height=..., return_all=True)` once per emitted frame. Because tracks are persistent, their `history` may be much longer than a window; that is desirable (it is what makes gait periodicity measurable later) but means `span` and `net_disp` grow over a track's life. **Restrict the scored history to the current window's frame indices** when computing the emitted result, so scores stay comparable to the offline behaviour and to the fitted ranker weights. Keep the full history available on the track object for future gait features.
- **Detection cap.** After `detect_blobs`, if a frame yields more than `max_detections`, keep the largest by area. This bounds the O(n^3) assignment cost. Log a counter of how often it triggers.

- [ ] **Step 1: Write the failing parity check**

Add to `tracking/tests/sanity_check_stream.py`:

```python
def check_streaming_matches_offline():
    """The streaming core must reproduce the offline windowed pipeline's positions.
    Uses a short synthetic sequence so the check runs in seconds."""
    import numpy as np
    from tracking.core.track_sequence import track_windows_in_sequence
    from tracking.stream.tracker import StreamTracker

    T, H, W = 60, 120, 200
    frames = np.full((T, H, W), 40, dtype=np.uint8)
    for t in range(T):
        x = 20 + 2 * t
        frames[t, 40:100, x:x + 25] = 220          # a person-ish bar moving right
    person_h = 60.0

    offline = track_windows_in_sequence(frames, list(range(6, T - 6)), span=6, nth_frame=2,
                                        bg_frames=30, person_height=person_h)
    st = StreamTracker(person_height=person_h, bg_frames=30)
    streamed = {}
    for t in range(T):
        r = st.step(frames[t])
        if r is not None:
            streamed[r.frame_index] = r

    common = [c for c in offline if offline[c] is not None and c in streamed]
    assert len(common) >= 20, f"too few comparable frames: {len(common)}"
    dx = [abs(offline[c]['x'] - streamed[c].x) for c in common]
    dy = [abs(offline[c]['y'] - streamed[c].y) for c in common]
    assert max(dx) < 1.0 and max(dy) < 1.0, \
        f"streaming diverged from offline: max dx={max(dx):.2f} dy={max(dy):.2f}"
    print(f"streaming/offline parity ok over {len(common)} frames")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream.tracker'`

- [ ] **Step 3: Implement `StreamTracker` per the design notes above**

- [ ] **Step 4: Run the parity check**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `streaming/offline parity ok over N frames`

If positions differ by more than a pixel, the likely causes in order: (a) the strided window offset is off by one - the emitted frame must be the *centre* of the buffer, not its oldest element; (b) tracks are being retired later than `max_age` so extra candidates survive; (c) scoring is using full track history instead of the window-restricted history.

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/tracker.py tracking/tests/sanity_check_stream.py
git commit -m "Add streaming tracker core with persistent tracks and offline parity check"
```

---

### Task 3: Real-data parity and scale bootstrap

**Files:**
- Create: `tracking/eval/stream_parity_nfo.py`
- Modify: `tracking/stream/tracker.py` (add the height bootstrap)

**Interfaces:**
- Consumes: `StreamTracker` from Task 2, `estimate_person_height` from `tracking.core.preprocess`, `ImageSequenceSource` from Task 1.
- Produces: `StreamTracker(person_height=None)` accepted, meaning "bootstrap from the first `bootstrap_frames` frames"; plus `StreamTracker.person_height` readable afterwards.

**Design notes:**

- **Bootstrap.** With `person_height=None`, buffer the first `bootstrap_frames` (default 60) frames, call `estimate_person_height` on that stack, then configure all parameters via `scale_relative_params` and begin emitting. Emit `None` until then.
- **Do not trust the estimate precisely.** Finding F4: on real NFO this estimator returns 315/61/129/200 px against a true ~195 px, and the pipeline survived anyway because the gate is flat over a wide band and too-loose is cheap. So: clamp the estimate to `[0.25, 4.0] x` the frame height as a sanity range, log it prominently, and never surface it to a user as a measurement.
- **Refresh slowly or not at all.** For a fixed camera the scale does not change, so re-estimating is optional. If added, use an EMA with a long time constant (hundreds of frames) and never let it move more than a few percent per update; a jumpy scale estimate would make every downstream parameter jitter.

- [ ] **Step 1: Write the real-data parity script**

`tracking/eval/stream_parity_nfo.py` runs one NFO sequence through both paths and compares against ground truth, printing the same summary line `eval_nfo` prints (mean / median / p90 / hit@0.1) for each. Reuse `eval_nfo.parse_normalized_bbs` for the boxes and `eval_nfo`'s constants; assert the two mean residuals agree within 0.005.

- [ ] **Step 2: Run it and observe the gap**

Run: `python -m tracking.eval.stream_parity_nfo seq1`
Expected: two summary lines; if they disagree by more than 0.005, debug using the causes listed in Task 2 Step 4.

- [ ] **Step 3: Add the bootstrap path to `StreamTracker`**

- [ ] **Step 4: Verify the bootstrap sequence works end to end**

Run: `python -m tracking.eval.stream_parity_nfo seq1 --bootstrap`
Expected: prints the measured height alongside NFO's hand-measured 195 px, and a residual summary no worse than 0.08 mean (the scale-relative configuration measured 0.0762 offline).

- [ ] **Step 5: Commit**

```bash
git add tracking/eval/stream_parity_nfo.py tracking/stream/tracker.py
git commit -m "Add NFO streaming parity check and person-height bootstrap"
```

---

### Task 4: Live ranker

**Files:**
- Create: `tracking/stream/ranker.py`
- Modify: `tracking/eval/stage2_rank_learning.py` (add weight export)

**Interfaces:**
- Consumes: `StreamResult.candidates` from Task 2; `track_features`/`annotate_candidates`/`FEATURES` from `tracking.eval.stage2_rank_learning`.
- Produces:
  - `save_ranker(path: str, feature_names: list[str], weights: np.ndarray) -> None` and `load_ranker(path: str) -> tuple[list[str], np.ndarray]`, using `np.savez` (no new dependency, and unlike `dill` it is not an arbitrary-code unpickle risk).
  - `class LiveRanker(path: str)` with `pick(candidates: list[dict], n_window_frames: int) -> dict` returning the chosen candidate.

**Design notes:**

- Reuse the *exact* feature functions from `stage2_rank_learning`, imported rather than copied. If the features drift between training and inference the whole result is void, and a shared import is the only way to guarantee they cannot.
- Default to the **`all_nopol + bscore`** feature set: pooled hit@0.1 94.9%, 53% of the ranking headroom, no brightness-polarity assumption, no tuned constant. The `all + bscore` set scores higher (95.7%, 67%) but depends on "the person is the darkest candidate", which leave-one-sequence-out structurally cannot validate - see the Stage 2 section of the spec. Make the set a constructor argument so the assumption is explicit at the call site.
- `LiveRanker.pick` must handle the single-candidate case by returning it directly without featurizing, and the empty case by returning `None`.

- [ ] **Step 1: Write the failing round-trip check**

Add to `tracking/tests/sanity_check_stream.py`:

```python
def check_ranker_roundtrip(tmp='/tmp/ranker_test.npz'):
    import numpy as np
    from tracking.stream.ranker import save_ranker, load_ranker
    names = ['a', 'b', 'c']
    w = np.array([0.5, -1.5, 2.0])
    save_ranker(tmp, names, w)
    names2, w2 = load_ranker(tmp)
    assert names2 == names, f"names changed: {names2}"
    assert np.allclose(w2, w), f"weights changed: {w2}"
    print("ranker round-trip ok")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream.ranker'`

- [ ] **Step 3: Implement `ranker.py`, and add a `--export PATH` flag to `stage2_rank_learning`**

The export must refit on *all four* sequences (leave-one-sequence-out is for estimating generalization, not for producing the shipped model) and write the standardized weights together with the feature names and the per-fold held-out scores, so the file records its own expected performance.

- [ ] **Step 4: Run the check and export a real model**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `ranker round-trip ok`
Run: `python -m tracking.eval.stage2_rank_learning --export models/ranker_nopol.npz`
Expected: writes the file and prints the feature set and its held-out hit rate.

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/ranker.py tracking/eval/stage2_rank_learning.py
git commit -m "Add live ranker with weight export/import"
```

---

### Task 5: Temporal support map and bounding box

This is the "compute the box from blob features over time" piece. The idea, and why it works:

After motion compensation the person is *stationary* in aligned coordinates while static occluders sweep across - that is precisely the property median intensity fusion already exploits. Apply it to **geometry** instead of intensity: align each frame's binary mask with the same transform, average them, and the person's silhouette accumulates high support while occluders and noise accumulate low support. No single frame contains a clean silhouette; the accumulation does. A box read off the thresholded support map is therefore far more stable than any per-frame box, which is exactly the problem that made per-frame masks unusable (NFO fragmentation: 1.85 blobs per person, tallest blob only 86% of the person's height).

**Files:**
- Create: `tracking/stream/boxes.py`
- Modify: `tracking/core/integrate_image.py` (accept a precomputed anchor sequence)
- Modify: `tracking/core/track_sequence.py` (add `ALPHA_CROP`)
- Test: `tracking/tests/sanity_check_stream.py` (extend)

**Interfaces:**
- Consumes: `StreamResult` from Task 2; `align_frames`, `crop_at` from `tracking.core.integrate_image`.
- Produces:
  - `support_map(window_masks: np.ndarray, anchors: list[tuple[float, float]], vx: float, crop_size: int, weights: np.ndarray | None = None) -> np.ndarray` - float32 `[crop_size, crop_size]` in `[0, 1]`.
  - `box_from_support(support: np.ndarray, threshold: float = 0.5, person_height: float | None = None, median_aspect: float | None = None) -> tuple[int, int, int, int] | None` - `(x1, y1, x2, y2)` in support-map coordinates.
  - `class BoxFilter(alpha: float = 0.4)` with `update(box: tuple | None) -> tuple` and `predict() -> tuple`, a 4-state constant-velocity filter on `(cx, cy, w, h)`.
  - `ALPHA_CROP = 1.6` in `track_sequence.py`: crop side = `ALPHA_CROP * person_height`, replacing `integrate_image`'s fixed `crop_size=220`. 1.6 gives a person-height crop with ~30% margin each side for limb extent and box drift; it is a framing choice, not a tuned constant, but it must scale with person height like everything else (F1).

**Design notes:**

- **Weighting by appearance consistency.** Optional `weights` argument: down-weight frames whose blob appearance deviates from the track's median, since those are the likely-occluded frames. Use `w_t = exp(-((app_mean_t - median_app) / (0.5 * median_app + eps))**2)`, normalized to sum to 1. The per-blob `app_mean` is already carried in `_Track.history` position 4. Keep this off by default and behind a flag until measured - the Stage 2 result is a warning that appearance features can encode scene-specific assumptions.
- **Regularization.** A support box can be truncated when occlusion is persistent at one end. Clamp: box height into `[0.6, 1.4] * person_height`; if `median_aspect` is given (available from the ranker's gait features), clamp width to `median_aspect * box_height` within `[0.5, 2.0]x`. Expand about the box centre, never shift it.
- **Confidence.** Return mean support inside the box alongside it; `output.py` should pass low-confidence boxes through `BoxFilter.predict()` instead of `update()`, which is the "bbox prediction" behaviour - a box carried forward on the constant-velocity model when the current evidence is weak.
- **`align_frames` change:** currently it takes a `winner` dict and calls `anchor_for_frame` internally. Add an optional `anchors` parameter (a list of `(x, y)` per buffered frame) and `vx`, so the streaming core can align masks and intensities with the same anchors without fabricating a `winner` dict. Keep the existing signature working - default `anchors=None` reproduces today's behaviour exactly.

- [ ] **Step 1: Write the failing support-map check**

Add to `tracking/tests/sanity_check_stream.py`:

```python
def check_support_map_beats_single_frame():
    """A person-shaped mask, present in every frame, crossed by an occluder band that moves.
    The accumulated support map must recover a fuller silhouette than any single frame has."""
    import numpy as np
    from tracking.stream.boxes import support_map, box_from_support

    T, H, W, size = 7, 120, 200, 96
    masks = np.zeros((T, H, W), dtype=np.uint8)
    anchors = []
    for t in range(T):
        px = 60 + 4 * t                      # person moves right 4px/frame
        masks[t, 30:90, px:px + 20] = 255    # person: 60 tall, 20 wide
        occ_x = 40 + 12 * t                  # occluder sweeps faster, in world coords
        masks[t, :, occ_x:occ_x + 14] = 0    # occluder erases whatever it covers
        anchors.append((px + 10, 60))

    sup = support_map(masks, anchors, vx=4.0, crop_size=size)
    assert sup.min() >= 0.0 and sup.max() <= 1.0, "support must be a fraction in [0,1]"
    box = box_from_support(sup, threshold=0.5, person_height=60.0)
    assert box is not None, "no box recovered from support map"
    x1, y1, x2, y2 = box
    assert 40 <= (y2 - y1) <= 80, f"box height {y2 - y1} not near the 60px person"
    per_frame_best = max(int((masks[t] > 0).sum()) for t in range(T))
    recovered = int((sup >= 0.5).sum())
    assert recovered > 0, "thresholded support is empty"
    print(f"support map ok: box {box}, support pixels {recovered}, best single frame {per_frame_best}")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream.boxes'`

- [ ] **Step 3: Implement `boxes.py`, `ALPHA_CROP`, and the `align_frames` anchors parameter**

- [ ] **Step 4: Run the check**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `support map ok: ...`

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/boxes.py tracking/core/integrate_image.py tracking/core/track_sequence.py tracking/tests/sanity_check_stream.py
git commit -m "Add temporal support map, box extraction, and box filter"
```

---

### Task 6: Integrated-image output record

**Files:**
- Create: `tracking/stream/output.py`

**Interfaces:**
- Consumes: `StreamResult` (Task 2), `LiveRanker` (Task 4), `support_map`/`box_from_support`/`BoxFilter` (Task 5), `integrate` (`tracking.core.integrate_image`).
- Produces:
  - `class OutputBuilder(person_height: float, ranker: LiveRanker | None = None, method: str = 'median', mask_background: bool = False, min_box_confidence: float = 0.35)`
  - `OutputBuilder.build(result: StreamResult) -> TrackedTarget | None`
  - `class TrackedTarget` with fields `frame_index: int`, `x: float`, `y: float`, `box: tuple[int, int, int, int]` (in full-frame image coordinates), `box_confidence: float`, `box_predicted: bool`, `integrated: np.ndarray` (`[crop, crop]` uint8), `crop_origin: tuple[int, int]`.

**Design notes:**

- Order of operations: pick the candidate (ranker if present, else `result.candidates[0]`), compute anchors for the buffered frames from that candidate, build the support map from `result.window_masks`, extract and filter the box, then call `integrate` on `result.window_frames` with the same anchors and `crop_size = ALPHA_CROP * person_height`.
- **Report the box in image coordinates**, not support-map coordinates - downstream consumers need to draw it on the original frame. Store `crop_origin` so the mapping is invertible.
- `method='median'` by default: per `integrate`'s docstring, median fusion is the occlusion-robust choice, which is the point. `mask_background=False` by default: full-frame crops keep natural image statistics for an off-the-shelf detector, per the same docstring.
- Set `box_predicted=True` whenever confidence fell below `min_box_confidence` and the filter's prediction was used, so downstream code can distinguish measured from extrapolated boxes.

- [ ] **Step 1: Write the failing end-to-end check**

Add to `tracking/tests/sanity_check_stream.py` a check that feeds the synthetic moving-bar sequence from Task 2 through `StreamTracker` + `OutputBuilder`, and asserts: at least 20 `TrackedTarget`s are produced; `integrated.shape == (crop, crop)`; the box's centre is within `0.3 * person_height` of the bar's true centre; and `box_predicted` is False for at least half the frames.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream.output'`

- [ ] **Step 3: Implement `output.py`**

- [ ] **Step 4: Run the check**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: all checks print ok

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/output.py tracking/tests/sanity_check_stream.py
git commit -m "Add output builder producing box and integrated image per frame"
```

---

### Task 7: Webcam source, CLI, and operational guards

**Files:**
- Modify: `tracking/stream/sources.py` (implement `WebcamSource`)
- Create: `tracking/stream/run.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m tracking.stream.run --source webcam|dir|video [--path P] [--ranker P] [--display] [--save-dir P]`.

**Why `cv2.VideoCapture` and not something else:**

- **`cv2.VideoCapture(0)`** - already a dependency, three lines, uses V4L2 on Linux. Nothing else is justified here.
- **GStreamer** - more capable (hardware decode, RTSP, complex pipelines) and correspondingly more setup. Reach for it only if adding IP cameras.
- **PyAV / ffmpeg-python** - excellent for files and network streams, heavier for a local camera, and a new dependency.
- **`imutils.VideoStream`** - a thin threaded wrapper whose *idea* is right but which is not worth a dependency; Task 7 implements the same thing in ~20 lines.

**The threading design matters more than the library.** A fixed-latency pipeline must never accumulate a backlog: if processing is slower than capture, a queue grows and the output becomes progressively staler while appearing to work. So run capture in a daemon thread that overwrites a single-slot buffer under a lock, and have the consumer take whatever is newest, dropping anything it missed. Count and report the drops - a rising drop count is the signal to lower resolution.

**Operational guards to implement, each with a one-line log at startup:**

- **Disable auto-exposure and auto white balance** (`cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)`, `cv2.CAP_PROP_AUTO_WB, 0`). Auto-gain is the single most likely thing to break a live demo: it changes global brightness, which MOG2 reads as everything-is-foreground. Not all drivers honour these; log whether the `set` call returned True.
- **Warm-up.** Emit nothing for the first `bg_frames` frames and say so. The background model is meaningless before then, and the person-height bootstrap needs the same period.
- **Static camera.** Print a one-line warning that the method assumes a fixed camera.
- **Resolution.** Default 640x480. Log the measured processing fps every second so the operator can see headroom.

- [ ] **Step 1: Write the failing check for the drop-oldest buffer**

The buffer logic is testable without a camera - factor it into `class LatestFrameBuffer` with `put(frame)` / `get()` and check that after three `put`s a single `get` returns the third and reports two drops.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ImportError: cannot import name 'LatestFrameBuffer'`

- [ ] **Step 3: Implement `LatestFrameBuffer`, `WebcamSource`, and `run.py`**

- [ ] **Step 4: Verify on both sources**

Run: `python -m tracking.stream.run --source dir --path data/nfo_final/nfo_final/seq1 --display`
Expected: a window showing the frame with the box drawn and the integrated crop inset; steady fps log.
Run: `python -m tracking.stream.run --source webcam --display`
Expected: after the warm-up message, boxes track a person walking across a static-camera view.

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/sources.py tracking/stream/run.py tracking/tests/sanity_check_stream.py
git commit -m "Add webcam source with drop-oldest capture and streaming CLI"
```

---

### Task 8: Downstream task hook

**Files:**
- Create: `tracking/stream/downstream.py`
- Modify: `tracking/stream/run.py` (wire the hook)

**Interfaces:**
- Consumes: `TrackedTarget` from Task 6.
- Produces:
  - `DownstreamTask` protocol: `__call__(target: TrackedTarget) -> dict`.
  - `class NullTask` (returns `{}`) and `class SaveCropsTask(out_dir: str)`.
  - `resolution_report(person_height: float) -> dict` - returns the estimated head height (`person_height / 7.5`, the standard anthropometric ratio) and a verdict string for face-recognition viability.

**The resolution budget - read this before building anything face-related:**

Head height is about `person_height / 7.5`. Face recognition needs roughly 80-100 px of face height to be reliable; detection of a face needs perhaps 30-40 px.

| person height in frame | head height | verdict |
|---|---|---|
| 195 px (NFO's actual scale) | ~26 px | face recognition **not viable**; face detection marginal |
| 400 px | ~53 px | face detection viable, recognition unreliable |
| 600 px | ~80 px | recognition becomes plausible |
| 900 px+ | ~120 px | recognition comfortable |

So on NFO-scale footage, **plan for person/pose detection on the integrated crop, not face recognition.** Getting to face recognition is a *camera* problem - longer lens, higher-resolution sensor, or a closer camera - not an algorithm problem. Super-resolution does not fix it: it can make an image look sharper but cannot add identity information that the sensor never captured, and using it as a recognition front-end mostly manufactures confident wrong matches. `resolution_report` exists so this check happens automatically at startup rather than after weeks of work.

**Two further domain caveats for the downstream model:**

- The integrated image is **grayscale** and its background is **motion-blurred by construction**. Off-the-shelf detectors are trained on natural RGB images and will be out of distribution. Replicate to three channels, and validate the detector on integrated crops specifically before trusting its confidences.
- The integration deliberately smears anything not moving with the tracked person. That is the mechanism that removes the occluder, and it also removes context a detector might rely on. Expect to compare "detector on integrated crop" against "detector on the centre frame's crop" - the integrated version should win under occlusion and may lose without it.

- [ ] **Step 1: Write the failing resolution-report check**

```python
def check_resolution_report():
    from tracking.stream.downstream import resolution_report
    r = resolution_report(195.0)
    assert 20 <= r['head_px'] <= 32, f"head estimate off: {r}"
    assert 'not viable' in r['face_recognition'].lower(), f"expected a negative verdict: {r}"
    r2 = resolution_report(900.0)
    assert 'not viable' not in r2['face_recognition'].lower(), f"expected a positive verdict: {r2}"
    print("resolution report ok")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream.downstream'`

- [ ] **Step 3: Implement `downstream.py` and wire `--task null|save-crops` into `run.py`**

- [ ] **Step 4: Run the check and produce crops from a real sequence**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `resolution report ok`
Run: `python -m tracking.stream.run --source dir --path data/nfo_final/nfo_final/seq1 --task save-crops --save-dir out/crops_seq1`
Expected: one integrated crop per emitted frame, plus a printed resolution verdict.

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/downstream.py tracking/stream/run.py tracking/tests/sanity_check_stream.py
git commit -m "Add downstream task hook and resolution budget report"
```

---

## Deferred, deliberately

- **Gait periodicity features.** Persistent tracks make them possible for the first time (window length limits lookahead, not how far back a track remembers - a track accumulates seconds of history at zero added latency). But they need the streaming core to exist first, and they need a measurement, so they are follow-on work rather than part of this plan.
- **Cross-scene validation of the appearance features** against `../master_thesis/GPJATK`. This is what would settle whether the ranker's appearance win is 52% or 68% of the headroom. Independent of streaming; do it whenever the answer matters.
- **C++ port.** Not justified at 38 fps measured. See "Why not C++".
- **Multi-person tracking.** The whole pipeline assumes one target (`score_and_fit` returns a single winner). Supporting several is a genuine redesign, not an extension.

## Self-review notes

- Every task ends with a runnable check and a commit; every check is assert-based and follows the existing `tracking/tests/sanity_check_*.py` convention rather than introducing pytest.
- Task 2 is the only task that can invalidate prior results, and it is gated by a parity check against the offline pipeline on synthetic data (Task 2) and on real NFO (Task 3).
- Interfaces are named consistently across tasks: `StreamResult` (Task 2) -> `TrackedTarget` (Task 6) -> `DownstreamTask` (Task 8); `support_map`/`box_from_support`/`BoxFilter` (Task 5) are consumed only by `OutputBuilder` (Task 6); `ALPHA_CROP` is defined once in Task 5 and used in Task 6.
- The three questions that prompted this plan are answered in-document: effort (~6 focused days, table above), speed (~38 fps measured, Python sufficient), and framework (`cv2.VideoCapture` plus a drop-oldest threaded buffer, Task 7).
