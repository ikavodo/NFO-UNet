# Streaming Real-Time Tracker Implementation Plan (3-day version)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live-looking demo: an NFO sequence played as a fake stream, with the tracked person's box and a motion-compensated integrated crop shown per frame, and an object detector's boxes plus confidences drawn on that crop.

**Architecture:** One class, `StreamPipeline`, holds a ring buffer of the last 13 frames and exposes `step(frame) -> Result | None`, emitting for the frame 6 behind the newest. Day 1 implements `step` by calling the *existing, already-correct* windowed code on the buffer. Day 2 swaps the internals for persistent tracks behind the same API, validated by diffing the two engines frame by frame. That ordering is the whole point of this revision: **there is a working demo at the end of Day 1**, and the risky restructure becomes an optimization with a fallback rather than a prerequisite.

**Tech Stack:** Python 3, NumPy, OpenCV, SciPy. No new dependencies. Detector is OpenCV's built-in HOG people detector, which needs no model download and returns confidences directly.

**Spec:** `docs/streaming_minimal_start.md` (the dependency closure and the three things needed), `docs/scale_generalization_plan.md` (evidence behind every constant).

## Global Constraints

- **No new pip dependencies.** `cv2`, `numpy`, `scipy` are enough for all three days.
- **No pytest.** Repo convention is assert-based scripts at `tracking/tests/sanity_check_*.py` with a `main()`, run as `python -m tracking.tests.sanity_check_<name>`.
- **No absolute pixel constants.** Every length or area comes from person height via `tracking.core.track_sequence.scale_relative_params`. Finding F1: absolute constants fail silently, accuracy collapsing ~91% -> ~2% over a 2x change in person size.
- **Do not change existing tracker behaviour.** Two small additive changes are allowed and specified below (`merged_center(..., return_box=)`, `integrate_image` scale-relative crop). Nothing else in `tracking/core/` changes semantics.
- **Calibrated coefficients:** `ALPHA_MAX_DIST = 0.25`, `ALPHA_MERGE = 0.625`, `ALPHA_EXP_HEIGHT = 0.95`, `H_CALIB = 120.0`. `SEQ_SIZE = 7`, `NTH_FRAME = 2`, so `SPAN = 6` and the buffer is 13 frames.
- **Latency:** 6 frames of lookahead, inherent to the centred window. Do not add to it.

## Premises already de-risked by measurement - do not re-litigate these

Run before writing any code, on 60 real seq1 windows, HOG at 2x upscale (at 1x it never fires
at all - upscaling is mandatory, not optional):

| crop fed to HOG | fires on | mean best confidence |
|---|---|---|
| **centre frame only** | **30.0%** | 0.46 |
| gaussian fusion, sigma=1.0 | 13.3% | 0.47 |
| median fusion (the occlusion-robust default) | 10.0% | 0.45 |
| gaussian sigma=2.0 | 8.3% | 0.36 |
| gaussian sigma=0.6 | 8.3% | 0.32 |
| median, tighter 1.15x crop | 3.3% | 0.59 |

**HOG detects the integrated crop 2-3x WORSE than the plain centre frame, and temporal gaussian
weighting does not rescue it.** The reason is structural rather than a tuning failure: integration
aligns the person's *centroid*, but a walking person's limbs articulate within the 13-frame span,
so the fused image has a sharp torso and smeared legs - and HOG is a histogram of oriented
gradients over the whole 64x128 window, legs included. Integration buys occlusion robustness by
spending exactly the edge crispness HOG measures.

Consequences for this plan, already applied below:

1. **Run the detector on the centre-frame crop, and display the integrated crop beside it.** The
   deliverable becomes tracker box + integrated crop + detections with confidence, which is what
   was asked for, and additionally gives the integrated-vs-centre comparison for free - which is
   the genuinely interesting measurement, not a consolation prize.
2. **If detection quality matters, swap HOG for a `cv2.dnn` ONNX detector.** `cv2.dnn` ships with
   OpenCV so there is no new pip dependency, but it does need a one-off ~10-25 MB model download,
   which is the only external artifact anywhere in this plan. A CNN detector is far more robust to
   blur than HOG, so it is also the only route by which the integrated crop could plausibly *beat*
   the centre frame. Worth an hour on Day 3 if the comparison is the point.
3. `integrate()` takes `crop_size` directly and has no `person_height` parameter - compute
   `int(round(1.6 * person_height))` at the call site. That removes one of the two planned changes
   to `tracking/core/`.

`DISPLAY` is set on this machine, so `cv2.imshow` works; `--no-display --save-dir` remains for
headless runs.

## What was cut from the 6-day version, and why

| dropped | why it is safe to drop |
|---|---|
| Webcam source, threaded capture, auto-exposure guards | The goal is a fake stream. Live camera is a separate afternoon once the pipeline exists. |
| Person-height bootstrap from the footage | For NFO we know the answer (~195 px) and F4 says the estimator is unreliable on NFO anyway (it returns 61-315 px). Pass height as a CLI argument. |
| Learned ranker integration | Worth 90.0% -> 94.9% hit, but `score_and_fit`'s argmax already works. Pure accuracy upgrade, zero structural risk, do it later. |
| Separate `sources.py`, `boxes.py`, `output.py`, `downstream.py`, `run.py` | Five files for a demo is bookkeeping. Two files carry all of it. |
| Support-map bounding box | Day 1 gets a box nearly free from `merged_center`, which already computes a merged bbox internally and throws away everything but its centre. The support map is a Day 3 upgrade, not a dependency. |
| `resolution_report` module | One printed warning line in the demo does the same job. |

Everything dropped is additive. Nothing dropped is load-bearing for the demo.

## Fastest possible schedule (unlimited parallel workers)

The 3-day figure assumes one worker going task by task. With parallel workers the binding
constraint stops being work volume and becomes **serialization plus wrong premises** - which is
why the de-risk table above exists: 15 minutes of compute invalidated a headline assumption that
would otherwise have cost a day of implementation.

**The interface is the only real dependency.** `Result` and `StreamPipeline.step()` are fully
specified in Task 1, and nothing else needs the *implementation* - only the contract. So:

| phase | serial? | who waits on what |
|---|---|---|
| 0. Freeze the contract: `Result` fields, `step()` signature, `merged_center(return_box=)` | **serial**, ~30 min | everything |
| 1a. `pipeline.py` windowed engine (Task 1) | parallel | contract only |
| 1b. `demo.py`: `iter_directory`, `detect_on_crop`, `annotate`, CLI (Task 2) | parallel | contract only - build against a hand-made `Result` stub |
| 1c. incremental engine (Task 3) | parallel | contract only |
| 1d. `support_box` (Task 4) | parallel | contract only |
| 2. Integrate 1a+1b, run on seq1 | **serial**, ~1 h | 1a, 1b |
| 3. Engine-parity diff and fix | **serial**, ~1-2 h, the main risk | 1a, 1c |
| 4. Eyeball on seq1 and seq2 | **serial**, ~15 min, human | 2, 3 |

**Realistic wall clock: half a day to a day**, not three. Everything in phase 1 is independent
because each piece touches a different file and depends only on names fixed in phase 0.

Two rules that make the parallelism actually pay:

- **Workers must not share files.** 1a and 1c both touch `pipeline.py`, which is the one
  collision. Resolve it by having 1c write its engine as a standalone function
  `step_incremental(state, window) -> winner` in its own module, which 1a's class then calls -
  or simply let 1c start after 1a lands, since 1a is the smaller job.
- **Give every worker the de-risk table.** The single largest time sink available is a worker
  independently rediscovering that HOG does not fire on integrated crops.

Where the remaining risk sits, in order: (i) the engine-parity diff failing for the three reasons
listed in Task 3, (ii) `score_and_fit` behaving differently once track history is unbounded, which
is the same issue seen from the other side, (iii) nothing else - Tasks 1, 2 and 4 are wiring over
code that already works and is already measured.

## File Structure

- **Create `tracking/stream/__init__.py`** - empty package marker.
- **Create `tracking/stream/pipeline.py`** - `StreamPipeline` (ring buffer, `step`, both engines) and `Result`. The only file holding pipeline state.
- **Create `tracking/stream/demo.py`** - frame iteration from a directory, HOG overlay, display/save, CLI `main()`.
- **Create `tracking/tests/sanity_check_stream.py`** - assert-based checks.
- **Modify `tracking/core/blob_tracker.py`** - add `return_box` to `merged_center`.
- **Modify `tracking/core/track_window.py`** - have `position_from_track` optionally return the merged box too.
- **Modify `tracking/core/integrate_image.py`** - `crop_size=None` means `1.6 * person_height`.

---

## DAY 1 - end-to-end demo using existing tracker code

### Task 1: Ring-buffer pipeline over the existing windowed tracker

**Files:**
- Create: `tracking/stream/__init__.py`, `tracking/stream/pipeline.py`
- Modify: `tracking/core/blob_tracker.py`, `tracking/core/track_window.py`
- Test: `tracking/tests/sanity_check_stream.py`

**Interfaces:**
- Consumes: `foreground_mask`, `refine_mask`, `filter_by_shape` (`tracking.core.preprocess`); `detect_blobs`, `track_blobs`, `score_and_fit` (`tracking.core.blob_tracker`); `position_from_track` (`tracking.core.track_window`); `scale_relative_params` (`tracking.core.track_sequence`).
- Produces:
  - `class Result` with fields `frame_index: int`, `x: float`, `y: float`, `box: tuple[int, int, int, int] | None`, `window_frames: np.ndarray`, `window_masks: np.ndarray`, `window_detections: list[list[dict]]`, `winner: dict | None`.
  - `class StreamPipeline(person_height: float, engine: str = 'window', seq_size: int = 7, nth_frame: int = 2, bg_frames: int = 30, max_detections: int = 40)` with `step(frame: np.ndarray) -> Result | None` and attribute `span: int`.
  - `merged_center(detections_at_frame, anchor_x, anchor_y, merge_radius, return_box=False)` - returns `(cx, cy)` as before, or `(cx, cy, (x1, y1, x2, y2))` when `return_box=True`; returns `None` for the box when no detection is in range.
  - `position_from_track(winner, window_detections, merge_radius, return_box=False)` - same pattern; the result dict gains a `'box'` key when asked.

**Design notes:**

- **Ring buffers.** Three `collections.deque(maxlen=2*span+1)` for frames, masks and detection lists. `span = (seq_size // 2) * nth_frame = 6`, so `maxlen = 13`.
- **Per-frame work in `step`:** apply the single persistent MOG2 instance to the new frame, then `refine_mask(mask[None])[0]` and `filter_by_shape(mask[None])[0]` - both are per-frame loops internally, so calling them on a one-frame stack is identical to the batch call and avoids duplicating logic. Then `detect_blobs` on that one frame, capped at `max_detections` largest by area (Hungarian is O(n^3), so this bounds worst-case cost).
- **Emit** only once the buffer is full: the emitted frame is the buffer's *centre*, index `span` from the oldest, and its absolute index is `frames_seen - 1 - span`.
- **`engine='window'`:** take the strided window `[buf[i] for i in range(0, 2*span+1, nth_frame)]` and call `track_blobs` then `score_and_fit` then `position_from_track(..., return_box=True)`. This is exactly what `track_windows_in_sequence` does per centre, so it is correct by construction - that is why Day 1 is low-risk.
- **`merged_center` change:** the function already computes `x1, y1, x2, y2` and returns only the centre. Return the box as well under the flag. Two lines, no behaviour change for existing callers.
- **Configuration** comes from `scale_relative_params(person_height)`; do not hardcode anything.

- [ ] **Step 1: Write the failing check**

```python
import numpy as np
from tracking.stream.pipeline import StreamPipeline

def make_moving_bar(T=40, H=120, W=200, speed=3):
    frames = np.full((T, H, W), 40, dtype=np.uint8)
    for t in range(T):
        x = 20 + speed * t
        frames[t, 40:100, x:x + 25] = 220
    return frames

def check_pipeline_emits_after_span():
    frames = make_moving_bar()
    pipe = StreamPipeline(person_height=60.0, engine='window')
    results = []
    for t in range(len(frames)):
        r = pipe.step(frames[t])
        if r is not None:
            results.append(r)
    assert pipe.span == 6, f"expected span 6, got {pipe.span}"
    assert len(results) == len(frames) - 2 * pipe.span, \
        f"expected {len(frames) - 2 * pipe.span} emissions, got {len(results)}"
    assert results[0].frame_index == pipe.span, \
        f"first emission should be for frame {pipe.span}, got {results[0].frame_index}"
    xs = [r.x for r in results if r.x is not None]
    assert xs == sorted(xs), "tracked x should increase monotonically for a rightward bar"
    boxed = [r for r in results if r.box is not None]
    assert len(boxed) > len(results) // 2, f"only {len(boxed)}/{len(results)} results carry a box"
    print(f"pipeline ok: {len(results)} emissions, {len(boxed)} with boxes")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream'`

- [ ] **Step 3: Implement `merged_center(return_box=)`, `position_from_track(return_box=)`, and `StreamPipeline` with `engine='window'`**

- [ ] **Step 4: Run the check**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `pipeline ok: 28 emissions, N with boxes`

- [ ] **Step 5: Commit**

```bash
git add tracking/stream tracking/core/blob_tracker.py tracking/core/track_window.py tracking/tests/sanity_check_stream.py
git commit -m "Add ring-buffer stream pipeline over the existing windowed tracker"
```

---

### Task 2: Demo - fake stream, integrated crop, HOG overlay

**Files:**
- Create: `tracking/stream/demo.py`
- Modify: `tracking/core/integrate_image.py`

**Interfaces:**
- Consumes: `StreamPipeline`, `Result` (Task 1); `integrate` (`tracking.core.integrate_image`).
- Produces:
  - `iter_directory(path: str, suffix: str = '.jpg', fps: float | None = None) -> Iterator[np.ndarray]` - yields grayscale frames; `fps` paces against a monotonic start time so pacing does not drift with processing time.
  - `detect_on_crop(crop: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]` - HOG boxes with confidences.
  - `annotate(frame_bgr, result, crop, detections) -> np.ndarray` - the display frame: tracker box on the full frame, integrated crop inset top-right with detector boxes and confidences drawn on it.
  - `main()` behind `python -m tracking.stream.demo --path DIR [--person-height 195] [--fps 25] [--save-dir DIR] [--no-display]`.

**Design notes:**

- **No `integrate_image` change needed.** It already takes `crop_size`; compute `crop = int(round(1.6 * person_height))` in `demo.py` and pass it. 1.6 gives the person plus ~30% margin each side for limb extent and box drift.
- **Integrated crop:** `integrate(result.window_frames, result.winner, crop_size=crop, method='median')`. Median is the occlusion-robust fusion and is what makes the integrated view worth showing; `mask_background=False` keeps natural image statistics.
- **Detect on the CENTRE-FRAME crop, not the integrated one** - measured 30.0% vs 10.0% firing, see the de-risk table above. Get the centre crop with `align_frames(result.window_frames, result.winner, crop_size=crop)[len(...)//2]`, which is the same geometry as the integrated crop so the two are directly comparable side by side. Draw detections on both panels and label each with its firing status; that comparison is a deliverable in itself.
- **HOG:**

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(crop, winStride=(8, 8), padding=(8, 8), scale=1.05)
```

  Build the detector **once** at module import or in `main`, never per frame. `weights` are SVM decision values; draw each as e.g. `f"{w:.2f}"`. **Upscale the crop 2x before detecting and scale the boxes back** - at 1x, HOG fired on 0% of 60 real windows, measured. This is mandatory, not a tuning nicety.
- **Print one warning line at startup** if `person_height / 7.5 < 40`, saying face-level tasks are not viable at this scale and detection is the realistic downstream task. NFO's 195 px gives a ~26 px head.
- **Display:** `cv2.imshow` plus `cv2.waitKey(1)`; `--no-display` for headless runs, `--save-dir` writes annotated PNGs. Print measured fps once a second so the operator sees headroom.

- [ ] **Step 1: Write the failing check for the detector wrapper**

```python
def check_detect_on_crop_shape():
    import numpy as np
    from tracking.stream.demo import detect_on_crop
    crop = np.zeros((312, 312), dtype=np.uint8)
    crop[80:240, 130:190] = 200          # a crude person-shaped light region
    dets = detect_on_crop(crop)
    assert isinstance(dets, list), f"expected a list, got {type(dets)}"
    for box, conf in dets:
        assert len(box) == 4, f"box should be 4 numbers, got {box}"
        assert isinstance(float(conf), float)
    print(f"detector wrapper ok: {len(dets)} detections on the synthetic crop")
```

Note: this checks the *contract*, not that HOG fires on a rectangle - it may legitimately return zero detections.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ModuleNotFoundError: No module named 'tracking.stream.demo'`

- [ ] **Step 3: Implement `integrate_image`'s scale-relative crop, then `demo.py`**

- [ ] **Step 4: Run the check, then the demo end to end**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `detector wrapper ok: ...`
Run: `python -m tracking.stream.demo --path data/nfo_final/nfo_final/seq1 --person-height 195 --fps 25`
Expected: a window showing seq1 with the tracker box following the person, the integrated crop inset, detector boxes and confidences on the crop, and a steady fps line. **This is the deliverable.**

- [ ] **Step 5: Commit**

```bash
git add tracking/stream/demo.py tracking/core/integrate_image.py tracking/tests/sanity_check_stream.py
git commit -m "Add fake-stream demo with integrated crop and HOG detection overlay"
```

---

## DAY 2 - make it actually streaming

### Task 3: Persistent-track engine, gated by diffing against Day 1

**Files:**
- Modify: `tracking/stream/pipeline.py`
- Test: `tracking/tests/sanity_check_stream.py`

**Interfaces:**
- Consumes: `_Track` (`tracking.core.blob_tracker`), `linear_sum_assignment` (`scipy.optimize`).
- Produces: `StreamPipeline(engine='incremental')`, same `step` signature and same `Result` fields. No new public names.

**Design notes:**

- **Why:** measured 19.33 ms/frame against the windowed engine's 73.66, a 3.8x speedup, taking the whole pipeline to ~26.5 ms/frame or ~38 fps at 800x600. That is the entire reason for this task.
- **The state** is a list of live `_Track` objects. Per *strided* frame: `predict()` each, build the cost matrix with `np.hypot` over predicted positions and detections, `linear_sum_assignment`, `update()` pairs whose cost `<= max_dist`, increment `misses` on unmatched tracks, spawn `_Track` for unmatched detections, and **drop** tracks with `misses > max_age` rather than accumulating them. Dropping matters: scoring 2350 accumulated tracks measured 42 ms.
- **Scoring:** call `score_and_fit` on the live tracks, but **restrict each track's history to the current window's frame indices** before scoring. Persistent tracks can be seconds long, and `span`/`net_disp` would otherwise grow without bound and stop matching both the windowed engine and the calibrated coefficients. Keep the full history on the track object - it is what makes gait features possible later.
- **The gate is a diff, not an approximation.** Both engines are kept, so the check runs the same frames through both and requires the positions to agree. That is a stronger and cheaper test than comparing against the offline evaluator.

- [ ] **Step 1: Write the failing engine-parity check**

```python
def check_engines_agree():
    import numpy as np
    from tracking.stream.pipeline import StreamPipeline
    frames = make_moving_bar(T=60)
    out = {}
    for engine in ('window', 'incremental'):
        pipe = StreamPipeline(person_height=60.0, engine=engine)
        got = {}
        for t in range(len(frames)):
            r = pipe.step(frames[t])
            if r is not None:
                got[r.frame_index] = r
        out[engine] = got
    common = sorted(set(out['window']) & set(out['incremental']))
    assert len(common) >= 20, f"too few comparable frames: {len(common)}"
    dx = [abs(out['window'][c].x - out['incremental'][c].x) for c in common]
    dy = [abs(out['window'][c].y - out['incremental'][c].y) for c in common]
    assert max(dx) < 1.0 and max(dy) < 1.0, \
        f"engines disagree: max dx={max(dx):.2f}, dy={max(dy):.2f}"
    print(f"engine parity ok over {len(common)} frames")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: fails inside `StreamPipeline` with an unknown-engine error

- [ ] **Step 3: Implement the incremental engine**

- [ ] **Step 4: Run the parity check**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `engine parity ok over N frames`

If they disagree, in likelihood order: (a) history not restricted to the window before scoring; (b) tracks retired later than `max_age`, leaving extra candidates; (c) the strided window offset is off by one - the emitted frame must be the buffer's centre.

- [ ] **Step 5: Verify accuracy did not move, on real data**

Run: `python -m tracking.stream.demo --path data/nfo_final/nfo_final/seq1 --person-height 195 --engine incremental --no-display --save-dir /tmp/inc`
Run the same with `--engine window`. Compare the printed mean residual against ground truth: they must agree to within 0.005, and both should land near `eval_nfo`'s 0.0661 for this configuration.

- [ ] **Step 6: Commit**

```bash
git add tracking/stream/pipeline.py tracking/tests/sanity_check_stream.py
git commit -m "Add persistent-track streaming engine with cross-engine parity check"
```

---

## DAY 3 - slack, then one upgrade if there is time

Day 3 is deliberately mostly buffer. If Days 1-2 ran long, this is where they land. If they did not, do Task 4, then stop.

### Task 4 (optional): Temporal support-map box

Only start this if the demo works and the engines agree. It is an accuracy upgrade to the *box*, not a dependency of anything.

**Files:**
- Modify: `tracking/stream/pipeline.py` (or add `tracking/stream/boxes.py` if it grows past ~60 lines)

**Interfaces:**
- Produces: `support_box(window_masks, winner, person_height, threshold=0.5) -> tuple[tuple[int,int,int,int] | None, float]` - box in support-map coordinates plus a confidence.

**Why this works:** after motion compensation the person is *stationary* in aligned coordinates while static occluders sweep across - the property median intensity fusion already exploits. Apply it to geometry: align each frame's binary mask with the same transform `align_frames` uses for intensity, average them into a support map in `[0, 1]`, threshold at 0.5, take the largest connected component. The person accumulates support; occluders and noise do not. This matters because no single frame has a clean silhouette - measured on NFO, a person is 1.85 blobs with the tallest covering only 86% of their height - but the accumulation does.

**Design notes:**
- `align_frames` currently takes a `winner` dict and computes anchors internally; it works on any `[T, H, W]` stack, so it can align masks unchanged. Pass `window_masks` directly.
- Clamp the resulting box height into `[0.6, 1.4] * person_height`, expanding about the centre, never shifting it. Persistent occlusion at one end otherwise truncates the box.
- Confidence is mean support inside the box. Below ~0.35, prefer the Day 1 `merged_center` box - do not invent a filter yet.

- [ ] **Step 1: Write the failing check**

```python
def check_support_box_recovers_silhouette():
    import numpy as np
    from tracking.stream.pipeline import StreamPipeline
    from tracking.stream.boxes import support_box   # or wherever it landed
    frames = make_moving_bar(T=40, speed=3)
    pipe = StreamPipeline(person_height=60.0, engine='window')
    last = None
    for t in range(len(frames)):
        r = pipe.step(frames[t])
        if r is not None:
            last = r
    assert last is not None and last.winner is not None, "no tracked result to test with"
    box, conf = support_box(last.window_masks, last.winner, person_height=60.0)
    assert box is not None, "no support box recovered"
    h = box[3] - box[1]
    assert 36 <= h <= 84, f"box height {h} not within [0.6, 1.4] x 60px"
    assert 0.0 <= conf <= 1.0, f"confidence out of range: {conf}"
    print(f"support box ok: {box}, confidence {conf:.2f}")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: `ImportError: cannot import name 'support_box'`

- [ ] **Step 3: Implement `support_box` and use it in the demo when confidence is adequate**

- [ ] **Step 4: Run the check and eyeball the demo**

Run: `python -m tracking.tests.sanity_check_stream`
Expected: prints `support box ok: ...`
Run the demo on seq2, which is the cluttered sequence, and confirm the box is visibly steadier than the Day 1 version.

- [ ] **Step 5: Commit**

```bash
git add tracking/stream tracking/tests/sanity_check_stream.py
git commit -m "Add temporal support-map bounding box"
```

---

## Deferred (was in the 6-day plan, still worth doing later)

- **Webcam source** - `cv2.VideoCapture(0)` plus a daemon thread writing a single-slot buffer, consumer takes the newest and drops the rest. A fixed-latency pipeline must never accumulate a backlog. Disable auto-exposure and auto white balance (`CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_AUTO_WB`): auto-gain shifts global brightness and MOG2 reads that as everything-is-foreground. Static camera only.
- **Learned ranker** - 90.0% -> 94.9% hit@0.1, or 52% of the measured ranking headroom, using the `all_nopol + bscore` feature set. Export weights from `tracking/eval/stage2_rank_learning.py` with `np.savez` and swap `score_and_fit`'s argmax for the ranker's.
- **Person-height bootstrap** - `estimate_person_height` over the first ~60 frames, for footage whose scale is unknown. Clamp and log it; F4 says do not trust it as a measurement.
- **Gait periodicity features** - persistent tracks make these possible for the first time, since window length limits lookahead but not how far back a track remembers.
- **Better detector** - a `cv2.dnn` ONNX model behind `detect_on_crop`'s signature, once the pipeline is proven and there is a reason to care about detector quality.
- **GPJATK cross-scene validation** - the only way to settle whether the appearance features' win is 52% or 68% of headroom.

## Self-review notes

- Spec coverage: fake stream (Task 2 `iter_directory`), streaming core (Tasks 1 and 3), detection overlay with certainty (Task 2 `detect_on_crop` + `annotate`), integrated image (Task 2), box (Task 1 cheap version, Task 4 upgrade). All four items the user asked for are on Days 1-2; Day 3 is slack.
- The demo exists at the end of Day 1 and never regresses: Day 2 changes only `StreamPipeline`'s internals and is gated by a diff against Day 1's engine, which is retained rather than deleted.
- Names are consistent across tasks: `StreamPipeline`/`Result` (Task 1) are consumed by `demo.py` (Task 2) and re-implemented internally in Task 3 with no signature change; `merged_center(return_box=)` (Task 1) feeds `Result.box`; `support_box` (Task 4) is the only new public name after Day 1.
- Every task ends with a runnable assert-based check following the existing `sanity_check_*` convention, and a commit.
