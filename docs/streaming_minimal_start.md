# Minimal starting point: streaming tracker on a fake stream, with detection overlay

Read this first if you are starting a fresh session to implement the streaming tracker. It
exists to make the *surface area* minimal - the set of things you must read and may touch -
rather than to minimise the file count of the repository.

**The plan:** `docs/superpowers/plans/2026-08-28-streaming-realtime-tracker.md` - 4 tasks over
3 days, with a working demo at the end of Day 1.
**The evidence behind every constant:** `docs/scale_generalization_plan.md`.

**The one ordering decision that makes 3 days feasible:** the demo does *not* need the streaming
core. The existing per-window code already runs at ~12 fps, and it is already known-correct. So
Day 1 wires the existing code behind a `StreamPipeline.step(frame)` API and gets the full demo
working; Day 2 swaps the internals for persistent tracks (the measured 3.8x speedup) behind that
same API, gated by diffing the two engines against each other. The risky part becomes an
optimization with a fallback instead of a prerequisite, and there is something to show either way.

## Everything the streaming tracker depends on

Five files, 669 lines, and they import **nothing** from this repository outside themselves -
only `numpy`, `cv2`, `scipy`. No torch, no `config/`, no `dataset/`, no `network/`, no `utils/`.

| file | lines | what you need from it |
|---|---|---|
| `tracking/core/preprocess.py` | 133 | `foreground_mask` (MOG2), `refine_mask`, `filter_by_shape`, `estimate_person_height` |
| `tracking/core/blob_tracker.py` | 201 | `detect_blobs`, `_Track`, `track_blobs`, `score_and_fit`, `merged_center` |
| `tracking/core/track_window.py` | 96 | `position_from_track` |
| `tracking/core/track_sequence.py` | 126 | `scale_relative_params`, the `ALPHA_*` coefficients; `track_windows_in_sequence` is needed only for the parity check |
| `tracking/core/integrate_image.py` | 113 | `align_frames`, `fuse`, `integrate`, `crop_at`, `anchor_for_frame` |

The other ~55 Python files in the repo are the U-Net training stack and the offline
experiments. They are **inert**: nothing on the tracking path imports them. Ignore them; do not
delete them.

## The three things the minimal build needs

**1. A fake stream.** Read an NFO sequence frame by frame and hand frames to the tracker one at
a time, optionally paced to a target fps. `data/nfo_final/nfo_final/seq1` is 800x600 grayscale
JPEGs with ground truth in `groundtruth*.txt`, so the fake stream doubles as the accuracy check.
Task 1 of the plan.

**2. A ring buffer, then (only on Day 2) the streaming core.** Keep the last 13 frames, emit for
the frame 6 behind the newest - that 6-frame lookahead already exists in the offline design and
must not grow. Day 1 fills that buffer and calls the existing windowed code on it. Day 2 replaces
the internals with persistent tracks stepped once per frame: measured 19.3 ms/frame against 73.7,
which is the entire reason to bother, and the gate is a frame-by-frame diff against Day 1's
engine rather than an approximation.

**3. Detection overlay with confidence.** Use OpenCV's built-in HOG people detector - **zero new
dependencies**, and it returns confidences directly:

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(integrated_crop, winStride=(8, 8))
```

`weights` are SVM decision values - use them as the certainty to draw next to each box.

**Two things measured on 60 real seq1 windows before writing any code, both mandatory:**

- **Upscale the crop 2x before detecting.** At 1x, HOG fired on 0% of windows. Not a nicety.
- **Detect on the CENTRE-FRAME crop, not the integrated one.** HOG fires on 30.0% of centre-frame
  crops against 10.0% of median-integrated crops, and temporal gaussian weighting does not rescue
  it (13.3% at its best sigma). The reason is structural: integration aligns the person's
  *centroid*, but limbs articulate across the 13-frame span, so the fused image has a sharp torso
  and smeared legs - and HOG is a histogram of oriented gradients over the whole window, legs
  included. Integration buys occlusion robustness by spending exactly the edge crispness HOG
  measures.

Stratifying by how occluded the centre frame actually is does not rescue integration either - the
centre frame wins in all three occlusion terciles (13.8/16.2/10.0% against median fusion's
7.5/12.5/2.5%), including the most-occluded one, over 240 windows. And the integrated arm was given
*more* tuning freedom than the baseline throughout, so the comparison already favours it.

So the demo shows **both** panels side by side - integrated crop and centre-frame crop, same
geometry - and runs the detector on the centre frame. That gives the integrated-vs-centre
comparison for free, which is the more interesting measurement anyway.

**Do not plan to swap in a CNN detector: measured, they find nothing here.** `torchvision`
Faster R-CNN and `ultralytics YOLOv8n` both return 0 persons with top-any-class confidence 0.000 on
NFO frames, KTH frames, and every crop variant, while both find 4 people at 0.87-0.999 in a stock
photo - and still do after greyscaling it, so greyscale is not the cause. The footage is (KTH
intensity std 15.3; NFO's person small and behind foliage). **HOG is the only detector that works
here at all.** That is also the cleanest evidence for why this tracker exists: if an off-the-shelf
detector worked on this footage, none of this machinery would be needed.

Integration's value is *reconstruction*, not detectability - if you want to show it pays, measure
error on the pixels occluded in the centre frame, not whether a gradient detector fires.

Why HOG rather than a modern detector: it needs no model download, no new dependency, and no GPU,
so the first end-to-end version can exist in an afternoon. Its accuracy is mediocre by current
standards, and that is fine - the point of the minimal build is to prove the *pipeline* end to
end, and to compare "detector on integrated crop" against "detector on the plain centre-frame
crop". Swap in a `cv2.dnn` model (YOLO/SSD ONNX) later, behind the same interface, once the
pipeline works.

## Two numbers to know before you start

- **`ALPHA_MERGE = 0.625`, `ALPHA_MAX_DIST = 0.25`** and friends in `track_sequence.py` are
  measured, not guessed, and they are expressed as multiples of person height on purpose.
  Reintroducing an absolute pixel constant anywhere is the one change guaranteed to break this:
  accuracy collapses from ~91% to ~2% over a 2x change in person size, silently, with no error.
- **Face recognition is not viable at this scale.** Head height is about person_height/7.5, so
  NFO's 195px person gives a ~26px head. Person detection on the integrated crop is the
  realistic downstream task; recognition needs a ~600px person, which is a camera change, not an
  algorithm change.

## The box is nearly free - do not build machinery for it

`merged_center` in `blob_tracker.py` already computes the merged bounding box of the detections
near the tracked anchor and then returns only its centre. Adding a `return_box=True` flag is two
lines and gives you a usable box on Day 1. The fancier version - averaging motion-aligned masks
into a temporal support map, which recovers a clean silhouette even though no single frame has
one - is a Day 3 upgrade, not a dependency.

## Three files, not six

`tracking/stream/pipeline.py` (ring buffer, `step`, both engines) and `tracking/stream/demo.py`
(frame iteration, HOG overlay, display, CLI), plus checks in
`tracking/tests/sanity_check_stream.py`. Two small additive changes to `tracking/core`:
`merged_center(return_box=)` and a scale-relative `crop_size` in `integrate_image`.

## The Day 1 deliverable

```
python -m tracking.stream.demo --path data/nfo_final/nfo_final/seq1 --person-height 195 --fps 25
```

seq1 playing as a stream, tracker box following the person, integrated crop inset, HOG boxes and
confidences drawn on the crop, steady fps line. Self-contained and showable.
