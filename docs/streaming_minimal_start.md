# Minimal starting point: streaming tracker on a fake stream, with detection overlay

Read this first if you are starting a fresh session to implement the streaming tracker. It
exists to make the *surface area* minimal - the set of things you must read and may touch -
rather than to minimise the file count of the repository.

**The plan:** `docs/superpowers/plans/2026-08-28-streaming-realtime-tracker.md` (8 tasks).
**The evidence behind every constant:** `docs/scale_generalization_plan.md`.

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

**2. The streaming core.** Persistent tracks stepped once per frame instead of the offline code's
per-window re-tracking. Measured: 19.3 ms/frame against 73.7, i.e. the entire reason to do this.
Emit results for frame `t - 6`; that 6-frame lookahead already exists in the offline design and
must not grow. Task 2, and its parity check against `track_windows_in_sequence` is the gate that
proves the restructure did not change results.

**3. Detection overlay with confidence.** Use OpenCV's built-in HOG people detector - **zero new
dependencies**, and it returns confidences directly:

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(integrated_crop, winStride=(8, 8))
```

`weights` are SVM decision values - use them as the certainty to draw next to each box. Verified
working in this environment. It runs on grayscale, which suits the integrated image directly.

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

## Suggested first milestone

Fake stream from `seq1` -> streaming core -> integrated crop -> HOG boxes with confidences drawn
on top, displayed live, with the offline parity check passing. That is Tasks 1, 2 and a trimmed
Task 6/8 from the plan, and it is a self-contained demo you can show.
