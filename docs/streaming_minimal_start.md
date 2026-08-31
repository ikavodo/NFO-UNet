# Minimal starting point: streaming tracker on a fake stream, with detection overlay

> **2026-08-31, second finding: A FIRING RATE IS NOT A DETECTION RATE. The "HOG fires on
> 30.0% of centre-frame crops" figure below is not evidence that HOG detects the person - it
> was never checked for WHERE it fired.** Re-measured on data/ido_walk.mkv (person 421px at
> scale 0.5), HOG on a person-shaped centre-frame crop centred on the tracker's own position,
> over the 277 tracked person-present frames:
>
> | HOG SVM margin cutoff | frames firing | mean best margin | IoU(best HOG box, merged tracker box) |
> |---|---|---|---|
> | >= 0 (default) | 11 / 277 = 4.0% | 0.28 | median **0.09**, fraction > 0.3: **0.00** |
> | >= -0.5 | 151 / 277 = 54.4% | 0.18 | boxes land on leaves and stems |
>
> **Not one of the 11 positive-margin detections overlaps the tracked person by IoU > 0.3.**
> HOG is firing on vertical foliage, which produces pedestrian-like oriented-gradient patterns.
> Corroborated from the other side: on data/ido_rotate.mkv, where the plant fills the frame and
> the tracker's winning-track scores are 0-4 (matching ido_walk's person-ABSENT regime, median
> 2, against 22-151 when present), HOG fires on 84.5% of frames at margin >= -0.5 with a HIGHER
> mean margin (0.495) than on the clip that does contain a person. A detector more confident on
> the person-free clip is detecting the plant.
>
> So **HOG does not work on this footage either** - which strengthens the project's premise
> rather than weakening it, and makes HOG the third detector family to fail here after
> torchvision Faster R-CNN and YOLOv8n. Any future detector comparison must score IoU against a
> person box, never a bare firing rate.
>
> The "upscale 2x, mandatory" rule was the right observation with the wrong parameterisation.
> HOG's window is 64x128, so what governs firing is the person's height in the image handed to
> the detector RELATIVE TO 128, not a factor measured once at one person size. Swept via
> `--hog-target`: 128 -> 0.0%, 192 -> 0.0%, 256 -> 3.1%, 320 -> 3.1%, 384 -> 4.6% at margin
> >= 0. Nothing fires below ~2x the window height, and NFO's 195px person at "2x" is 390px -
> the old finding reproduces, now with a mechanism.
>
> **Integrated-image work stopped here by direction (2026-08-31): it is not helping.** The demo
> path (--split, annotate_split, integrated_panels) is removed. For the record,
> tracking/eval/buffer_depth.py measured that under median fusion the reachable fraction FALLS
> monotonically with depth (0.214 at T=7 to 0.007 at T=61), because the measured not-detected
> duty cycle inside the person's box is 0.59, already past the median's 1/2 breakdown point,
> while the at-least-one-clean-look fraction RISES and saturates (0.569 -> 0.821 by T~31-41).
> Buffer depth was never the lever. The align_frames fix and its regression test are kept - only
> the demo path was removed, not the core.
>
> **Environment:** cv2.HOGDescriptor was REMOVED in OpenCV 5. The detector path needs an OpenCV
> 4 interpreter (`~/miniconda3/envs/spacejam/bin/python` here, cv2 4.10.0); hog_detector()
> raises with that instruction rather than an AttributeError. Everything else runs on either.

> **ALIGNMENT BUG FOUND AND FIXED 2026-08-31 — the integration results quoted below were
> computed on a DE-ALIGNED stack and must not be reused.** `align_frames` cropped each frame
> at `anchor_for_frame(winner, t) - vx*(t - center_t)`. But `anchor_for_frame` already
> returns the person's position *at frame t*, so subtracting `vx*dt` cancelled the alignment
> and left a window fixed to `p(center_t)` in **world** coordinates. With a static occluder
> that is exactly backwards: the occluder stayed sharp and the median removed the *moving
> person*. Measured on a synthetic bar translating at a known 8 px per strided frame behind a
> static striped occluder, the "aligned" person drifted **+8.6 px per frame (52 px across the
> 7-frame window)** against +0.78 px after the fix, and the median then recovered 20% more of
> the person's own pixels. Regression test:
> `tracking/tests/sanity_check_integrate.py::check_alignment_follows_the_person`.
>
> **What this invalidates.** Every number below that came from `integrate()`: the HOG
> integrated-vs-centre comparison (30.0% vs 10.0%, and the 240-window occlusion-tercile
> table), and the quoted 19.75-vs-22.04 MAE figures. The stated *mechanism* is wrong too —
> "integration aligns the person's centroid but limbs articulate" — the person was not
> centroid-aligned at all, they were smeared across ~50-100 px of translation, which is a far
> larger effect than limb articulation. Note also that **neither** harness exists in this
> repo: `grep -rn HOGDescriptor` returns nothing, and there is no script, metric definition
> or data behind the MAE figures. Both the pro- and anti-integration measurements are
> currently unreproducible. Re-run before citing either.
>
> **The design constraint this exposed, which is closed-form.** In aligned coordinates a
> static occluder moves at -vx, so a pixel is occluded for a fraction d ~ min(1, w / (v*(T-1)))
> of the window, for occluder width w and speed v px per strided frame. The median recovers
> the pixel only when d < 1/2, i.e. **v*(T-1) > 2w**. Measured on ido_walk.mkv: |vx| median
> 16.1 px per strided frame, so the window sweeps 96 px and only occluders narrower than
> ~48 px can be cleared. Wider ones survive as smeared streaks no matter how the fusion is
> weighted — which is visible in `images/stream/split_montage.png`. That is a property of
> window length times speed, not of the fusion rule, and it is the first thing to check
> before tuning sigma.

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
