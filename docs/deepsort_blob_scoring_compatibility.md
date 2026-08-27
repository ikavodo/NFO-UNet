# DeepSORT-style scoring for the classical blob tracker - compatibility and scale-robustness assessment

**Goal:** the classical Kalman+Hungarian tracker (`tracking/core/blob_tracker.py`) currently
beats the trained U-Net on NFO, using hand-tuned, dataset-specific constants. Two open
questions from prior discussion: (1) is DeepSORT's learned-association-score approach
compatible with the current architecture, and (2) is the current tracker (or a DeepSORT-style
version of it) robust to person scale varying with camera distance? This doc answers both from
the actual code, not from a fresh design.

## Current architecture, mapped onto DeepSORT's structure

DeepSORT (Wojke, Bewley & Paulus, 2017, "Simple Online and Realtime Tracking with a Deep
Association Metric") extends the original SORT (Bewley et al., 2016) with three pieces:
1. Constant-velocity Kalman filter per track (**already present**: `_Track` in
   `blob_tracker.py`, position+velocity state, no acceleration - identical structure).
2. **Mahalanobis-gated** motion cost instead of raw Euclidean distance, using the Kalman
   filter's own innovation covariance `S = H P H^T + R` (**partially present**: `_Track.update`
   already computes `S` and `K` every step - `track_blobs`'s cost matrix just doesn't use it;
   `track_blobs` uses `np.hypot(px - d["x"], py - d["y"])`, raw pixel distance, thresholded by a
   single scalar `max_dist`).
3. A learned **appearance embedding** (cosine distance in a Re-ID feature space), combined with
   the motion gate to disambiguate visually-distinct objects at similar predicted positions
   (**absent, but recoverable**: `blob_tracker.py`'s detections carry only `{x, y, area, bbox}` -
   no appearance descriptor. `foreground_mask`/`refine_mask`/`filter_by_shape` do re-binarize the
   mask at every stage (`preprocess.py`), so the *mask itself* is a pure 0/255 silhouette with no
   surviving intensity signal - correction from an earlier draft of this doc, which claimed no
   appearance signal exists at all. That's wrong: `detect_blobs` only ever reads the mask, never
   the raw grayscale frame the mask was computed from. `raw_frame` cropped/masked by the blob's
   own binary region (`frame[y1:y2, x1:x2] * mask[y1:y2, x1:x2]`) has real intensity variation -
   clothing texture, shading gradient - currently discarded entirely, not fundamentally absent.
   A small shape/intensity descriptor (mean/std masked intensity, gradient histogram) is
   available cheaply without a CNN; a learned Re-ID embedding is still overkill for this data,
   but "no appearance signal" was the wrong reason to rule it out).

**Verdict: structurally compatible, and closer to already-there than expected.** The
architecture skeleton (per-track Kalman filter, frame-by-frame Hungarian assignment on a cost
matrix) is identical to what DeepSORT runs. Adding Mahalanobis gating is a same-file change, not
a new component - `S` just needs to be returned from `predict()`/exposed on `_Track` and used in
`track_blobs`'s cost computation instead of raw distance. The appearance-embedding half of
DeepSORT does not transfer directly - there's no appearance to embed. What *does* transfer from
the same lineage: `score_and_fit`'s `expected_height` Gaussian-falloff term is already playing
the same role appearance embedding plays in DeepSORT (a non-motion cue that disambiguates a
motion-consistent-but-wrong candidate, i.e. the "swaying branch outscored the person" failure
mode documented in its own docstring) - it's a shape/size cue standing in for an appearance cue,
which is the right substitution given what NFO's detections actually contain.

## Scale robustness: currently absent, confirmed in the constants themselves

The literal numbers in `tracking/eval/eval_nfo.py`:
```python
MAX_DIST = 25.0        # measured GT centroid displacement (p99) at NFO's native 800x600, nth_frame=2
MERGE_RADIUS = 100.0   # measured mean NFO person height (~195px) / 2
EXPECTED_HEIGHT = 195.0  # measured mean NFO person height, NFO's native 800x600
```
its own comment: *"derived from NFO's own native-resolution ground truth, not reused/rescaled
from KTH."* All three are absolute pixel constants measured on one dataset at one resolution.
Inside `_Track`, `P` (initial covariance), `Q` (process noise), and `R` (measurement noise) are
likewise fixed absolute-pixel constants (`np.eye(4) * 50.0`, `np.eye(4) * 2.0`, `np.eye(2) *
9.0`) - none scale with the detected blob's own size. **This confirms directly, from the code,
the scale-robustness gap raised earlier**: run this tracker on a scene where people appear at
half NFO's pixel scale (further from camera), and every one of these constants is wrong by
roughly the same factor, silently - `max_dist` gates too loose, `expected_height` rejects every
real detection, `Q`/`R` under/over-trust motion relative to real per-pixel displacement.

**Does adding DeepSORT's Mahalanobis gating fix this for free? Partially, not fully.**
Mahalanobis distance normalizes the *motion* cost by the Kalman filter's own uncertainty (`S`),
which does make frame-to-frame association more forgiving of scale automatically, since `S`
grows/shrinks with whatever `Q`/`R` produce - but `Q` and `R` themselves are still the fixed
absolute constants above, so `S` doesn't correctly reflect a different scale's real uncertainty
without also fixing those. `EXPECTED_HEIGHT`/`MERGE_RADIUS` are outside DeepSORT's association
step entirely (they're `score_and_fit`/merge-stage inputs) and gain nothing from Mahalanobis
gating regardless. **Same conclusion as the earlier discussion, now grounded in the actual
numbers**: real scale robustness needs the constants themselves expressed relative to a
per-scene measured quantity (e.g. detected blob height, or a one-time calibration frame), not a
DeepSORT-style association upgrade alone. DeepSORT and scale-relative normalization are
complementary, not substitutes for each other.

## Recommended integration plan (if pursued)

1. Expose `S` (innovation covariance) from `_Track.predict()`, use Mahalanobis distance
   (`(z-Hx)^T S^-1 (z-Hx)`) in `track_blobs`'s cost matrix instead of `np.hypot(...)`, gate on
   chi-squared threshold (standard DeepSORT/SORT practice) instead of a raw `max_dist` pixel
   cutoff.
2. Do **not** attempt an appearance-embedding CNN - no appearance signal exists in binary
   foreground masks. If a non-motion disambiguation term beyond `expected_height` is wanted,
   extend the existing shape-based approach (aspect ratio, area-consistency across frames) rather
   than importing a Re-ID network built for a different kind of input.
3. Independently of DeepSORT: make `MAX_DIST`, `EXPECTED_HEIGHT`, `MERGE_RADIUS`, `Q`, `R`
   scale-relative (normalized by a measured or detected blob-height reference) before claiming
   scale robustness - this is the harder, more load-bearing change of the two.
4. Validate step 1 and step 3 **separately** - conflating them would make it impossible to tell
   which one actually fixed (or didn't fix) a given scale-robustness failure.

## Literature

- Bewley, Ge, Ott, Ramos, Upcroft (2016), **SORT** - the base architecture already matched here
  (Kalman + Hungarian on IoU/distance, no appearance).
- Wojke, Bewley, Paulus (2017), **DeepSORT** - adds Mahalanobis gating + learned appearance
  cosine-distance embedding on top of SORT.
- Wojke & Bewley (2018), "Deep Cosine Metric Learning for Person Re-Identification" - the actual
  metric-learning method DeepSORT's embedding uses; not applicable here for the reason above
  (no appearance signal in binary masks).
- Zhang et al. (2022), **ByteTrack** - same SORT lineage, but explicitly keeps and associates
  *low-confidence/partial* detections instead of discarding them before matching - directly
  relevant to this project's fragmented-occlusion setting, where a person is frequently split
  into multiple small, low-confidence blobs (head/torso/legs) rather than one clean detection.
  Worth a closer look before DeepSORT specifically, given the domain match.
- No literature found (nor searched for exhaustively here) that solves multi-scale/multi-distance
  robustness as part of the tracking-association step itself - every SORT-lineage tracker
  surveyed assumes the input scale is roughly fixed per deployment and calibrates constants
  accordingly, same as this project's current `eval_nfo.py` constants. This matches the
  conclusion above: scale robustness is a separate, unsolved problem from association quality,
  not something DeepSORT already solves as a side effect.

## Update: appearance/scoring feature plan, occluder domain-fidelity, and a kill test before training anything

Three follow-up threads, in order of how much they should actually be built before the next one.

**1. Temporal blob-dimension features (recommended first, cheapest, most interpretable).**
`_Track.history` already stores `(x, y, height)` per frame - the raw material for a scale-
*invariant* shape descriptor is already there, unused beyond `mean_height`. Concretely:
aspect-ratio mean/variance over the track, frame-to-frame size growth rate (not absolute size),
and size normalized by the track's own running median (so the descriptor's *shape*, not
magnitude, is what's compared - the actual scale-invariance trick). This is a drop-in
replacement target for `score_and_fit`'s existing hand-picked formula (same inputs, same scalar
output, learned instead of derived) - much lower risk than a fresh architecture, and targets the
exact documented failure mode (swaying-foliage track outscoring the real person).

**2. Real appearance signal exists, just needs plumbing (do second, if #1 isn't enough).**
Correction to this doc's earlier "no appearance signal" claim: the *mask* is genuinely binary at
every pipeline stage, but `detect_blobs` never reads the raw grayscale frame the mask came from.
`raw_frame` cropped/masked by the blob's own region has real intensity variation (texture,
shading) - a cheap, non-CNN shape/intensity descriptor (masked mean/std intensity, gradient
histogram) is available without new infrastructure. A learned CNN-embedding-with-residual-toward-
clean-KTH-features approach (regressing occluded-blob features toward the same frame's
unoccluded-KTH CNN features - feasible since KTH's synthetic occlusion gives exact clean/occluded
pairs for free) is a real, coherent escalation, but meaningfully heavier - full backbone, feature
pipeline, training loop - and still carries the trained-on-synthetic → deployed-on-real-NFO
domain-gap risk every learned component in this project has carried. Treat as escalation, not a
starting point.

**3. Occluder domain fidelity - adopted, then upgraded.** `generate_occlusion_branch` added to
`utils/occlusion_utils.py`, originally ported from `master_thesis/src/occluders.py:occ_branch`
(sinusoidally-swaying branch lines), then **replaced** with a cleaner port of
`~/PycharmProjects/MovingMNIST-OcclusionBench/occluders.py:branches_mask` - direct density
control (binary search over how many pre-sampled branch segments to draw, converging to a target
coverage within tolerance, plus a small erode/dilate refinement pass) and optional restriction to
a bounding box (e.g. a person's own bbox), so density is meaningful relative to the actual
occlusion target instead of the whole frame. Sway was dropped entirely on reconsideration: NFO's
occluder geometry is treated as fixed per sequence in this project's own model (see the "Key
constraint" note above) - the earlier ported version's animated sway was over-modeling motion
this project doesn't assume exists, not an unambiguous realism win. Verified: density lands
within tolerance at 0.1/0.3/0.5 targets, and bbox-restricted occlusion produces exactly zero
pixels outside the given box. Any future KTH-based training/eval for the scale-robustness work
below should use this, not `generate_occlusion_morph`, for the shape-fidelity reasons already
discussed.

**Kill test, before building any training pipeline.** Before training anything (feature-based
scorer or otherwise), run the *existing*, unmodified `score_and_fit`/`track_blobs` heuristic
against synthetic KTH+`generate_occlusion_branch` sequences resized to 2-3 different pixel-height
buckets (simulating different camera distances), using:
- (a) constants (`MAX_DIST`/`EXPECTED_HEIGHT`/`MERGE_RADIUS`/`Q`/`R`) correctly recomputed per
  scale bucket (no learning - just measuring the equivalent of `eval_nfo.py`'s constants at each
  scale), vs.
- (b) one fixed scale's constants applied to all buckets (mimicking today's actual deployment:
  calibrated once, run everywhere).

If (b) craters relative to (a), that confirms the scale-sensitivity problem is real and worth
solving (with either scale-relative normalization or a learned scorer). **If (b) doesn't degrade
much, that kills the motivation for building anything further** - it would mean the existing
heuristic is already tolerant enough across scale in practice, without any new machinery. This
test requires no training and is the cheapest possible falsification step - it should run before,
not after, committing to item 1 above.

## Start here (for whoever picks this up next, e.g. a fresh session)

Build in this order - each step is a go/no-go gate for the next one, do not skip ahead:

1. **Kill test first, no training.** Run the existing, *unmodified* `score_and_fit`/
   `track_blobs` heuristic on synthetic multi-scale KTH+`generate_occlusion_branch` sequences:
   per-scale-correct constants vs. one fixed scale's constants applied everywhere (today's
   actual deployment pattern). If the fixed-constant version doesn't meaningfully degrade
   relative to the per-scale-correct one, **the whole learned-scorer motivation dies right
   there** - stop, report that, don't proceed to step 2.
2. **Only if step 1 shows real degradation:** build the temporal blob-dimension feature scorer
   (aspect ratio, size-growth-rate, running-median-normalized size - see "Temporal blob-
   dimension features" above) as a drop-in replacement for `score_and_fit`'s hand-picked
   formula. Cheap, interpretable, targets the documented swaying-foliage-outscores-person
   failure mode directly.
3. **Only if step 2 turns out insufficient:** CNN-embedding-with-residual-toward-clean-KTH-
   features. Real and coherent (KTH's synthetic occlusion gives exact clean/occluded pairs for
   free), but meaningfully heavier - full backbone, feature pipeline, training loop - and
   carries the same synthetic→real domain-gap risk every learned component in this project has
   carried. Treat as escalation, not a starting point.

Before any of the above: `generate_occlusion_branch` (`utils/occlusion_utils.py`, now the
density-controlled/bbox-restricted, non-swaying version - see "Occluder domain fidelity" above)
still produces flat-color occlusion when composited (no per-branch intensity variation, no
Perlin-like light-canvas modulation) - fine for pure mask/geometry use (steps 1-2 above only
consume mask geometry), not yet sufficient if step 3's appearance work needs realistic-looking
composited frames.

## Open questions / not yet done

- No code changes made yet beyond `generate_occlusion_branch` (mask geometry only) - this is
  still primarily a compatibility/planning doc.
- ByteTrack's low-confidence-detection handling hasn't been compared against the current
  `min_area` hard cutoff in `detect_blobs` - possibly a more directly relevant fix than
  Mahalanobis gating, given the fragmented-blob domain.
- Scale-relative constant normalization is unimplemented and unvalidated - this doc only
  confirms the current constants are scale-specific, not fixes it.
- A classical patch-based texture-synthesis occluder (Efros-Leung/image-quilting, sampling real
  occluder texture from NFO's own no-person frames instead of a synthetic line/noise model) was
  discussed as a cheaper, no-training alternative to a learned generative occluder model, but is
  unimplemented and unspec'd beyond this note - real advantage (uses actual NFO texture
  statistics) traded against needing a first-pass occluder/background separation step on those
  reference frames.
- `generate_occlusion_branch`'s density/bbox controls are unit-tested informally (this session's
  smoke test only) - not validated against real NFO occluder coverage statistics (what fraction
  of a person's bbox is actually occluded on average in real footage), which would be the right
  reference point for choosing `density` values in the kill test.

## Step 1 result: the kill test ran. Scale sensitivity CONFIRMED - learned-scorer motivation NOT confirmed

Implemented in `tracking/eval/kill_test_scale.py` (run: `python -m tracking.eval.kill_test_scale`,
~4 min). 6 KTH `d1` sequences (person01-03 walking + jogging), 853 windows per bucket,
`seq_size=7`/`nth_frame=2` as in the training config. Buckets are whole-frame resizes of
`kth_processed`'s 224x224 (0.5x / 1x / 2x -> person height ~59 / 119 / 237px, bracketing NFO's
measured 195px). Occluder: one *static* `generate_occlusion_branch` mask per sequence at
`density=0.35`, restricted to the union of that sequence's real per-frame GT boxes, with occluded
pixels filled from the sequence's own per-pixel temporal median. Measured effect: ~36% of each
frame's real GT box covered, person mask fill inside the GT box drops 0.31 -> 0.16 (0.5x) /
0.23 -> 0.15 (1x) / 0.21 -> 0.15 (2x) - i.e. the occluder really does erase and fragment the
person, which is the point.

Metric: frame-normalized centroid residual (same definition as `eval_nfo.py`) and hit rate at
the eval pipeline's 0.1 threshold, with no-track counted as a miss.

| bucket | person px | (a) per-scale-correct | (b) fixed 1x constants | no-track (a) -> (b) |
|--------|-----------|-----------------------|------------------------|---------------------|
| 0.5x   | 59        | hit 37.0%, med 0.078  | hit **79.8%**, med 0.048 | 38.1% -> 0.8%     |
| 1.0x   | 119       | hit 61.5%, med 0.054  | hit 61.5% (identical by construction) | 15.5% -> 15.5% |
| 2.0x   | 237       | hit **74.4%**, med 0.035 | hit 2.7%, med 0.091 | 11.8% -> **95.2%** |

**(b) craters in the "people bigger than calibration" direction**: -71.7pp hit rate, no-track goes
11.8% -> 95.2%. The tracker essentially stops producing tracks at all. Repeating the run with
morphology/`min_area` frozen at their defaults instead of scaled per bucket (`--frozen-preprocess`)
gives the same picture, slightly stronger (-84.3pp, no-track 4.2% -> 83.2%), so the effect is not
an artifact of how the segmentation front-end was scaled. **The scale-sensitivity claim in this doc
is confirmed, and it is not a mild degradation - it is total failure.**

**But the leave-one-in ablation says the damage is entirely in the association gate, not in the
scoring formula.** Correcting exactly one constant per bucket and leaving the rest fixed:

| corrected constant | 0.5x (gap -42.8pp) | 2.0x (gap +71.7pp) |
|--------------------|--------------------|--------------------|
| `MAX_DIST`         | 66% of gap         | **72% of gap**     |
| `MERGE_RADIUS`     | 47% of gap         | 2% of gap          |
| `EXPECTED_HEIGHT`  | **0% of gap**      | **0% of gap**      |
| Kalman `P`/`Q`/`R` | **0% of gap**      | **0% of gap**      |

`EXPECTED_HEIGHT` is the *only* part of `score_and_fit`'s hand-picked formula that this test can
break by mis-scaling, and mis-scaling it by 2x in either direction changes nothing at all -
because it multiplies every candidate track's score by a near-identical Gaussian factor when all
candidates are fragments of the same person, so the `argmax` is unchanged. Kalman `Q`/`R` likewise
contribute nothing: the filter is only used for one-step prediction inside a gate that a wrong
`MAX_DIST` has already opened or closed.

**Go/no-go for step 2: no-go as motivated.** The failure this test found is a 1-line-fixable
absolute-pixel gate (`MAX_DIST` should be a fraction of detected person height, doc item 3), not a
weakness in the score. Training a learned temporal-feature scorer would not have fixed any of the
71.7pp. Two things follow:

1. **Do doc item 3 (scale-relative constants) first, and specifically `MAX_DIST`** - and note the
   0.5x row before copying `eval_nfo.py`'s recipe: at 0.5x the *"correctly recomputed"* p99-GT-
   displacement `MAX_DIST` was **worse than the wrong one** (38.1% no-track vs 0.8%). Blob-centroid
   jitter under fragmented occlusion does not shrink with the person - mask noise stays roughly
   constant in pixels - so measured GT displacement underestimates the gate needed at small scale.
   A scale-relative gate needs a jitter floor: `max_dist = max(k * person_height, floor)`.
2. **This test is under-powered on the scoring question, so step 2 is not dead, it is unmotivated
   by *this* evidence.** KTH + branch occlusion contains no competing non-person mover, which is
   exactly the failure mode the height term (and any learned replacement) exists for - the
   documented "swaying foliage outscored the person" case. To get a real go/no-go on step 2, the
   synthetic data needs an independent moving distractor of non-person size/shape, not just an
   occluder. That is a data-generation change, and it should come before any training.

Notes on fidelity, for the record: `_Track`'s previously-inline `50.0`/`2.0`/`9.0` covariance
constants are now `_Track.P_VAR`/`Q_VAR`/`R_VAR` class attributes with those same values (no
behavior change - the 1x bucket reproduces byte-identically across both arms, asserted in the
script) purely so the test could rescale them. `score_and_fit`/`track_blobs` logic is untouched.
Scale is simulated by resizing the whole frame, so the person's *fraction* of the frame is
constant; a real camera-distance change would keep frame size fixed and shrink the person within
it, which needs person/background compositing this test does not do. Occluder density 0.35 is
still uncalibrated against real NFO coverage statistics (pre-existing documented gap).
