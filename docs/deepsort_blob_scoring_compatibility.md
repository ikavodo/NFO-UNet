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

## Step 1b: sway added (a real moving distractor), and two scale-invariance candidates measured

`--sway PX` in `tracking/eval/kill_test_scale.py` shears the branch mask about its base by
`+-PX*sin(2*pi*t/40)`, PX scaling with the bucket, so branch tips swing while the base stays put
(`utils/occlusion_utils.sway_masks`). Sway was re-added not for realism but because it is the
only way to put a **moving non-person object** in front of the scorer - the exact case
`score_and_fit`'s height term exists for. A fully static occluder is absorbed into the MOG2
background model and generates no competing detections at all, which is why step 1 could not
evaluate the scoring question.

Confirmed as a side effect: switching the occluder's appearance from background-valued to
dark-foliage-valued (`OCC_DARKEN=0.45`, keeping the background's own texture) changed the static
results by *nothing* - 0.5x/1x/2x identical to the step-1 run - because a frame-fixed occluder is
background to MOG2 whatever colour it is. Only motion makes an occluder visible to this pipeline.

Sway = 8px at 1x (~7% of person height), period 40 frames against a 13-frame window, so within
one window the sway reads as consistent drift - a plausible competing track rather than obvious
jitter.

**Caveat on this design, which matters for reading the numbers:** sway does two things at once,
and they push in opposite directions. It adds distractor tracks (intended), but it also *weakens
the occlusion*, because dynamic occluder pixels raise MOG2's per-pixel variance, so the person
leaks through where a static occluder would have erased them. Net effect measured at
per-scale-correct constants: 1x went 61.5% -> 74.0% and 2x went 74.4% -> 91.4%, i.e. the swaying
version is *easier* overall despite the added distractors. So absolute levels are not comparable
between the static and sway runs; only within-run contrasts are. The clean design for the next
iteration is to separate the two roles: keep a static occluder on the person's path for the
occlusion, and add a *separate* swaying object away from that path for the distractor.

### The height term (what step 2 would replace) becomes ~4x more load-bearing under sway

Per-scale-correct constants everywhere, varying only `score_and_fit`'s shape term:

| bucket | correct height | no height term | height mis-scaled, rest correct |
|---|---|---|---|
| 2.0x, static | 74.4% | 72.3% (-2.1pp) | 72.1% (-2.3pp) |
| 2.0x, **sway** | 91.4% | 88.6% (-2.8pp) | **82.2% (-9.2pp)** |
| 1.0x, sway | 74.0% | 72.3% | 74.0% |
| 0.5x, sway | 38.0% | 37.4% | 38.0% |

Two things follow. (1) With a moving distractor present, mis-scaling the height term costs 9.2pp
instead of 2.3pp - the scoring term *is* real and *is* scale-sensitive, but only once a
non-person mover exists, which is exactly why step 1's static test measured 0% for it. (2) A
**mis-scaled shape prior is worse than no shape prior at all** (82.2% vs 88.6%): a wrong
`expected_height` actively down-weights the true person. That argues first for making the term
scale-relative, and only secondarily for making it learned.

Keep the magnitude in view: 9.2pp is still an order of magnitude below the 85.6pp the association
gate costs at the same bucket. Step 1's ordering stands - fix the gate first.

### Two candidates for scale invariance, measured (hit@0.1, sway run)

| bucket | (a) per-scale-correct | (b) fixed 1x | `scale_rel` | `canon` |
|---|---|---|---|---|
| 0.5x | 38.0% | 80.3% | **75.0%** | 51.1% |
| 1.0x | 74.0% | 74.0% | **93.2%** | 60.0% |
| 2.0x | 91.4% | 5.9% | **89.4%** | 65.7% |
| spread across buckets | 53.5pp | 74.4pp | **18.2pp** | 14.5pp |
| worst bucket | 38.0% | 5.9% | **75.0%** | 51.1% |

`scale_rel` derives *every* pixel constant - `max_dist`, `expected_height`, `merge_radius`,
`min_area`, both morphology kernels, and `P`/`Q`/`R` - from one measured number `h_ref`, with
coefficients identical at every bucket. It beats the per-scale-correct GT-measured recipe at two
of three buckets and lifts the worst bucket from 5.9% to 75.0%, with no-track at ~0% everywhere.
Static run: 74.8% / 96.5% / 98.8%. **This is the fix, and it involves no learning.**

`canon` - resize the input until the measured `h_ref` equals a canonical value, then run the
existing pipeline with the existing fixed constants - is also consistent but clearly worse in
level (51/60/66%): canonicalizing the 2x bucket downward discards real resolution the tracker
was using. Rescaling the constants beats rescaling the image.

### The load-bearing detail: the scale proxy must itself be scale-equivariant

The first `h_ref` was p95 of raw foreground-component heights. Measured, it goes **42 -> 56 -> 76
px** while the true person height goes 64 -> 128 -> 256 - only ~1.35x per 2x of real scale. A
fixed-pixel front-end fragments a large person into *relatively* smaller pieces than a small one,
so fragment statistics are not equivariant, and no amount of coefficient fitting repairs a
non-equivariant proxy. That version of `scale_rel` scored a flat, uniformly-bad ~30% at every
bucket: consistent and wrong, which is the trap to watch for - low spread alone does not mean
scale-invariant.

Fix (`estimate_h_ref`): bridge the fragments before measuring, with a dilation radius derived
from the current height estimate, and iterate `h -> radius(0.25h) -> h` to a fixed point. The only
pixel quantity in the loop is itself a function of `h`, so the estimator carries no absolute
scale of its own. Measured: **60 -> 122 -> 253 px**, within ~5% of the true GT person height at
every bucket - recovered from occluded footage with no ground truth at all.

### Coefficients, and where the residual scale-dependence lives

`--alpha-sweep` (3 sequences, sway; `max_dist` and `merge_radius` as multiples of `h_ref`):

| md alpha | merge alpha | 0.5x | 1.0x | 2.0x | worst |
|---|---|---|---|---|---|
| 0.15 | 0.5 | 55.7% | 86.7% | 98.5% | 55.7% |
| 0.25 | **0.75** | 72.3% | 93.3% | 90.0% | **72.3%** |
| 0.25 | 1.0 | 73.4% | 88.7% | 82.1% | 73.4% |
| 0.40 | 1.5 | 74.0% | 86.2% | 78.1% | 74.0% |

`max_dist` is remarkably insensitive (0.15 -> 0.40 moves the result ~2pp at fixed merge): the gate
mainly has to be *big enough*. The measured GT-displacement ratio (~0.095*`h_ref` here, and
25/195 = 0.128 on NFO - reassuringly close across two datasets) is a lower bound, not the right
value, because blob centroids jump between fragments. `merge_radius` is the sensitive
coefficient, and it is where the residual scale-dependence lives: 0.5 is best at 2x and worst at
0.5x, 1.0 the reverse. Recommended single pair: **md=0.25, merge=0.75*`h_ref`** (`eval_nfo.py`
currently uses height/2, i.e. ~0.5 - slightly too tight).

### How to proceed - concrete order

1. **Promote `estimate_h_ref` out of the test file** into `tracking/core/` and call it once per
   sequence from `eval_nfo.py`/`track_sequence.py`. Replace `MAX_DIST`/`EXPECTED_HEIGHT`/
   `MERGE_RADIUS` and the `min_area`/kernel/`P`/`Q`/`R` defaults with `ALPHA_* * h_ref`. That is
   the whole scale-invariance fix, and the coefficients above are already calibrated. The real
   test on NFO: check `h_ref` lands near NFO's measured 195px, then confirm the existing NFO
   numbers reproduce with *no* NFO-specific constant anywhere.
2. **Re-measure the residual spread afterwards.** What remains is concentrated in the merge stage
   and in mask sparsity at small scale - both segmentation problems, not scoring problems. Fix or
   bound those before considering anything learned.
3. **Only then reconsider a learned scorer**, with the measured headroom in mind: the shape term
   is worth ~3pp when correct, ~9pp when the alternative is a mis-scaled one, against 85pp for
   the gate.

### If the score is to be learned, how to make it generalizable rather than another constant

The measured facts constrain this fairly tightly:

- **Learn a ranking, not a score.** `score_and_fit`'s output is only ever consumed through an
  `argmax` over the tracks in one window. Absolute calibration is irrelevant, and it is exactly
  what would fail to transfer. Train a pairwise ranking loss on (true-person track, distractor
  track) pairs drawn from the same window.
- **Every feature must be dimensionless.** This is the actual generalization requirement, and the
  doc's existing feature list has the right instinct: aspect ratio, frame-to-frame size *growth
  rate*, size over the track's own running median, `net_disp / (span * h_ref)`, `resid_std /
  h_ref`, and oscillation energy (net displacement over path length - a swaying branch has large
  path length with near-zero net displacement, and this test now generates exactly that). Any
  feature in raw pixels reintroduces the problem just measured. `_Track.history` already carries
  `(x, y, height)` per frame, so no new plumbing is needed.
- **Keep it tiny.** 5-8 dimensionless features with logistic regression or a depth-2 tree. At
  ~850 windows per configuration, anything larger fits the occluder generator rather than the
  problem. A CNN embedding (step 3 of the build order) is not justified by a 9pp ceiling.
- **The training data must contain the failure mode, or the scorer is untestable.** Step 1 could
  not measure the scoring term at all because a static occluder produces no competing tracks.
  `--sway` is the minimum; better is the separated design noted above - static occluder for
  occlusion, independent swaying object for the distractor - plus distractors of clearly
  non-person aspect ratio and non-person (oscillatory, zero-net-displacement) motion, which are
  what the features above can actually discriminate.
- **Validate with the per-bucket, leave-one-in protocol used here, not pooled accuracy.** Report
  every bucket and report the spread. A learned scorer that improves the mean while widening the
  spread across scales has not solved this problem; it has re-tuned to one scale.
- **Honest expectation.** After item 1 above, a learned scorer competes for single-digit pp
  against a two-coefficient dimensionless formula, while carrying the synthetic->real domain-gap
  risk every learned component in this project has carried. The defensible framing is therefore
  not "learned scorer beats heuristic" but "the heuristic's hand-picked terms can be replaced by
  a dimensionless, calibration-free scorer with no per-dataset constants" - the contribution
  being the removal of the constants, not the accuracy.

Remaining limitations, unchanged or new: occluder density 0.35 is still uncalibrated against real
NFO coverage; scale is still simulated by whole-frame resize rather than person/background
compositing; sway amplitude and period were chosen to be plausible, not measured from NFO
footage; and sway currently confounds "more distractors" with "weaker occlusion" (see the caveat
above).

## Step 1c: separated occluder/distractor, real-NFO wiring, and a reinterpretation of what the shape term does

### The scale-relative parameterization now lives in the pipeline

`estimate_person_height` -> `tracking/core/preprocess.py`; coefficients and
`scale_relative_params` -> `tracking/core/track_sequence.py`, which now accepts a
`person_height` kwarg and derives `max_dist`, `merge_radius`, `expected_height`, `min_area`,
both morphology kernels and the Kalman variances from it. `eval_nfo.py` has a third config
using it, plus a CLI selector so each config can be one job.

### Result on real NFO: nearly free, but the estimator does not transfer

3495 windows, all four sequences:

| config | mean resid | median | p90 | no-track | NFO-specific constants |
|---|---|---|---|---|---|
| no shape term | 0.1912 | 0.0468 | 0.5929 | 0.2% | 8 |
| hand-tuned (baseline) | 0.0698 | 0.0250 | 0.1049 | 0.2% | 8 |
| scale-relative (measured) | 0.0762 | 0.0316 | 0.1381 | 0.0% | **0** |

Dropping all eight hand-tuned constants costs +0.006 mean / +0.007 median. But the
validation criterion set out in step 1b **failed**: measured person height came out at
315 / 61 / 129 / 200 px on seq1-4 against a true ~195px, where on synthetic KTH the same
estimator was within 5% at every scale. NFO's occlusion is far denser than the synthetic
0.35 coverage, so the fragment-bridging step either welds the person to surrounding foliage
(seq1: 315px) or finds too little connected foreground to measure (seq2: 61px).

Accuracy survived a 3x error in the estimate for two reasons, and only one is a real
robustness claim: the association gate is genuinely flat over a wide band (already known
from the coefficient sweep), *and* seq2's too-small estimate produced a smaller merge radius
and made that sequence **better** (0.1095 -> 0.0783). That second one is luck cancelling a
mis-set constant, and it points at a separate finding: **`MERGE_RADIUS = 100` looks too large
for some NFO sequences** - an over-large merge sweeps neighbouring foliage blobs into the
person's box and drags the reported centre off. Worth testing on its own.

So: the parameterization is sound and the estimator is not, on real data. Do not report
`estimate_person_height` as a measurement of person height; it is a scale *proxy* that
happens to be adequate for parameter derivation in this regime.

### Separated occluder/distractor: implemented, and it does not reproduce the failure mode

`--distractor PX` keeps the occluder over the person's path perfectly static (identical
occlusion to the static run: coverage 0.365 in both) and puts an independent swaying branch
cluster of full person height (mean 156px vs ~120px person) in a canvas strip added beside
the frame, with guaranteed zero overlap. Residuals stay normalized by the original frame
size, so the runs are directly comparable to the static run. The strip is needed because
KTH's person traverses the whole frame: the only person-free region *inside* the frame is
the band above their path, ~30% of a person height, too small to distract.

Height term, per-scale-correct constants, at two distractor strengths:

| bucket | correct height | no height term | height mis-scaled |
|---|---|---|---|
| 2.0x, distractor 8px | 72.6% | 69.8% | 68.9% |
| 2.0x, distractor 40px | 72.7% | 69.9% | 70.2% |
| 1.0x, distractor 40px | 61.2% | 59.4% | 61.2% |

**Raising the distractor's sway amplitude 5x - to roughly the person's own per-window
displacement - changed nothing.** The synthetic distractor is not competing at all. Reason:
a rigid shear of a dense branch band moves the branch *tips* by the full amplitude, but the
*centroid* of the resulting large connected blob barely moves, and `score_and_fit` scores
centroid trajectories. Real foliage does something different - it occludes and reveals
background patches, so blobs appear and vanish and their apparent centroid jumps - which is
noise-like rather than translation-like.

Two separate attempts (swaying occluder in step 1b, separated swaying band here at 5x
amplitude) have now failed to synthesize a candidate that outscores the person on motion.
Producing that failure mode synthetically is harder than assumed, and the next attempt
should not be another swaying-geometry variant.

### Reinterpretation: on real data the shape term is doing fragment selection, not distractor rejection

The doc has assumed throughout that `expected_height` exists to reject swaying foliage (its
own docstring says so). The NFO numbers above say something different. Turning the term off:

- no-track rate is **unchanged** (0.2% both ways) - the tracker still finds a track, it does
  not get captured by foliage and lose the person entirely;
- but mean residual worsens 2.7x (0.0698 -> 0.1912) - i.e. the position it reports gets much
  worse while still being a track.

That is the signature of picking the wrong *fragment of the person*, not of locking onto a
different object. NFO's people are heavily fragmented; the height term biases selection
toward the full-body-sized blob, which makes the merged centroid right. On KTH, where
fragmentation is much milder, the same term is worth only 2-3pp - which now reads as a
consistency, not a contradiction.

Consequences for the build order:

1. The most promising learned target is **fragment grouping/selection** (which blobs are one
   person, which fragment to anchor on), not track scoring and not distractor rejection.
   It targets the measured failure, it has free labels from synthetic data, and it is also
   what the over-large NFO `MERGE_RADIUS` finding points at.
2. A learned *ranker* over dimensionless, within-window-normalized features (each track's
   height over the median candidate height, displacement over median displacement, net
   displacement over path length, gait periodicity) has a real advantage that this session's
   data now supports directly: it needs **no scale estimate at all**, and the explicit scale
   estimator is exactly the component that just failed to transfer to real footage. Scale
   cancels in a within-window comparison because all candidates share a frame.
3. That still cannot fix the association gate, which is applied before any candidate exists
   (at 2x, 90% of windows produce nothing to rank). The gate needs to be made relative
   too - a ratio test against the runner-up match, or k x the median nearest-neighbour
   distance in the frame, or Mahalanobis gating with online-estimated noise. Those are
   scale-free without measuring anything about people.

### Reproducing

    python -m tracking.eval.eval_nfo [noshape|fixed|relative]      # ~3.5 min per config
    python -m tracking.eval.kill_test_scale [--sway PX] [--distractor PX] [--frozen-preprocess]
    python -m tracking.eval.kill_test_scale --alpha-sweep --sway 8
    sbatch tracking/eval/kill_test_scale.sbatch                    # all six variants in parallel

The kill-test cliff reproduces in every variant run so far: fixed constants at the 2x bucket
land at 2.1-5.9% against 72-91% for per-scale-correct, with 79-95% of windows producing no
track. `scale_rel` lands at 74.6% / 95.8% / 95.0% (weak distractor) and 74.6% / 96.0% / 96.5%
(competitive), i.e. ~21pp spread against 77pp for fixed constants.
