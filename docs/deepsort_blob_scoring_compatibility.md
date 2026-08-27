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
   (**absent**: `blob_tracker.py`'s detections carry only `{x, y, area, bbox}` - no appearance
   descriptor at all. This isn't a small gap: NFO's foreground detections come from
   background-subtraction masks (MOG2, in `preprocess.py`), which are binary silhouettes with no
   color/texture signal to embed - there is no pixel content to run a Re-ID CNN on).

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

## Open questions / not yet done

- No code changes made yet - this is a compatibility/planning doc only.
- ByteTrack's low-confidence-detection handling hasn't been compared against the current
  `min_area` hard cutoff in `detect_blobs` - possibly a more directly relevant fix than
  Mahalanobis gating, given the fragmented-blob domain.
- Scale-relative constant normalization (item 3 above) is unimplemented and unvalidated - this
  doc only confirms the current constants are scale-specific, not fixes it.
