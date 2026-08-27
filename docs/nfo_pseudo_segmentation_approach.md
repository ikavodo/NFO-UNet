# Generating pseudo-segmentation masks for NFO — current approach, and a request for a second opinion

**Status:** working, iteratively patched on visually-inspected real segments, about to be run to
completion across all 4 sequences at native resolution.

**Goal, precisely stated:** the masks don't need to be pixel-perfect - they need to be close
enough to serve as usable pseudo-ground-truth (weak training signal / a "good enough" stand-in
for real annotation). Keep that bar in mind when judging the approach below.

**Ask: a pipeline-complexity judgment call.** The approach below has grown by iterative
patching - each new filter/threshold/prompt-format change fixed a specific artifact found by
visually inspecting a segment. Given the "good enough, not perfect" bar above, is this the right
amount of machinery, or has it drifted past the point of diminishing returns for what's actually
needed? The tension: more filters/checkpoints/tuning can push mask quality up, but each one also
makes the pipeline harder to reason about and re-tune when it inevitably meets a new sequence or
failure mode - is the current balance defensible, or should some of it be cut in favor of a
simpler, more interpretable pipeline even at some cost to raw quality?

(A related resolution question - native vs. 224 - came up during development and was checked
against real ground truth via a KTH-based experiment; see "Native vs. 224 resolution" below for
the result. That one's considered settled, included here for context rather than as an open ask.)

## Why this exists

NFO (the real-world outdoor "fragmented occlusion" dataset this project evaluates on) has
per-frame bounding-box ground truth (used for training/eval throughout this project) but **no
segmentation masks**. A separate exploration (killed as an architecture-paper direction, see
`~/.claude/projects/.../memory/research-evaluations/2026-08-26-learned-stn-registration-nfo-unet.md`)
wanted segmentation masks as weak training signal for a learned motion-alignment + segmentation
network. That specific architecture idea was killed (residual localization error turned out to
be scene-independent, not occlusion-driven, undercutting the motivation), but generating decent
pseudo-masks remains a useful, semi-independent asset - and is the actual open engineering
problem this report is about.

**Key constraint:** the camera is static per sequence. Occluder geometry (tree trunks, bushes)
is fixed for the whole sequence; only the person moves.

## The pipeline, as it currently exists

### 1. Segment continuously-visible spans

`gen_data/nfo_segment_utils.py:find_segments` - a person's GT box has a sentinel row (`x=-1`)
in `groundtruth.txt` on frames where they're not visible at all. Contiguous runs with no
sentinel are "segments." Measured: NFO's 4 sequences have exactly 8 segments each, lengths
74-155 frames (median ~110).

### 2. Find geometric "clear corridors" per sequence

`gen_data/nfo_visibility.py` - since the camera and occluders are static, build a clean
background image (median of the many no-person frames every sequence has), then find x-ranges
where the background is unoccluded across the walking-path height band (derived from the GT box
y-center range). No single automated thresholding formula (column-mean, column-min,
fraction-occluded were all tried) worked across all 4 sequences - seq1/seq3 have discrete
bare-branch trunks with real gaps, seq2/seq4 have denser, more continuous foliage. Settled on a
manually curated per-sequence region list:

```python
CURATED_CLEAR_REGIONS = {
    'seq1': [(44, 85), (154, 203)],
    'seq2': [(2, 38), (56, 117), (153, 174)],
    'seq3': [(2, 30), (110, 166)],
    'seq4': [(18, 49), (151, 181)],
}
```
(raw pixel x-ranges, 224px-wide frames - clear-region finding and checkpoint selection stay in
224-space even though propagation itself now runs at native resolution, see below; this step
only decides *which frame* to seed from normalized GT position, not any pixel coordinates that
get passed to SAM2). seq1 was briefly widened to 4 regions to reduce unlabeled tail coverage,
then reverted back to 2 - more checkpoints means more independent propagation runs, each an
independent chance for one to get stuck near an occluder, which the coverage gain didn't offset.

### 3. Cross-reference GT trajectory against clear corridors → checkpoints

`geometric_checkpoints()`: for each segment, find frames where the GT box is *fully* contained
in one of the clear corridors (not just centroid-in-region). Group consecutive such frames into
"visibility bursts," take the middle frame of each burst as a checkpoint. A segment ends up with
0 (fallback to naive middle-frame), 1, or several checkpoints - typically 2-3 with the current
curated regions.

### 4. Multi-checkpoint SAM2 video propagation, neighbor-bounded, native resolution

`gen_data/gen_nfo_pseudo_masks.py` - each checkpoint is seeded with **both** a point prompt (GT
box center) and a box prompt (the full GT box) into `SAM2VideoPredictor`, then propagated in both
directions via `propagate_in_video`, at native (800x600) resolution - frames are staged directly
from the native source, not the project's usual 224x224 training resolution (see "Native vs. 224
resolution" below for why). Bounding rule: checkpoint *i* propagates outward unbounded on the
side with no neighboring checkpoint, and only as far as the neighbor on the side that has one -
every frame between two consecutive checkpoints gets double coverage (two independent
propagation runs), frames beyond the outermost checkpoints get single coverage. A direction is
early-stopped after 5 consecutive empty-mask frames (confirmed empirically: a dead run doesn't
self-recover) or 5 consecutive near-static frames (a mask whose centroid barely moves is treated
as a stuck tracker, since the person is continuously in motion throughout this dataset by
construction).

Box+point (not point-only) prompting exists because a point-only prompt has to grow into
low-contrast regions via appearance similarity alone, which failed specifically on feet/pants the
same color as the foliage background - the box prompt tells SAM2 explicitly where the object's
known extent is.

### 5. Combine multiple checkpoints' masks per frame

Production default (`union_gt_outlier`): union of all available masks per frame, then clip to the
GT box (dilated a few px) and reject connected components narrower than a few px (real fragments
are always wider than a stray trunk-edge sliver). A simpler majority-vote alternative
(`majority`) also exists in code for comparison - intersection-like at the common n=2 case, much
higher precision but noticeably sparser coverage (real person-pixels dropped whenever two masks
don't perfectly agree).

## Failure modes found, and how each was patched

Discovered by visually inspecting individual segments (contact-sheet overlays, then full-segment
videos) and patching the specific artifact seen:

1. **Total mask collapse over long unaided propagation** - a single prompt propagated across a
   full ~150-frame segment came back completely empty for 41/155 frames, one contiguous block,
   never self-recovering. → the whole multi-checkpoint architecture (step 4).
2. **Confidently-wrong blob locked onto background**, surviving because a lone propagation has no
   independent check. → multiple checkpoints for cross-agreement.
3. **Persistent thin vertical sliver** (a trunk edge one checkpoint kept including) inflating
   disagreement metrics without reflecting real quality loss. → bounding-box-width filter.
4. **SAM2's memory-based tracking getting "stuck"** on a frozen spatial location after the person
   walked past it. → clip-to-GT-box (directly removes anything outside the box, replacing an
   earlier, weaker centroid-distance-based filter), plus a separate temporal-staticness check
   (mask centroid barely moving for several consecutive frames, independent of position relative
   to GT) since a stuck mask near the true trajectory could otherwise survive the distance check
   indefinitely.
5. **Feet/pants same color as foliage** - point-only prompting had no appearance signal to grow
   into that region. → point+box prompting (step 4).

Verified on the two previously-worst segments (seq1 segments 3 and 7, originally 100%/86%
low-IoU checkpoint disagreement): both dropped to 34%/35% after the fixes above, in the same
range as segments never flagged as problematic - confirmed both numerically (diagnostics CSV) and
visually (full-segment video render).

## Native vs. 224 resolution

The pipeline originally ran at the project's standard 224x224 training resolution throughout.
`gen_data/compare_resolution.py` reran the same checkpoint/propagation/combination logic
unchanged on one segment (seq1 seg3) at native (800x600) resolution instead, downsampling only
the final masks to 224 for comparison. Two conflicting signals came out of that:

- **Box-recovery IoU** (bbox of the final mask vs. GT box - a free, no-extra-annotation proxy)
  came back *worse* at native resolution: 0.434 (native) vs 0.545 (224).
- **Direct visual comparison** (overlaying both mask sets frame-by-frame) showed the opposite -
  native-resolution masks looked visibly cleaner, with less spurious background inclusion.

The visual read was trusted over the metric (box-recovery IoU only checks bounding-rectangle
alignment, not actual mask boundary quality, so it's a coarse proxy at best) and the pipeline now
runs natively by default. This also required rescaling several pixel-space constants
(`STATIC_THRESHOLD_PX`, `box_dilate_px`, `min_width_px`) that had been calibrated at 224-space -
now scaled by `img_w / 224` at the point of use.

That single-segment, no-real-GT comparison (the IoU-vs-visual disagreement is itself a sign the
evidence there was thin) motivated a second check with actual ground truth:
`gen_data/kth_occlusion_resolution_experiment.py` takes a short KTH segment that already has a
real per-frame mask (`gen_sam_masks.py`'s `_sammask.png`, computed by single-image SAM2 on the
*unoccluded* frame - reliable there, since KTH scenes are clean, so treated as ground truth),
synthetically composites in an occluder, then runs the same propagation+combination core on the
occluded sequence at two resolutions and scores each directly against that real mask (rather than
a bounding-box proxy). Note KTH's own native resolution (160x120) is *lower* than 224, the
opposite relationship to NFO's native 800x600 - so this tests 224 vs. a further-downsampled 112,
not "native vs. 224" in the NFO sense; the underlying question (does a coarser pixel grid hurt
SAM2 mask quality) is the same either way.

Result on one 24-frame segment: 224 mean IoU 0.771 vs. 112 mean IoU 0.759 - higher resolution
won against real ground truth, consistent with the NFO visual read and inconsistent only with
NFO's box-recovery IoU proxy. Native resolution is being kept as the pipeline default on the
strength of this. Caveat: still a single short segment, one occluder pattern, one resolution
pair - not a systematic sweep.

## Every tunable constant in the pipeline, and how each was actually arrived at

None of these were derived from a systematic measurement or sweep over the dataset. For
comparison, `MAX_DIST=25px` in `tracking/eval_nfo.py` (the classical tracker baseline elsewhere
in this project) *was* derived that way - measured directly from real GT centroid displacement.
Nothing below got that treatment.

| constant | value | basis |
|---|---|---|
| `MAX_CONSECUTIVE_EMPTY` | 5 frames | reasoned default, not measured |
| `STATIC_THRESHOLD_PX` | 0.5px (224-space), scaled by `img_w/224` at native resolution | tightened from an initial 2.0px after measuring real GT motion (1.4-1.7px/frame in 224-space) and finding 2.0px sat above it, triggering on legitimately-moving frames |
| `MAX_CONSECUTIVE_STATIC` | 5 frames | arbitrary, chosen by analogy with `MAX_CONSECUTIVE_EMPTY` |
| `MERGE_GAP` | 10px | tuned by eye across several rendered images |
| `BG_SAMPLES` | 40 frames | arbitrary "enough for a stable median," no convergence check |
| `MIN_WIDTH_FRAC` | 0.85x median box width | derived from recovering one specific missing region in seq2, not validated elsewhere |
| `CURATED_CLEAR_REGIONS` | per-sequence, hand-picked | fully manual, eyeballed per sequence |
| `box_dilate_px` | 3px (224-space), scaled by `img_w/224` at native resolution | reasoned small margin |
| `min_width_px` | 4px (224-space), scaled by `img_w/224` at native resolution | reasoned + verified only on synthetic data, not a real-artifact-width distribution. **Known bias, flagged by review:** median GT box width is 22-25px (224-space) / 77-90px (native), so 4px→14px-native rejects anything narrower than ~17% of body width - above the 1-3px sliver it was built to catch, meaning it likely also deletes genuinely narrow *visible* fragments during occlusion, which is exactly the regime this dataset is named for. Not changed yet (would force a re-run of the in-flight full pipeline pass) - see "What's NOT yet done" below. |

## What's NOT yet done / open

- The pipeline has not yet been run to completion across all 4 sequences with the current
  (native-resolution, post-patches) code version.
- No held-out quality metric exists for NFO itself beyond the diagnostics computed from the
  pipeline's own intermediate outputs (checkpoint agreement) - there's no independent ground
  truth to check final NFO mask quality against (unlike the KTH experiment above, which is a
  proxy scene, not NFO itself).
- None of the constants above were systematically measured or swept (see table above).
- `min_width_px` likely over-rejects narrow-but-real occlusion fragments (see table) - worth
  lowering and re-running once there's time, not blocking today's run.
- Independent review verdict on the pipeline-complexity question: keep the current filters as-is
  (each traces to a distinct observed failure, no redundant pair) - the actual interpretability
  risk is `CURATED_CLEAR_REGIONS` (the one fully-manual, non-generalizing piece, which filter
  changes don't touch) and the `--combine-method` default previously disagreeing with the code's
  own docstring and this doc about which method is production (now fixed: `union_gt_outlier` is
  the default). Later, not now: log what `min_width_px` and the staticness stop actually reject
  per segment - staticness is the one filter partly subsumed by clip-to-GT-box and worth
  revisiting once that's visible.
