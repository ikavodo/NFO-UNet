# Generating pseudo-segmentation masks for NFO — current approach, and a request for a second opinion

**Status:** working, iteratively patched, functional but not yet validated at scale.
**Ask:** is this the right *strategy*, or are we optimizing the wrong layer of the problem?
The specific alternative under consideration is using a stronger model (e.g. SAM3) with a
**text prompt** ("person walking/running/jogging behind foliage") instead of, or in addition to,
the bounding-box-driven approach below. This report is written for a fresh, stronger model to
critique the approach - not to justify it.

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
74-155 frames (median ~110) - i.e. long, continuous visibility spans separated by a handful of
full-occlusion gaps, not constant flickering.

### 2. Find geometric "clear corridors" per sequence

`gen_data/nfo_visibility.py` - since the camera and occluders are static, build a clean
background image (median of the many no-person frames every sequence has - NFO has hundreds
available), then find x-ranges where the background is unoccluded across the walking-path
height band (derived from the GT box y-center range). Several thresholding strategies were
tried (column-mean, column-min, fraction-occluded) and each worked for some sequences and
failed for others (seq1/seq3 have discrete bare-branch trunks with real gaps; seq2/seq4 have
denser, more continuous foliage with no clean discrete gap structure). **No single automated
formula worked across all 4 sequences** - settled on a manually curated per-sequence region
list, each picked by eyeballing whichever automated attempt looked best, plus one
manually-identified corridor a human found that every automated width threshold missed:

```python
CURATED_CLEAR_REGIONS = {
    'seq1': [(3, 30), (44, 91), (106, 141), (152, 223)],
    'seq2': [(2, 38), (56, 117), (153, 174)],
    'seq3': [(2, 30), (110, 166)],
    'seq4': [(18, 49), (151, 181)],
}
```
(raw pixel x-ranges, 224px-wide frames)

This is already a flag worth raising to a reviewer: **the region-finding step required
human-in-the-loop tuning per sequence**, not a generalizable formula. That's a strong signal
this whole approach may not scale to a 5th sequence, let alone a different dataset.

### 3. Cross-reference GT trajectory against clear corridors → checkpoints

`geometric_checkpoints()`: for each segment, find frames where the GT box is *fully* contained
in one of the clear corridors (not just centroid-in-region - full-body containment). Group
consecutive such frames into "visibility bursts," take the middle frame of each burst as a
checkpoint. A segment ends up with 0 (fallback to naive middle-frame), 1, or several
checkpoints - typically 2-5 with the current curated regions.

### 4. Multi-checkpoint SAM2 video propagation, neighbor-bounded

`gen_data/gen_nfo_pseudo_masks.py` - each checkpoint is seeded with **both** a point prompt
(GT box center) and a box prompt (the full GT box) into `SAM2VideoPredictor`, then propagated
in both directions via `propagate_in_video`. Bounding rule: checkpoint *i* propagates outward
unbounded on the side with no neighboring checkpoint, and only as far as the neighbor on the
side that has one. This means every frame between two consecutive checkpoints gets **double
coverage** (two independent propagation runs), while frames beyond the outermost checkpoints get
single coverage. A direction is early-stopped after 5 consecutive empty-mask frames (a dead
run doesn't self-recover - confirmed empirically, see "Failure modes" below).

Rationale for box+point (not point-only): a point-only prompt has to grow into low-contrast
regions via appearance similarity; this failed specifically on **feet/pants that are the same
color as foliage** - the model had no reason to extend the mask down into a texturally
indistinguishable region. The box prompt tells it explicitly where the object's known extent is.
**This fix is unverified on real data as of this report** - implemented and unit-tested (pixel
math only), not yet run on GPU.

### 5. Combine multiple checkpoints' masks per frame

Two strategies exist, selectable via `--combine-method`:

- **`majority`** (default): per-pixel majority vote across whichever checkpoints reached a given
  frame. At n=2 (the common case), this is intersection-like - high precision, but a lot of real
  person-pixels get dropped whenever the two masks don't perfectly agree (observed: very sparse
  masks, sometimes just a few dozen pixels of a person that should occupy hundreds).
- **`union_gt_outlier`** (current working direction): union of all available masks, then reject
  outlier connected components using the **GT box as an anchor** (not a statistic derived from
  the masks themselves, which is unreliable at n=2) - a component survives if its centroid is
  within `1.25 × GT-box-width` of the GT box center, **and** its bounding-box width is ≥4px.

## Failure modes found, and how each was patched (this is the part worth scrutinizing)

This is the crux of the concern driving this report: **every fix so far has been reactive,
discovered by visually inspecting individual segments and patching the specific artifact seen.**

1. **Total mask collapse over long unaided propagation.** A single prompt propagated across a
   full ~150-frame segment: 41/155 frames came back completely empty, all in one contiguous
   block, never self-recovering. → led to the whole multi-checkpoint architecture (step 4).
2. **Confidently-wrong blob locked onto background**, not the person, surviving because a lone
   propagation has no independent check. → led to using multiple checkpoints for cross-agreement
   in the first place.
3. **Large unlabeled tails** whenever the GT trajectory spent an extended stretch outside the
   (initially too-narrow) clear corridors. → widened the curated region set for seq1 specifically
   (more human-in-the-loop tuning).
4. **Persistent thin vertical sliver** (a trunk edge one checkpoint's mask kept including, 1-3px
   wide even where tall) inflating disagreement metrics without reflecting real quality loss. →
   added a bounding-box-width filter (reject components <4px wide) to `union_gt_outlier`.
5. **SAM2's memory-based tracking getting "stuck"** on a frozen spatial location after the person
   walks past it, retained for several frames after the tracker has actually lost the real
   target. → tightened the GT-distance outlier filter (2.0× → 1.25×), and re-derived the
   distance metric itself: it was scaling tolerance by `max(box_width, box_height)`, which for a
   standing person (height ≫ width) made the *effective* tolerance in pixels much larger than
   the multiplier suggested, since drift is a horizontal phenomenon. Switched to width-only
   scaling.
6. **Feet/pants same color as foliage** - point-only prompting had no appearance signal to grow
   into that region. → switched to point+box prompting (step 4). Unverified on real data yet.

**Every one of these was found by rendering a contact-sheet overlay and eyeballing it**, not by
a metric flagging it automatically (the `min_pairwise_iou` diagnostic catches *some* of these -
notably #2's total-disagreement signature - but not #4, #5, or #6, which required visual
inspection to even notice).

## The core question for review

Is this the right level to be solving this problem at? The pattern above - discover an artifact
visually, add a targeted filter/threshold/prompt-format tweak, repeat - has a real chance of
being a whack-a-mole process that never converges, especially given:

- Region-finding already required manual per-sequence curation (step 2) - the foundation isn't
  even fully automated.
- Several of the "fixes" are threshold tunings (1.25× vs 2.0×, 4px vs some other width) chosen
  by eyeballing one or two segments, not validated against a broader sample.
- The underlying failure modes (drift, stuck-tracking, low-contrast-region growth) are known,
  general weaknesses of memory-based video segmentation trackers - not NFO-specific quirks. It's
  plausible a fundamentally different tool sidesteps the whole class of problems rather than
  requiring per-symptom patches.

**The specific alternative on the table:** SAM3 (if available/appropriate) with a **text
prompt** describing the scene ("person walking/running/jogging behind foliage"), used instead of
or alongside the GT bounding box. The appeal: if SAM3 supports strong per-frame
language-grounded segmentation, re-grounding from a semantic description at every frame (rather
than propagating memory forward from a seed frame) would have no drift/stuck-tracking failure
mode at all, since there's no accumulating state to drift - each frame's segmentation would be
independent. This trades the current problem class (propagation artifacts) for a different one
(per-frame detection reliability/consistency, whether text-grounding is precise enough to avoid
grabbing nearby vegetation that's also plausibly "near a person," computational cost of running
heavy per-frame inference on potentially thousands of frames instead of a handful of
checkpoints).

Honest caveat: this report's author has read SAM2's actual source in depth this session (video
predictor API, `propagate_in_video`, point/box prompting) but has **no verified hands-on
knowledge of SAM3's actual capabilities, API, or whether/how it supports text-grounded video
segmentation** - that part of this report is the user's hypothesis, not a researched claim, and
is exactly what needs an informed second opinion.

## What's NOT yet done / open

- `union_gt_outlier` is marked TEMPORARY in code, not yet the default - still being compared
  against `majority` on real segments.
- Box+point prompting (item 6's fix) is implemented but **not yet run on real GPU data**.
- No held-out quality metric exists beyond the `min_pairwise_iou`/coverage diagnostics computed
  from the pipeline's own intermediate outputs - there is no independent ground truth to check
  final mask quality against (by definition - that's the problem being solved). All quality
  assessment so far has been visual/manual, on a handful of specifically-flagged segments, not a
  representative sample.
- The pipeline has never been run to completion + validated across all 4 sequences with the
  current (post-patches) code version.
