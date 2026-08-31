# Gait-consistency-weighted motion-compensated fusion, feeding a UNet segmentation head

**Goal:** use the classical tracker's own Kalman motion estimate to build a motion-compensated,
gait-consistency-weighted "integrated image" - not a raw stacked-frame input - and have the UNet
segment *that*, instead of asking the network to implicitly learn motion compensation end-to-end.

## Why this, and why now

This project has repeatedly shown the classical Kalman/blob tracker beating the UNet at
localization. A prior direction (`~/.claude/projects/.../memory/research-evaluations/
2026-08-26-learned-stn-registration-nfo-unet.md`, KILL verdict) tried to inject motion into the
UNet via an implicitly-*learned* spatial transformer, trained end-to-end. That was killed for
several reasons, the most fundamental being architectural: an unsupervised photometric/STN loss
on windblown vegetation is just as likely to sharpen the occluder as the person, and the
segmentation head's contribution was unmeasurable (no NFO segmentation ground truth existed at
the time).

**This proposal is different in kind, not just degree**: it doesn't ask the network to learn
motion compensation at all. It hands the network an *already* motion-compensated image, produced
by the classical tracker (`tracking/core/integrate_image.py`) this project has already validated
outperforms the UNet at exactly that job - closer to feature engineering (compute a better input,
classically) than to end-to-end learned registration. And the "unmeasurable" objection is
substantially weaker now than when the STN idea was killed: NFO now has a SAM2 pseudo-mask
pipeline (`gen_data/nfo_pseudo_masks/`), and KTH's real segmentation masks are already the
training/eval target for the in-flight anisotropic-heatmap experiment
(`config/train_config.py`'s `kth_train_anisotropic`, plan at `/home/akovi/.claude/plans/
sparkling-munching-valiant.md`). A UNet trained to segment from this fused input is directly,
fairly measurable against both.

## What already exists (`tracking/core/integrate_image.py`, all confirmed by reading the code)

- `align_frames` (line 46): crops every frame to a fixed size centered on the Kalman-tracked
  winning track's anchor point, shifted by `-vx*dt` so every frame samples the same real-world
  point as the center frame - this **is** the Kalman motion compensation, already implemented
  and already the thing this project's own tracker uses.
- `fuse` (line 61) has three combination methods:
  - `'median'` (default): robust to a minority of occluded frames at a given pixel - the
    occluder's intensity gets outvoted by the true value.
  - `'mean'`: blends occluder and true value together (ghosting), kept only for comparison.
  - `'gaussian'`: weights frames by a Gaussian in **temporal distance** from the center frame.
    Its own docstring already states the exact tension this proposal targets: *"trades
    occlusion-robustness for pose fidelity (limb articulation across a gait cycle changes shape
    frame to frame)."* This is the precise hook point - the weight is currently a function of
    `|t - center_t|` only, with no actual awareness of whether the person's pose/shape at frame
    `t` resembles the center frame's pose at all.
- `integrate` (line 89): the end-to-end pipeline (align, optionally restrict to the person's own
  blob via `restrict_to_nearby`'s connected-component masking, then fuse). Two modes:
  - `mask_background=False` (current default): fuses full-frame crops - the docstring notes this
    is recommended for feeding an off-the-shelf detector (YOLO), since full-frame integration
    naturally blurs the background while keeping the aligned subject sharp.
  - `mask_background=True`: additionally restricts each aligned frame to only the person's own
    connected blob before fusing - i.e. the classical tracker's own coarse segmentation already
    seeds the fusion. **`integrate()` has never been used as input to a segmentation network
    before** - both modes exist for a different downstream consumer (a detector), and using
    either as UNet input is new, unvalidated territory, not a drop-in reuse.

## The proposed change

Replace (or add as a new `fuse` method) the `'gaussian'` weighting's temporal-distance-only
signal with a **gait-consistency-derived** weight: per-frame shape descriptors (eccentricity/
orientation from the anisotropic-heatmap work - `cv2.moments` on the real mask during training-
data construction, or eventually the UNet's own predicted anisotropic channel at inference) tell
you how similar frame `t`'s body configuration is to the center frame's, independent of raw
temporal distance. A frame at `dt=3` with a nearly-identical gait phase to the center frame is a
better fusion candidate than one at `dt=1` mid-stride-transition - something pure temporal-
distance weighting can't express. Concretely: `weights[t] = f(shape_similarity(t, center_t))`
instead of `f(|dt|)`, normalized the same way `'gaussian'` already normalizes its weights (line
84), otherwise reusing the exact same `fuse` structure.

This is analogous to (and could literally reuse) the same Mahalanobis-covariance-as-uncertainty
idea already discussed for the Kalman filter's measurement noise (`docs/
deepsort_blob_scoring_compatibility.md`) - the same anisotropic covariance signal, applied to a
different consumer (fusion weighting instead of Kalman `R`).

Two output flavors are worth comparing, not just one:
1. `mask_background=False` + gait-weighted fusion → full motion-compensated, gait-weighted
   image, UNet does the entire segmentation from that.
2. `mask_background=True` + gait-weighted fusion → the classical tracker's own coarse blob mask
   already seeds/restricts the fused image before the UNet ever sees it - closer to "UNet refines
   an already-coarse classical segmentation" than "UNet segments from scratch." Different risk/
   reward: likely higher baseline quality (less for the network to get wrong), but inherits
   whatever the classical blob's own limitations are (the same fragmentation/MOG2-noise issues
   already documented as reasons "winning blobs as segmentation" alone was insufficient).

## Related literature, calibrated honestly

- Cao et al. (CVPR 2017), **OpenPose** / Part Affinity Fields: jointly trains per-joint
  confidence heatmaps alongside per-limb orientation vector fields, sharing a backbone across
  both. Real, validated precedent for training a shared representation against both a scalar
  heatmap and a richer structural target - but its headline result is about **multi-person
  association quality** (grouping joints into distinct people), not proof that the auxiliary
  field sharpens single-target localization precision on its own. Cited here as precedent for
  the *pattern* (joint training over structured + scalar targets), not as direct evidence for
  the specific localization-sharpening claim made earlier in this project's discussion.
- Newell et al. (2016), **Stacked Hourglass Networks**: iterative multi-stage heatmap refinement,
  each stage consuming the previous stage's heatmap prediction as additional input. Relevant
  architectural pattern if a later iteration wants the UNet to consume its own prior-stage
  output (e.g. a predicted anisotropic heatmap) as part of a refinement loop, rather than a
  single forward pass - not proposed for the current scope, noted for later.

## Measured result: gait-weighted fusion does NOT beat temporal weighting (negative result)

`eval/compare_fusion_weighting.py` implements and tests the proposal above against the existing
`fuse()` methods on KTH (real per-frame masks, independent `generate_occlusion_branch` occluder
per frame at density 0.35, frames aligned by GT bbox center as a stand-in for
`align_frames`' Kalman motion estimate). Scored as reconstruction MAE against the clean center
frame, restricted to person pixels that are actually occluded in the center frame.

Two methodological corrections were needed before the numbers meant anything, both of which
initially produced a *false positive* for the proposal:

1. **Untuned baseline.** Compared against `fuse()`'s literal default-ish `sigma=2.0`,
   gait-weighting looked ~10% better. But `gait_weights` has two free hyperparameters and the
   temporal baseline has one; sweeping the temporal sigma erased the entire advantage.
2. **Degenerate metric.** Scoring MAE over the whole person mask rewards *not fusing at all* -
   the occluded center frame already has perfect pose, so a weighting that puts ~79% of its mass
   on the center frame (`sigma=0.5`) won outright while removing no occluder. Fixed by scoring
   only pixels that are occluded in the center frame - the only ones fusion can improve.

With both fixed (n=34 windows, seq_size=15, nth_frame=1, spanning ~1.5 gait cycles):

| method | MAE on recoverable pixels |
|---|---|
| no fusion (center frame only) | 23.93 |
| `mean` | 25.42 |
| `median` | 23.08 |
| gait_weighted | 22.04 |
| **temporal_gaussian (sigma=1.0)** | **19.75** |

**Temporal proximity is a better proxy for "safe to fuse" than gait-phase similarity here.**
Immediately adjacent frames are already near-identical in pose *and* carry independently-drawn
occluders - exactly what fusion needs. Gait-similarity weighting dilutes weight away from those
ideal neighbours to reach same-phase frames a full period away, where the person has translated
and residual pose/appearance drift costs more than the occluder it removes. At the shorter
`seq_size=7, nth_frame=2` window the two are roughly tied (23.94 vs 23.71) and *both* barely beat
not fusing at all (24.85), so nothing was gained there either.

**Narrower claim that survives:** gait-phase weighting could still matter in the regime this test
does *not* cover - a person occluded across a long *contiguous* stretch, where every temporally
nearby frame is also occluded and the only usable material is a distant same-phase frame. That is
a different experiment (deliberately correlated, persistent occlusion rather than independent
per-frame occluders) and remains untested.

## Open questions / not yet built

- No code has been written for this - this is a design doc, following the same review-before-
  build pattern as `docs/deepsort_blob_scoring_compatibility.md`.
- Exact form of the gait-consistency weight (a soft continuous weight into `fuse`, vs. a hard
  reject like the Kalman tracker's `max_dist` gate) is undecided.
- Circularity risk: if gait-phase is used both as the fusion weight and as something the network
  is separately learning to predict, the two must stay clearly separated - weighting should come
  from real/measured shape at training-data-construction time, not from the network's own
  in-progress prediction feeding back into its own input.
- Neither `mask_background` mode of `integrate()` has ever been validated as segmentation-network
  input before (both exist for a detector consumer) - this needs its own check before assuming
  either is a reasonable starting point.
- This is a natural next phase gated on the in-flight anisotropic-heatmap training experiment's
  result (does the network even learn the gait-shape target well from raw pixels first) - not a
  replacement for it, and not scoped to start before that result is in.
