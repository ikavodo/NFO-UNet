# Anisotropic (gait-shape) heatmap experiment - measured findings

Companion to the experiment implemented on `anisotropic-heatmap-training`
(`gen_data/gen_kth_data/gen_anisotropic_heatmap.py`, `config/train_config.py`'s
`kth_train_anisotropic`, `eval/compare_anisotropic_baseline.py`). Records measured results so
they are not re-derived.

## 1. The network does learn gait-shape from raw pixels (positive)

`eval/compare_anisotropic_baseline.py`, n=300 held-out KTH validation samples, anisotropic
checkpoint at epoch ~31 (pre-NaN) vs. the fully-converged circle baseline:

| | baseline (circle, converged) | anisotropic (epoch ~31) |
|---|---|---|
| localization error, windowed centroid | 3.03 px mean / 2.71 median | 4.93 px mean / 4.33 median |
| gait-signal recovery vs. real GT mask | n/a | orientation err 2.2 deg median, eccentricity err 0.013 median |

Gait-shape recovery is strong - the network predicts real 2nd-moment shape from pixels alone,
closely matching the GT mask's own statistics. Localization is currently worse than the baseline,
but the comparison is not yet fair: the anisotropic run was still mid-training (epoch ~31 of 100,
early stopping never triggered) while the baseline early-stopped at epoch 45. Rerun after
convergence before concluding anything about a localization tradeoff.

**Measurement caveat that mattered:** extracting the predicted position by `argmax` (the existing
`MaxEval` convention) is a poor estimator for a spread-out, low-peak anisotropic target. A
whole-image weighted centroid is worse still - it inherits a confound, since MSELoss-trained
output is more diffuse than LogisticLoss-trained output, so whole-image centroid punished the
anisotropic model for its loss function rather than its position accuracy (it reported 11.78 px
vs. the windowed centroid's 4.93 px). The fair estimator is a weighted centroid restricted to a
local window around the argmax, computed identically for both models.

## 2. Domain transfer: eccentricity transfers, absolute scale does not

Shape statistics of the masks the anisotropic target is derived from (n~1500 each,
`*_sammask.png`; KTH = single-prompt SAM2, NFO = union-of-checkpoints pseudo-masks):

A first pass compared all masks unconditionally and found eccentricity nearly identical (KTH
0.936 +/- 0.062 vs NFO 0.934 +/- 0.060). A peer session correctly challenged that as comparing
different objects: KTH masks are whole silhouettes (~5188 px^2) while the average NFO pseudo-mask
is a ~300 px^2 fragment, and a thin sliver of a torso is highly eccentric too, so the agreement
could be coincidental. Re-measured conditioning on masks that actually cover the person -
coverage `(mask ∩ box)/box` in [0.30, 1.0] and at most 25% of mask lying outside the box, which
excludes both fragments and over-segmentation blowouts:

| statistic | KTH person01_jogging_d1 | NFO seq1 | seq2 | seq3 | seq4 |
|---|---|---|---|---|---|
| coverage (mask∩box)/box, all masks | 0.382 | 0.215 | 0.256 | 0.289 | 0.200 |
| eccentricity, all masks | 0.933 +/- 0.080 | 0.950 +/- 0.054 | 0.926 +/- 0.050 | 0.934 +/- 0.036 | 0.909 +/- 0.097 |
| **eccentricity, well-covered** | **0.959 +/- 0.020** | **0.950 +/- 0.018** | **0.945 +/- 0.021** | **0.941 +/- 0.024** | **0.948 +/- 0.014** |
| well-covered n / total | 78 / 90 | 98 / 395 | 113 / 399 | 183 / 400 | 59 / 400 |
| major-axis sd, well-covered | 34.3 px | 12.1 | 10.7 | 10.8 | 10.7 |
| mask area, well-covered | 2476 px^2 | 440 | 406 | 420 | 405 |

**The eccentricity claim survives, and is better evidenced than before.** On the properly
conditioned subset KTH 0.959 vs NFO 0.941-0.950 - agreement within 0.01-0.02, with tight and
comparable spreads (+/-0.02 on both sides, versus +/-0.05-0.10 unconditioned). The alternative
outcome the peer raised (that the well-covered subset would be nearly empty, which would itself
be the finding) does not hold: 15-46% of NFO frames qualify.

The fragment concern is nonetheless real for the *average* NFO mask, and it is a mask-quality
property rather than a domain-scale property. Measured person height is 127.6 px (KTH) vs
52.6-55.5 px (NFO), a **2.34x** genuine scale ratio - consistent with this repo's documented
45-64 px NFO / median-109 px KTH figures. Scale alone therefore predicts a 5.5x area ratio, but
the all-masks area ratio is ~17x (5188 vs ~300), so the average NFO mask captures roughly a third
of the scale-predicted silhouette. On the **well-covered** subset that deficit essentially
vanishes: 405-440 px^2 against a scale-prediction of 2476 / 2.34^2 = ~452 px^2, i.e. 90-97%.

**Absolute scale does not transfer: ~2.9-3.2x** on well-covered masks (34.3 vs 10.7-12.1 px),
decomposing into a 2.34x genuine person-scale ratio plus a residual ~1.3x. This independently reproduces this
project's known dominant KTH->NFO factor from a new instrument (mask 2nd moments), matching
`docs/training_failure_hypotheses.md`'s 2x-upscale result (precision 0.52 -> 0.98) and a peer
session's finding that OpenCV HOG fires on 0% of NFO windows at 1x but 30% at 2x.

Why this is a sharper problem for the anisotropic target than for the circle baseline: the CIRCLE
heatmap's radius is a fixed fraction of image size (`kth_config.hm_circle_radius = 0.07`),
completely independent of how large the person is, so the network never has to learn a
person-size -> blob-size mapping. The anisotropic target's extent IS the person's own covariance,
so that mapping is exactly what it must learn - and any residual train/deploy scale mismatch
shows up directly in its output.

`rand_zoom_out(0.4, 1.0)` in `kth_train`'s transforms mitigates this and is already correctly
applied to the heatmap as well as the frames (`utils/transform_utils.py:161` shrinks `hm` with
the same factor), so this is not a bug. But it does not fully cover the gap: scale factors are
sampled uniformly on [0.4, 1.0] (mean 0.7), while matching NFO's measured 3.1x needs ~0.32 -
below even the minimum. Most training samples therefore sit well above NFO's true scale.

Two options if this experiment is to transfer to NFO, neither tried yet:
1. widen `rand_zoom_out`'s range for this config specifically, or
2. render the target from a **box-size-normalized** covariance - keep the eigenvalue ratio and
   orientation (both shown above to transfer) and drop the absolute scale term (shown not to).
   Option 2 is the more targeted fix, since it removes the one component measured not to transfer
   while preserving the one measured to transfer nearly perfectly.

**Not over-read:** the orientation statistic was computed with linear mean/std on a circular
(mod-180) quantity and came out near-uniform in both domains (std ~81 deg), so it is not
informative as measured here. Only the eccentricity and scale rows above are trustworthy.

## 3. Also relevant, measured elsewhere

- Fill fraction inside the GT box depends entirely on which mask source is used, and the ordering
  between datasets **reverses** between sources: MOG2 motion masks give NFO 0.475 > KTH 0.230,
  while SAM2 semantic masks give KTH 0.382 > NFO 0.239. KTH is hard to segment by motion and easy
  to delineate semantically; NFO is the reverse. Any fill number needs its mask source attached -
  only the SAM2 numbers bear on this training work.
- A 3px dilation alone raises KTH box fill 0.382 -> 0.541 (+42%), so mask-construction choices
  (the NFO pseudo-masks union multiple propagations and clip to a dilated box) materially move
  any fill-based statistic.
- Two measurement traps hit while producing the table above, both worth avoiding:
  (i) **blob counts must filter speckle.** An unfiltered `connectedComponentsWithStats` count on
  KTH masks gave a nonsensical 4.75 blobs/mask with only 11% of masks having >1 (mutually
  inconsistent on their face); with a 20 px minimum blob area it is 1.01 mean / 1% >1, i.e. KTH
  masks are single whole silhouettes. NFO is 1.13-1.39 filtered.
  (ii) **two different "fill" definitions.** `(mask ∩ box)/box` answers "how much of the box is
  person" and is the one to use; `mask_area/box_area` double-counts mask lying outside the box and
  gave 0.732 where the correct figure is 0.382 for the same KTH masks (8% of KTH mask area falls
  outside the annotation box; NFO's is 0.1-1.2% because those pseudo-masks are clipped to the box
  by construction). Also: selecting a "high fill" subset by top quartile selects KTH's
  over-segmentation *failures*, not its best masks - bound the criterion from above as well.
