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

| statistic | KTH (train domain) | NFO (deploy domain) |
|---|---|---|
| eccentricity | 0.936 +/- 0.062 (p5 0.810, p95 0.981) | 0.934 +/- 0.060 (p5 0.822, p95 0.989) |
| major-axis sd | 32.6 px | 10.4 px |
| blobs per mask | 4.75 mean, 11% have >1 | 1.40 mean, 32% have >1 |

**Eccentricity - the statistic that actually carries the gait-phase signal - transfers almost
exactly.** Same mean, same spread, same tails. That is a real positive for this direction: a
KTH-trained anisotropic head predicts eccentricity into the correct numeric regime on NFO.

**Absolute scale does not transfer: 3.1x** (32.6 vs 10.4 px). This independently reproduces this
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
