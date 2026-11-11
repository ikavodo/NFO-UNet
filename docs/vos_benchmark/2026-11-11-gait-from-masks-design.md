# Gait (walk / jog) from masks under fragmented occlusion: pilot design (2026-11-11)

**Question.** Do better visible-part masks support action recognition when the box cannot? This is a cheap pilot
that decides whether a mask-to-action study on PanAf500 (Brookes et al., IJCV 2024; `docs/external/PanApe20K.md`) is
worth doing.

**Label.** Binary: NFO `walk` vs NFO `run`, where NFO `run` ≙ KTH `jogging` (2026-11-11, measured from GT boxes: the
run/walk speed ratio is NFO 1.6, KTH jog 1.6–1.9, KTH run 2.4–3.0; the step-rate ratio is NFO 1.21, KTH jog 1.40, KTH
run 1.56). KTH `running` is not used. Evidence for the NFO label: `benchmark/nfo_vos/tools/gait_check.py`.

## Information setup

- **Input at frame t.** The k stabilised masks of frames t−k+1 … t (causal). Each mask is cropped on a canvas that
  moves with the **GT** trajectory, which is the linear fit of the GT box centre over the crossing (speed removed
  exactly). The canvas height is 1.2 × the median GT box height of the crossing, it is resized to 24 × 16, and it is
  flipped so that motion is always rightward.
  - The GT is used for stabilisation only, and identically for every arm, so the mask is the only variable.
  - Translation speed and scale carry no information after this step.
  - Pose, limb configuration and step rhythm remain.
- **Label.** The crossing's gait, constant over the crossing.
- **Classifier.** Logistic regression (L2) on the flattened k × 24 × 16 stack, one model per k ∈ {1, 4, 8, 16, 24}. It is
  trained on KTH walking + jogging, scenarios d1, d3 and d4 (d2 zooms), at 25 fps. C is chosen by subject-grouped
  5-fold CV on KTH. Nothing is tuned on NFO.
- **Training masks.** SAM2.1-b+ image predictor prompted with the KTH GT box. They are trained in two variants:
  - *clean*;
  - *occ*: each crossing has a static procedural-foliage occluder (`synth/occluders.occ_perlin`, ρ ~ U[0.2, 0.6],
    10% ρ = 0) applied in frame coordinates, so the occluder slides across the canvas like the real foliage does.
- **Frame rate (measured 2026-11-11).** NFO is 25 fps: ffprobe on the seq1 video gives r_frame_rate = avg_frame_rate
  = 25/1, 2126 frames, 85.04 s, against 2128 JPEGs. KTH is also 25 fps, so there is no temporal resampling.
- **Known threat.** At 25 fps the NFO `run` step period is 13–15 frames (≈ 1.85 steps/s). That is close to the KTH
  *walking* period (14) and slower than KTH jogging (10). A classifier that relies on step rhythm will therefore call NFO
  runs walks, whatever the masks. This affects the absolute oracle accuracy (P1's ≥ 0.60 bar), not the paired
  oracle-vs-box difference.

## Arms

All arms are tested on the NFO 40-frame benchmark windows (local masks).

| row | mask source |
|---|---|
| oracle | SAM2 pseudo-masks from GT prompts (`data/nfo_processed/*/NNNNN_sammask_native.png`) |
| box-only | the filled GT box rectangle, which carries the box-width gait oscillation |
| B0, T1, trk | `b0-c2`, `t1-c2`, `trk` |
| ours | `adapter-pf`, `adapter-s1`, `adapter-s2` (seed mean) |
| KTH ceiling | subject-grouped CV accuracy on KTH |

**Metric.** Per-frame accuracy, averaged within a crossing and then over crossings (balanced classes).
- Headline: seq1–4, 32 crossings. Nothing is tuned on NFO.
- Also reported: seq2–4, 24 crossings.
- CIs: paired bootstrap over crossings.

## Pre-registered criteria (k = 16, occ-trained classifier)

- **P1, feasibility.** The oracle beats box-only, with a paired CI > 0, and the oracle accuracy is ≥ 0.60. If P1
  fails, masks carry no gait information beyond the box under this foliage, and PanAf in this form is not pursued.
- **P2, methods.** Ours (seed mean) beats T1, with a paired CI > 0.

## Result (2026-11-11, laptop, measured)

The KTH training set is 603 crossings (299 jogging) from 25 subjects. KTH subject-CV accuracy at k = 16 is 0.977 (occ)
and 0.986 (clean). Table: NFO seq2–4 (24 crossings), occ-trained, k = 16, per-frame accuracy:

| oracle | box (fair) | B0 | T1 | trk | ours (3 seeds) |
|---|---|---|---|---|---|
| 0.769 | 0.612 | 0.752 | 0.753 | 0.731 | 0.774 (0.766 / 0.770 / 0.787) |

- **Box control as pre-registered was unfair.** The filled rectangle was pushed through the *mask*-trained classifier
  (0.501, chance). The fair control is a classifier trained on KTH GT-box rectangles (`eval/gait_box_control.py`):
  0.612.
- **P1 (masks beyond box): borderline.** Against the fair box, the oracle gains +0.156 [−0.003, +0.317] and ours
  +0.162 [−0.000, +0.327]. The direction is clear but the CI touches 0. Against the pre-registered (unfair) control it
  passes: +0.268 [+0.019, +0.517].
- **P2 (ours > T1): fails as pre-registered.** +0.021 [−0.001, +0.048] (seq1–4: +0.023 [−0.004, +0.057]). The clean
  classifier ablation gives +0.047 [+0.015, +0.081], but it is not the pre-registered arm.
- **Rhythm threat did not materialise.** Per-class accuracy is balanced: the oracle gets walk 0.774 / run 0.763.
- **The oracle (SAM2 pseudo-masks) is not an upper bound here.** It is ≈ T1 and below ours, so recognition is
  insensitive to mask differences of this size.
- **k.** Accuracy rises monotonically to k = 16 (0.68 at k = 1 to 0.77 for ours) and is flat or slightly lower at k = 24.
