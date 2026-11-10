# Learned, threshold-free candidate selection for SAM2 trackers, trained on satellite tracking datasets (idea record, 2026-11-10)

Status: **idea, not started.** Gated on the MVOT headroom / failure-anatomy check
(`2026-11-10-mvot-integral-image-oracle-design.md`, H0 / H0b) and on a novelty pass. Decision of the user (2026-11-10):
learn every threshold from data; use the many box-labelled satellite tracking datasets for it.

## Why

- SatSAM2 (arXiv 2511.18264, `docs/external/satSAM2.md`) = SAMURAI-style memory selection + Kalman filter + a state
  machine, with ~9 hand-set parameters (Table II: α_kf, τ_h, τ_m, D, d_max, T_f, T_m, ...). The Kalman filter's only
  measurement is SAM2's own mask, so smooth mask errors (boundary leakage, low-contrast drift, size bias; their §VI-B)
  and distractors (their Fig. 1b, "vehicles moving in opposite directions") pass its gates.
- Our integral image adds evidence that is independent of SAM2's mask: pixels that differ from the empty scene and move
  consistently with the hypothesised target motion. Its hand-made form also has knobs (gate z₀, noise floor, τ, window
  H, its weight in the score). Replace all of them with features + a learned scorer.

## Formulation

Per frame, SAM2 returns 3 candidate masks M_j (prompted with the Kalman box, as in SatSAM2). For each, a feature vector
φ_j of raw, unthresholded statistics; a small MLP h_θ(φ_j) regresses IoU(box(M_j), GT box); the tracker takes
argmax_j h_θ. SAM2 frozen; a few thousand learned parameters.

| feature | meaning | replaces |
|---|---|---|
| s_sam, s_obj | SAM2 predicted IoU, object score | τ_h, τ_m |
| d²_j = (c_j − ĉ)ᵀ S⁻¹ (c_j − ĉ) | Kalman innovation (Mahalanobis) | d_max, τ_kf |
| log(area_j / S₀) | size change vs t0 | D |
| SNR_j = mean_{M_j} \|Ā − B\| / σ̂ | contrast of the motion-aligned average vs empty scene, in noise units | gate z₀, noise floor |
| C_j(τ), τ ∈ {3, 12, 30} | stack consistency (variance across aligned past frames inside M_j), all scales given | choice of τ, H |
| ρ_j = C_j(v̂) / max_v C_j(v) | shift-and-stack velocity likelihood ratio: does pixel evidence agree with the Kalman velocity? | drift check |
| cos(f_T, f_{M_j}) | SAM2-feature similarity to an integral-image template of the target | co-moving distractors (identity by appearance) |

Possible second stage: learned recency weights (attention over past frames) instead of fixed τ scales.

Mechanism notes (derived, this session): a distractor with relative velocity Δv and length L keeps evidence
a_d ≈ Σ_k w_k max(0, 1 − |Δv|k/L); separation needs |Δv|·T > L. Co-moving distractors (Δv ≈ 0) cannot be separated by
motion at any window, only by position, appearance (template) or mutual exclusion (multi-target assignment).
Measured on 10 MVOT similar-object videos: with τ = 12 / 30 frames, non-target moving blobs 870 → 527 (×0.61),
target evidence unchanged; with τ = 3.5 / 6 frames no suppression (cars move 1–3 px/frame, ~20 px long).

## Training data (boxes suffice)

Run SAMURAI on box-labelled satellite videos, log all 3 candidates per frame with φ_j and their IoU to GT.
Datasets (as listed in SatSAM2, not yet checked for licence / access): VISO, SatSOT, SAT-MTB, OOTB, MVOT (synthetic,
local at `~/PycharmProjects/MVOT`, HF `Frank0666/MVOT`). Train on VISO + SatSOT + SAT-MTB (+ MVOT train split), test
cross-dataset on OOTB and MVOT (SatSAM2's protocol: AUC, P@20, P_norm).

Baselines: SAM2, SAMURAI, SatSAM2 (published: MVOT AUC 52.68 / 52.88 / 68.17; OOTB 63.52 / 60.35 / 67.23; reported,
their protocol). Ablations: learned scorer on SAM2 + Kalman features only (= a learned SatSAM2) vs + pixel-evidence
features — the one variable that tests our contribution.

## Known problems

- Exposure bias: candidates depend on earlier choices; retrain on rollouts of the learned scorer (DAgger).
- Real satellite video drifts with the platform: the empty scene needs frame registration (homography for near-nadir
  views). MVOT is static (measured: 0.00 px shift over 50 frames, 24 videos); the real datasets are not checked.
- Rendered / compressed video: the robust noise scale collapses (MAD → 0); the SNR feature needs a noise estimate that
  survives this (measured on MVOT, 2026-11-10).
- Small targets (MVOT median 19×20 px, 1–2 SAM2 stride-16 cells): feature-level template similarity may be weak.

## Before building

1. MVOT headroom + failure anatomy (running 2026-11-10, `tools/headroom.py`): which failure class dominates decides
   which features matter (evidence / velocity likelihood vs template).
2. Novelty pass (15–30 min): learned re-ranking / quality prediction of SAM2 candidates (SAMURAI, DAM4SAM, SAM2.1++,
   IoU-Net / ATOM-style IoU regression) and learned selection in satellite trackers.
3. idea-critic pass on the claim "pixel-evidence features make a learned selector beat Kalman-only selection on
   distractor and low-contrast failures" (ask first).
4. Dataset access and licences for VISO / SatSOT / SAT-MTB / OOTB.

Natural MSc-thesis scope (see the interview doc): features + scorer + cross-dataset evaluation.
