# GPJATK test: does SAM2 feature fusion tolerate misalignment better than pixel fusion?

Status: plan, to run in a separate session (2026-10-05). Runs in `master_thesis` on its
existing GPJATK testbeds. No NFO data needed.

## 1. Why this test

The NFO VOS pilot (`docs/nfo_vos_pilot_findings.md`, `docs/nfo_failure_log.md`) found that
pixel-level integration (a median over aligned frames) is capped by articulation. A longer buffer
or stride makes it worse, even with oracle translation. Hypothesis: fusing **SAM2 encoder
features** across the aligned frames tolerates misalignment and articulation better than fusing
pixels, because coarse features are pooled over cells much larger than the misalignment.

Named ancestor: FGFA, feature aggregation along motion paths (Zhu et al., ICCV 2017,
arXiv 1703.10025). Ours differs: motion-model alignment instead of learned flow, a frozen SAM2
encoder, no training. SAM2's own memory attention is a learned feature fusion too, and is the
memory-based alternative this test sits beside.

**Step 0 (before writing code, 15–30 min):** a targeted novelty search for training-free
multi-frame feature fusion with SAM/SAM2 (e.g. "SAM feature aggregation video occlusion",
"multi-frame SAM embedding fusion amodal"). Use `research-companion:brainstormer`. If a paper
already does it, cite it and adopt its fusion rule instead of the one below.

## 2. Mechanism (derive before running)

SAM2's image encoder sees a 1024×1024 input. The predictor exposes three feature levels: the
image embedding at 64×64 (stride 16) and high-res features at 128×128 (stride 8) and 256×256
(stride 4). **Verify the exact names and shapes in the installed `sam2/sam2_image_predictor.py`
(`_features`, `_bb_feat_sizes`) before relying on them.**

A misalignment of ε input pixels moves a feature by ε/stride cells. Predicted tolerance:
- embedding fusion is near-invariant while ε ≪ 16 px (1024 space);
- fusing the stride-4 high-res features needs ε ≪ 4 px, so it behaves like pixel fusion;
- pixel fusion needs ε ≲ 1 px.

Therefore the decisive arm fuses **only the 64×64 embedding**, keeping frame t's own high-res
features for boundaries.

## 3. Data

`master_thesis/results/gpjatk_em_viability/testbed_d{50,65,80}/` (from
`em_severe_occlusion_testbed.py`): `occluded_video.pt` [T,1,H,W], `silhouette_mask.pt` (amodal
GT), `visible_mask.pt` (modal GT), `gt_motion_params.pt`. Alignment uses the existing
`warp_tensor` with GT motion. Gait articulation is real, which is the point.

## 4. Design: one variable = fusion domain

Same aligned frames, same prompt, same decoder for every arm:

| arm | what is fused | into the decoder |
|---|---|---|
| A0 | nothing (frame t alone) | frame t features |
| P | pixels: median of aligned frames, then encode | features of the median image |
| F_all | all three feature levels: encode each aligned frame, median over frames | fused features |
| **F_emb** | 64×64 embedding only (median over frames) | fused embedding + frame t's high-res features |

- **Prompt (oracle, identical across arms):** box = bbox of `silhouette_mask[t]`, point =
  distance-transform maximum of `visible_mask[t]`. It is an oracle so that prompt quality cannot
  differ between arms.
- **Injected misalignment:** the oracle warp plus a linear velocity error Δv, so frame t−k is
  shifted by k·Δv px. This matches what the NFO tracker shows: error grows linearly with k, about
  0.5 px/frame at 224, i.e. about 1.8 px/frame native. Scale Δv to GPJATK's resolution:
  Δv ∈ {0, 0.5, 1, 2} px/frame in native px.
- **Buffer:** N ∈ {7, 13, 25} consecutive frames ending at t (causal).
- **Targets:** every 5th frame t that has a full buffer, at all three densities.
- Deterministic pipeline: no seeds. State this in the run record.

Implementation note: `SAM2ImagePredictor.set_image_batch(aligned_frames)`, then take
`_features`, median over the batch dimension, write a batch-1 copy back, then `predict(point,
box)`. Median is the primary rule, matching arm P; mean is secondary.

## 5. Metrics

Per target frame: IoU vs `silhouette_mask` (**amodal, primary**) and vs `visible_mask` (modal,
secondary). Report the mean over targets per (arm, density, N, Δv), plus the tolerance curve
IoU(Δv) for each arm at N = 7. Wall-clock per target and peak GPU memory per arm.

## 6. Pre-registered decision rules

1. **F_emb tolerates misalignment:** at N = 7 and Δv = 1, F_emb − P ≥ +0.03 amodal IoU at ≥ 2 of 3
   densities, and F_emb's drop from Δv = 0 to Δv = 2 is smaller than P's. Then the feature-fusion
   SAM2 variant is motivated.
2. **More frames help only in feature space:** F_emb at N = 13 or 25 beats F_emb at N = 7 (Δv = 1),
   while P does not. Then a longer buffer is worth it, but only with feature fusion.
3. **No tolerance gain:** |F_emb − P| < 0.02 everywhere. Feature fusion is not motivated by
   alignment tolerance. Stop this direction.
4. **Fusion hurts even when aligned:** every fused arm < A0 at Δv = 0. Stop: SAM2 features of a
   single frame are already the ceiling.

## 7. Engineering

- Cache the aligned stacks per (density, N, Δv) and the per-frame encoder features per aligned
  frame. Re-encoding dominates the cost.
- Visual traces (dpi ≈ 80): per density, one montage of 4 target frames × arms at Δv ∈ {0, 1},
  N = 7, with predicted mask vs silhouette and visible contours. Save under
  `results/gpjatk_em_viability/feature_fusion/`.
- Rough cost (assumed): 27 targets × 3 densities × 3 N × 4 Δv. F arms encode N frames per
  target, so about 40k encodes at the largest N. That is minutes to an hour on a cluster GPU.
  Subsample targets further if needed.
- Run record: master_thesis commit, sam2 version, checkpoint (sam2.1 base_plus, as in the NFO
  pilot), GPU.
