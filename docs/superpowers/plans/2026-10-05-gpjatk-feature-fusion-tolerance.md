# GPJATK test: where should frames be fused for SAM2 under fragmented occlusion?

Status: plan, revision 2 (2026-10-06), self-contained for a separate session and model. Runs in
`master_thesis` on its existing GPJATK testbeds. No NFO data is needed. Revision 2 adds the
literature pass, late fusion and memory-side fusion arms, the ceilings and controls, and the
stride argument.

## 0. Why (what the NFO pilot established, 2 segments, descriptive)

Source: `NFO-UNet` branch `vos-benchmark-pilot`, `docs/nfo_vos_pilot_findings.md`,
`docs/nfo_failure_log.md`.
- SAM2 (Hiera-B+) under fragmented occlusion fails mainly at **initialisation**. SAMURAI's
  motion-aware memory alone does not fix it (J&F 0.779 / 0.773).
- **Pixel-level integration** (align past frames by a constant-velocity model, per-pixel median)
  helps amodal extent but not modal masks. It is capped by gait articulation: translation-only
  alignment works only up to about 7 frames, and more frames or stride make it worse even with
  oracle alignment.
- **Composite-prepend** (SAM2 cut-out of the integrated person pasted onto background as a
  synthetic, prompted frame 0) helps SAMURAI: 0.773 → 0.843 with GT alignment, 0.806 with
  GT-free tracker alignment. Vanilla SAM2 collapses on heavily occluded starts with the GT-free
  version.
- The NFO tracker's GT-free alignment error is a median of 7.7 native px. The person is about
  190 × 65 native px, so the error is about 12% of the person's width.

**Question:** is the articulation/misalignment cap a property of fusing *pixels*, or would
fusing SAM2 *features* (or SAM2 outputs, or SAM2 memory) tolerate it?

## 1. Prior work (from a novelty pass on 2026-10-06; re-check each before citing in a paper)

- **FGFA**, Flow-Guided Feature Aggregation (Zhu et al., ICCV 2017, arXiv 1703.10025): warps
  CNN features along learned flow and averages them with cosine-similarity weights, trained
  end-to-end. "Feature-level aggregation along motion paths helps degraded frames" is theirs.
  Ours differs: a frozen foundation model, no learning, static fragmented occluders, and an
  amodal vs modal split.
- **Temporal Probability Smoothing for SAM2 under Weak Prompts** (arXiv 2604.17115):
  training-free; warps SAM2 *output probabilities* with flow and blends them. This is the
  late-fusion competitor (arm L).
- **SAMWISE** (CVPR 2025, arXiv 2411.17646): a trained adapter that injects text and temporal
  context into SAM2's encoder features. Evidence that temporal modulation of encoder features
  works; not training-free.
- **MA-SAM2** (MICCAI 2025, arXiv 2507.09577): training-free memory management (confidence/IoU
  gating, multi-hypothesis). No feature fusion. A candidate fix for SAM2's collapse later, not
  an arm here.
- **Airborne Optical Sectioning** (Schedl, Kurmi, Bimber; arXiv 2009.08835, 2111.06959): registers
  and integrates frames to see people through foliage, so it is prior art for arm P. Cite it.
- Related, no encoder fusion: SAMURAI 2411.11922, DAM4SAM 2411.17576, SAM2Long 2410.16268, MoSAM
  2505.00739, TABE 2411.19210.
- A ~15-query pass did **not** find training-free fusion of frozen SAM2 image embeddings across
  motion-aligned frames. Treat that as "not found", not "new".

## 2. Hypothesis and mechanism (write down before running)

- SAM2's image encoder works at 1024×1024. The predictor exposes `image_embed` at 64×64 (stride
  16) and high-res features at 128×128 (stride 8) and 256×256 (stride 4). **Verify the names and
  shapes in the installed `sam2/sam2_image_predictor.py` (`_features`, `_bb_feat_sizes`) first.**
- A misalignment of ε input px moves a feature by ε/stride cells. Prediction: embedding fusion is
  near-invariant while ε ≪ 16 px (1024 space), and pixel fusion needs ε ≲ 1–2 px. **Any tolerance
  gain must be reported against the stride.** Otherwise it is just a coarse-resolution artefact.
- Hiera mixes context across positions, so a frame's embedding does not split into person plus
  occluder, and the median of encoder outputs ≠ the encoder output of the median image. Whether
  the median of embeddings still decodes to a cleaner person is the open question.

## 3. Data and tools

- Testbeds: `master_thesis/results/gpjatk_em_viability/testbed_d{50,65,80}/`, each with
  `occluded_video.pt` [T,1,H,W] float, `clean_video.pt`, `occlusion_mask.pt` (True = occluded),
  `silhouette_mask.pt` (**amodal GT**), `visible_mask.pt` (**modal GT**), and
  `gt_motion_params.pt`. Built by `experiments/prototypes/gpjatk_em/em_severe_occlusion_testbed.py`.
- Alignment: `warp_tensor` (`master_thesis/src/registration.py:752`) with the GT motion
  (`compute_motion_params`, `:659`), plus injected error (§4).
- Env: `master_thesis/.venv` (sam2 1.1.0). Checkpoint:
  `/home/akovi/PycharmProjects/samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt`
  (`configs/sam2.1/sam2.1_hiera_b+.yaml`). fp16 autocast for every SAM2 call (as in the NFO
  pilot).
- Work on a new `master_thesis` branch, e.g. `gpjatk-fusion-location`. Outputs go to
  `results/gpjatk_em_viability/fusion_location/`.

## 4. Design: one variable = where the frames are fused

Same aligned buffer, same oracle prompt, same checkpoint for every arm. The buffer is the N
frames ending at target t (causal).

| arm | fusion location | ancestor |
|---|---|---|
| **A0** | none: frame t alone | do-nothing |
| **P** | pixels: per-pixel **median** (and mean) of aligned frames → SAM2 image | AOS / NFO pilot |
| **E** | encoder: encode each aligned frame; fuse `image_embed` by **median** and by **FGFA cosine weighting** (per location, weight ∝ max(0, cos(f_k, f_t))); keep frame t's high-res features. Also a variant that fuses all levels (decoder-consistency check). | FGFA |
| **L** | output: SAM2 image on each aligned frame, then median or cosine-weight the **logits** | 2604.17115 |
| **M** | memory: SAM2 **video** predictor on the aligned buffer; prompt the N−1 past frames, predict frame t from memory alone (frame t unprompted) | SAM2 / SAMURAI / DAM4SAM |

**Ceilings and controls:**
- **Clean ceiling:** A0 on `clean_video[t]`, which bounds every arm.
- **Articulation-free control R:** N copies of `clean_video[t]`, each occluded by a different
  frame's occluder. For each k, warp `occlusion_mask[k]` and the occluder pixels of
  `occluded_video[k]` into t's coordinates and composite them over clean frame t. This gives
  perfect alignment and no articulation, so it isolates the occlusion-averaging gain. Run P and E
  on R.
- **Prompt (oracle, identical across arms):** box = bbox of `silhouette_mask[t]`, point =
  distance-transform maximum of `visible_mask[t]`, `multimask_output=False`. For M, prompt each
  past frame with its own GT box + point, in aligned coordinates.

**Settings:**
- Injected linear velocity error Δv ∈ {0, 0.5, 1, 2, 4} px/frame (native), so frame t−k is
  shifted by k·Δv. Convert to 1024 input px and report ε_max = (N−1)·Δv·s against stride 16,
  where s = 1024/max(H,W). Also mark the NFO-like point, where the error at age 6 is ≈ 12% of the
  person's width.
- N ∈ {7, 13}. 25 is optional: the NFO pilot showed pixel fusion degrading past 7.
- Densities d50, d65, d80. Targets: every 5th t with a full buffer.

## 5. Metrics

Per target: IoU vs `silhouette_mask` (amodal) **and** vs `visible_mask` (modal), reported
separately for every arm. Fusion should enlarge masks toward amodal, raising one score and
lowering the other. Also report mean over targets per (arm, density, N, Δv), the IoU(Δv) curves,
wall-clock per target and peak GPU memory. The pipeline is deterministic (no seeds); say so in the
run record.

## 6. Pre-registered decision rules

1. **Encoder fusion is motivated** if, at the NFO-like Δv and N = 7, best E − P ≥ +0.03 amodal IoU
   at ≥ 2 of 3 densities, **and** E ≥ L and E ≥ M. If L or M is as good, fusing outputs or
   memory suffices, and there is no encoder story.
2. **Tolerance, not resolution:** E's IoU drop from Δv = 0 to the largest Δv is smaller than P's,
   reported against ε/stride. If E's advantage appears only where ε < 16 px (1024 space), call it
   a resolution effect.
3. **Articulation is the bottleneck** if P on R ≫ P on real motion. If E closes most of that gap,
   that supports feature fusion. If neither does, the cap is integration itself.
4. **Stop** if every fused arm < A0 at Δv = 0: single-frame SAM2 is already the ceiling.
5. Modal scores are reported and never used to choose an arm. The NFO benchmark's modal question
   was settled separately.

## 7. Engineering

- Cache aligned stacks per (density, N, Δv) and encoder features per aligned frame. Encoding
  dominates the cost, and arms E, L and P-on-R reuse it.
- Visual traces (dpi ≈ 80): per density, 4 targets × arms at Δv ∈ {0, NFO-like}, N = 7, showing
  the mask with silhouette and visible contours. Also one image of the fused embedding's first 3
  PCA components against the frame-t embedding.
- Run record: master_thesis commit, sam2 version, checkpoint, GPU, and the alignment-error
  definition.
- Rough cost (assumed): 27 targets × 3 densities × 2 N × 5 Δv, about 20k encodes at N = 13.
  Minutes to an hour on a cluster GPU. Subsample targets if needed.
