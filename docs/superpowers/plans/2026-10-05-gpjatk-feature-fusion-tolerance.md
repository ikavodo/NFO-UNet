# GPJATK test: where should frames be fused for SAM2 under fragmented occlusion?

Status: plan, revision 3 (2026-10-06): simplified to 4 arms + 1 ceiling, adds a representation check before decoding and anticipated results. Previously revision 2 (2026-10-06), self-contained for a separate session and model. Runs in
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

## 4. Design: one variable = where the frames are fused (kept simple)

Same aligned buffer (the N frames ending at target t, causal), same oracle prompt, same checkpoint.

| arm | fusion location | ancestor |
|---|---|---|
| **A0** | none: frame t alone | do-nothing |
| **P** | pixels: per-pixel median of aligned frames → SAM2 image | AOS / NFO pilot |
| **E** | encoder: encode each aligned frame; fuse `image_embed` per 64×64 location by (i) median and (ii) FGFA cosine weighting, w_k ∝ max(0, cos(f_k, f_med)); keep frame t's high-res features; decode with the prompt | FGFA |
| **M** | memory: SAM2 **video** predictor on the aligned buffer. The N−1 past frames are prompted (each with its own GT box + point, aligned) and become conditioning memories. **Frame t is not prompted**, so its mask comes only from memory attention over the past frames | SAM2 / SAMURAI |
| **C** | ceiling: A0 on `clean_video[t]` | upper bound |

- **Prompt (oracle, identical across arms):** box = bbox of `silhouette_mask[t]`, point =
  distance-transform maximum of `visible_mask[t]`, `multimask_output=False`.
- **Injected linear velocity error** Δv ∈ {0, 1, 2, 4} px/frame (native): frame t−k is shifted by
  k·Δv. Report ε_max = (N−1)·Δv·s in 1024 input px against stride 16 (s = 1024/max(H,W)), and
  mark the NFO-like point (error at age 6 ≈ 12% of the person's width).
- N = 7 only (the NFO pilot showed pixel fusion degrading past 7; N = 13 is optional).
- Densities d50, d65, d80; every 5th target t with a full buffer.
- **Optional, not required:** late fusion of SAM2 outputs, ViperSAM (arXiv 2604.17115). No public
  code was found, and it relies on optical flow, which is unreliable under fragmented occlusion.
  Reimplement only if time allows (flow warp + entropy / forward-backward weighted blend).

### 4.1 Representation check first (cheap, before any decoding)

The user's earlier sanity check found that the FGFA-style fused embedding has a higher cosine
similarity to the **clean** frame's embedding than any single aligned frame does. That result is
**expected from averaging alone** (derived): write each aligned embedding at a location as
f_k = c + n_k (c clean, n_k the occlusion perturbation). The mean is c + n̄. If the n_k are roughly
independent and zero-mean with per-dimension variance σ² in D dimensions, |n̄|² ≈ Dσ²/N and

  cos(f̄, c) ≈ |c| / √(|c|² + Dσ²/N),

which rises with N for *any* averaged noisy copies. So a higher cosine is necessary, not
sufficient. Two refinements make it informative:
1. **Compare against P's embedding** (the encoder of the median image), not only single frames.
   The question is whether fusing after the encoder beats fusing before it.
2. **Split by location:** downsample the masks to 64×64 and report cos(·, c) separately for cells
   that are person-occluded in frame t, person-visible in frame t, and background. The gain that
   matters is on the person-occluded cells.

Report this for every Δv. If E ≤ P on the person-occluded cells here, skip decoding: the encoder
location has no advantage to decode.

## 5. Metrics

Per target: IoU vs `silhouette_mask` (amodal) **and** vs `visible_mask` (modal), reported
separately for every arm. Fusion should enlarge masks toward amodal, raising one score and
lowering the other. Also report mean over targets per (arm, density, N, Δv), the IoU(Δv) curves,
wall-clock per target and peak GPU memory. The pipeline is deterministic (no seeds); say so in the
run record.

## 6. Anticipated results (written before running) and decision rules

Anticipated, with reasons:
- **Representation (4.1):** E > A0 on person-occluded cells at every Δv (averaging). E ≈ P at
  Δv = 0, and E > P as Δv grows while ε < 16 px (one embedding cell), then converging. Cosine
  weighting ≥ median, because per-channel medians can leave the feature manifold.
- **Decoded amodal IoU:** E > A0 modestly. E ≥ P only at Δv > 0. The decoder was trained on
  single-frame embeddings, so a fused embedding may be partly out of distribution and lose part
  of the representation gain.
- **Decoded modal IoU:** E ≤ A0. Fused representations contain person that is hidden in frame t,
  so masks grow toward amodal, the same definitional conflict as pixel integration on NFO.
- **M:** memory attention reads past frames' *modal* masks, so M should be the best modal arm (≥ A0),
  but limited for amodal (≈ A0). Predicted pattern: **E wins amodal, M wins modal.**

Decision rules:
1. **Encoder fusion is motivated** if, at the NFO-like Δv, best E − P ≥ +0.03 amodal IoU at ≥ 2 of 3
   densities and E ≥ M on amodal IoU.
2. **Memory fusion is the route** if M ≥ E on amodal as well. In that case, invest in SAM2's memory
   (e.g. composite-prepend, memory selection), not in the encoder.
3. **Resolution, not tolerance:** if E's advantage over P exists only for ε < 16 px, report it as
   a stride effect.
4. **Stop** if every fused arm < A0 at Δv = 0.

## 7. Engineering

- Cache aligned stacks per (density, N, Δv) and encoder features per aligned frame. Encoding dominates the cost; arms E and the 4.1 check reuse it.
- Visual traces (dpi ≈ 80): per density, 4 targets × arms at Δv ∈ {0, NFO-like}, N = 7, showing
  the mask with silhouette and visible contours. Also one image of the fused embedding's first 3
  PCA components against the frame-t embedding.
- Run record: master_thesis commit, sam2 version, checkpoint, GPU, and the alignment-error
  definition.
- Rough cost (assumed): 27 targets × 3 densities × 4 Δv × N = 7 encodes, about 2.3k encodes plus the video-predictor runs for M.
  Minutes to an hour on a cluster GPU. Subsample targets if needed.
