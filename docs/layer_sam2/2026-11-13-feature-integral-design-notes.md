# Feature-space integral image and background: design notes (2026-11-13, branch exp/feature-integral)

Status: **design discussion only, no code.** Literature pass and decisions D1–D3 come before any implementation.

## Rhymes with recorded decisions
- `docs/layer_sam2/archive/2026-10-20-one-pass-adapter-design.md`: a feature-space composite with a learned gate.
  Not pursued (user decision, 2026-10-20).
- `docs/layer_sam2/2026-11-11-latent-background-and-integral-single-pass.md`: a latent background and integral
  image computed in one pass. Idea stage only.
- SAM2 memory attention is already cross-attention over the features of past frames. On NFO, the memory variants were
  closed with an oracle bound of +0.012 J (CLAUDE.md).
- **New here:** (i) attention constrained by alignment (motion-compensated positions); (ii) an intermediate
  feature-reconstruction loss against the encoder features of the *unoccluded* frame. Only planted occlusion provides
  those targets.

## Quantities
- E: the frozen SAM2 encoder.
- F_k = E(X_k) for frame k; F_t^clean = E(X_t^clean), the same frame without the planted occluder; F_B = E(B).
- The tracker box (c_k, w_k, h_k) for frame k.

## Integral image in feature space (learned)
Per stride s and token position p in frame t, aggregate the aligned tokens of frames t−6..t−1, together with F_t and F_B:
- **A0 (minimal, in the style of a temporal transformer):** at the motion-compensated position p − (c_t − c_k), attend
  over the 8 tokens {F_t, F_{t−1..t−6}, F_B}, with a learned softmax score per token. This replaces the hand-set
  w (τ = 3.5) and the change gate G. Cost: 8 keys per token.
- **A1:** a 3×3 neighbourhood around each aligned position (72 keys), to absorb residual misalignment and non-rigid
  motion. A deformable variant learns the offsets: Deformable DETR, Zhu et al., ICLR 2021, arXiv 2010.04159.
- **Alignment through positional encoding:** 2-D RoPE on box-relative coordinates u = (x − c_k)/w_k,
  v = (y − c_k)/h_k inside the box, and on image coordinates outside it, with a soft box mask. The logits then depend
  on the ape-relative displacement for ape tokens, and on the static-camera displacement for background tokens.
  - RoPE: Su et al., RoFormer, arXiv 2104.09864.
  - 2-D RoPE for ViTs: Heo et al., ECCV 2024, arXiv 2403.13298.
  - A0 with hard alignment is the zero-bandwidth limit of this.
- **Ancestor:** flow-guided feature aggregation, FGFA (Zhu et al., ICCV 2017, arXiv 1703.10025): warp the features
  of past frames, then weight them per position.

## Background in feature space
- **B0:** per-token temporal median of F_k. Zero parameters. Baseline.
- **B1:** RPCA, F = L + S (Candès et al., J. ACM 2011, arXiv 0912.3599), per video on stride-16 tokens. Analysis
  baseline, cost to be measured.
- **B2:** a learned background query per position that attends over all frames, with target E(B_true). B_true is
  known in planted clips and recorded in FGBG pairs.

## Losses
- The mask loss, as now.
- L_feat = Σ_s λ_s ‖Ĉ_s − F_{t,s}^clean‖² over the box, with the box mask dilated.
- Optionally L_bg = ‖B̂ − E(B_true)‖².

## Decisions open (user)
- **D1. Modal or amodal output.**
  - With L_feat, Ĉ represents the *unoccluded* ape, so a decoder on Ĉ tends towards the full silhouette (amodal).
  - The NFO benchmark scores the *visible* part; the PanAf boxes are amodal.
  - Pick one target per experiment.
- **D2. Ordering.** This needs the same data generator as Layer-SAM Stage 1: planted static occluders on clear
  apes, with clean frames retained. Build that once, then run three arms on identical data, changing one variable:
  SAM2 / pixel-integral adapter / A0.
- **D3. Oracle ceiling first.** Before training anything, feed F_t^clean inside the box to the frozen decoder on
  planted clips and measure the J gain (modal and amodal). If the ceiling is small, stop.

## User decisions, 2026-11-13
- **No feature warping (FGFA style).** Interpolated features are out of distribution for the decoder.
  Alignment comes only from masked attention plus positional encoding in box-relative coordinates. A0 is therefore
  implemented as attention restricted to the aligned position (a hard positional mask), not as a resampled feature map.
- **Occluder source:** the near-depth layer from high-near_edge scenes, PanAf500 first and PanAf-FGBG later.
  Validation pass: renders/depth_cues/near_edge_example.png.
  - Extract it from frames or regions without the ape (the GT box masked out, or FGBG background clips). A close ape
    is otherwise part of the near layer.
  - Pick a crop whose size matches the target box. Place it along the ape's track so that the ape moves *behind* it.
  - Paste it into the frames and into B (static occluder).
  - Check before any training: the fragmentation statistics of planted occluders (visible fragments per box,
    boundary/area, occluded fraction) against real occluded SBR masks, compared with the procedural occluders.

## Literature pass (2026-11-13, research-companion brainstormer; [V] = resolved by search; "nothing found" = 1–3 searches)
- **(a) Aligned masked temporal attention.**
  - Prior work:
    - FGFA, 1703.10025 [V], the warping contrast.
    - SELSA, ICCV 2019, 1907.06390 [V].
    - MEGA, CVPR 2020, 2003.12063 [V].
    - TransVOD, TPAMI 2022, 2201.05047 [V].
    - OVIS temporal feature calibration, IJCV 2022, 2102.01558 [V], the closest in motivation.
    - SOIT, AAAI 2022, 2112.11037 [V]: box-centre-relative PE, sinusoidal, no temporal component.
    - VideoRoPE, 2502.05173 [V].
    - Neighborhood Attention, 2204.07143 [V].
  - Not found: box-relative RoPE inside the box with image coordinates outside. A small delta.
  - Rhymes with the closed SAM2-memory direction.
  - **Kill test first:** a hard-coded box-relative nearest-token attention (no learning) against `-noC`.
- **(b) Clean-feature loss.**
  - Prior work: MaskFeat, 2112.09133 [V]; FCFormer, 2303.01656 [V]; SaVos, NeurIPS 2022, 2210.12733 [V];
    EoRaS, ICCV 2023, 2309.13248 [V].
  - Standard as distillation. The target is amodal and conflicts with the visible-part loss v3.4 → decision D1.
- **(c) Learned background.**
  - Prior work: BSUV-Net, 1907.11371 [V]; refRPCA-Net, 2010.00929 [V]; LRPCA, 2110.05649 [V].
  - Not found: a learned background query in frozen foundation-model features.
  - Baseline: encoder(pixel median) against a per-token feature median, no training.
- **(d) Real-foliage occluders from the near-depth layer.**
  - Prior work: Simple Copy-Paste, 2012.07177 [V]; TAO-Amodal paste-and-occlude, 2312.12433 [V];
    Amodal Cityscapes, 2206.00527 [V]; Sárándi et al., 1808.09316 [V];
    Learning to See Through Obstructions, CVPR 2020, 2004.01180 [V], the closest.
  - Not found: depth-mined real foliage from static-camera footage. **The strongest novel piece.**
  - Caveat: soft depth edges on thin foliage. Report the sensitivity to the threshold.
- **Domain.**
  - SA-FARI, 2511.15622 [V].
  - Yang, Mirmehdi and Burghardt 2019, 1908.11240 [V]: temporal attention for ape detection under occlusion on
    PanAf. The closest prior work in this domain.
- **Baselines a reviewer expects:**
  - SAM2.1, SAMURAI (2411.11922 [V]), DAM4SAM (CVPR 2025, 2411.17576 [V]).
  - SAM2Long (2410.16268 [V]) or Cutie (CVPR 2024, 2310.12982 [V]).
  - Benchmarks: MOSE (2302.01872 [V]), MOSEv2 (2508.05630 [V]), OVIS, SA-FARI.

## Decisions, later on 2026-11-13
- **Occluder examples accepted** (PanAf20K `renders/occluders/examples.jpg`, `tools/occluder_examples.py`).
- **Trajectory-wide planting.**
  - R = union of the GT boxes over the clip window. Fill R with one static foliage field: a single large near-layer
    crop, or a mosaic of smaller crops with feathered alpha.
  - Paste the field into every frame and into B.
  - Reject a clip if any frame's hidden fraction falls outside [0.15, 0.75] for more than 6 frames in a row.
- **Against memorisation.**
  - The occluder pool is all non-eval PanAf train videos plus FGBG background clips (thousands of cameras).
  - Split the occluder sources by camera or video into train and val, so val uses unseen foliage.
  - Augment: flip, scale 0.7–1.6×, small rotation, colour/brightness jitter matched to the IR or colour mode of the
    target.
  - Memorisation test: val J on seen vs unseen foliage.
- **D1 (user): either target is fine if the hidden parts are recovered.**
  - Recommended for the first run: the **visible-part** target. Real-data evaluation then uses the SBR masks
    (visible-part J, recall) and the PanAf500 boxes (whole-ape box IoU and recall).
  - An amodal head comes second.
- **Greatape-Detection** (github youshyee, Yang, Mirmehdi and Burghardt 2019, arXiv 1908.11240) is a temporal
  ape detector on the Pan African camera-trap data, with box XML and checkpoints. Use it as a stratified detection
  baseline next to MegaDetector.
