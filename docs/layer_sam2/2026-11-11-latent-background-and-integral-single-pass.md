# Latent-space background and integral image in a single encoder pass (handoff, 2026-11-11)

Status: **idea, no experiments run.** Written in a PanAf20K session (`../PanAf20K`) for a session working in this repo.

> **Review note (NFO-UNet session, 2026-11-11).**
> 1. **FGBG test-time background, likely misread (§2.2).** FGBG uses z = z_F − (1−α)·z_B with α going from 0 to 1 by
>    the end of training (their §5.4). At α = 1 the background term vanishes, so evaluation at the end of training uses
>    z_F only (derived from the formula; their code was not checked). The subtraction then acts as a training curriculum.
>    The paper's "reliant on background videos at test time" refers to the dual-stream F + B model. So (A) would remove
>    the paired background at *training* time, not inference. Settle this from their code before E1/E2.
> 2. **(B) overlaps a closed decision.** It is the one-pass adapter of
>    `docs/layer_sam2/archive/2026-10-20-one-pass-adapter-design.md`, "not pursued (user decision)". Reopen only
>    explicitly. Its measured encoder time, 24.8 ms/pass on the laptop, puts one-pass Layer-SAM at ≈ 36 ms/frame vs
>    61 ms now and 53 ms for SAM2 video (derived). The on/off switch (`eval/dual_expert.py`) already limits the second
>    pass to fragmented frames.
> 3. **For PanAf (§6), B need not be latent.** B is encoded once per window, so a pixel masked median or pixel RPCA
>    (Candès et al. 2011, video background subtraction) unblocks the WP1 headroom test at no cost. Priority: WP1 first
>    (`docs/panaf/2026-11-11-msc-project-direction.md`).
Every number below is either quoted from a paper or read from this repo's code, and the source is given. Nothing here
has been measured yet.

## 1. The goal

Show that **empty-scene ("background") estimation and the integral image can be computed in the encoder's feature space
from one encoder pass per frame**, with minimal loss against the current pixel-space versions. This should hold in two
settings:

- **(A) PanAf-FGBG behaviour recognition.** Brookes et al., CVPR 2025, chimpanzee camera-trap video, MViT-V2. This removes
  the need for a *paired background video*.
- **(B) NFO.** This repo's adapter. Computing the integral image's features from cached frame features removes the second
  per-frame encoder call.

**Follow-up direction, not part of the first study.** Integrate the subject across frames *and across videos* to get a
better (super-resolved) view of it. This would generalise the integral image (§7).

## 2. Facts this builds on

### 2.1 This repo (verified in `benchmark/nfo_vos/adapter/inference.py::fast_masks`)
- **Empty scene B.** `bg` is the pixel-wise median of the warm-up frames `w0..w1`, encoded once (`Bf = enc(bg)`).
- **Encoder calls per frame.** `enc(x)` runs on every frame. Once j ≥ `hist` (6), the integral image C (the previous 6
  frames shifted by the Theil–Sen x-fit of the tracker, then combined with gate and recency weights) is **encoded a
  second time**: `Cf = enc(C.clamp(0, 255))`. The comment "one encoder pass per frame" is therefore inaccurate. It is two
  passes after warm-up. The milliseconds saved by removing the second pass have not been timed.
- **Gate.** G is computed in pixel space (`_gate_gpu(x, Bg)`). The tracker already uses a feature-space cue: 1 − cos(F_t, F_B)
  at stride 16.
- The adapter reads `X16/8/4`, `C16/8/4`, `B16/8/4` and the evidence mass `a`. It is trained on `enc(C)`. **Replacing C's
  features with latent-integrated features means retraining the adapter**, and train and test must use the same
  construction (the parity lesson from loss v3.x and the gate threshold).

### 2.2 PanAf-FGBG (Brookes et al., CVPR 2025; the paper is at `../PanAf20K/docs/Brookes_The_PanAf-FGBG_...md`)
- 5,070 pairs of 15 s clips from 389 cameras. Each foreground video (chimp present) is paired with the **temporally closest
  real background video from the same camera**, i.e. a separately recorded empty clip. Labels are video-level multi-label
  (14 classes). There are two configurations: overlapping vs. disjoint camera locations between splits.
- Models: 2D/3D ResNet-50 and MViT-V2, Kinetics-400 initialised, 300 epochs, inputs of 16 × 256², frames sampled at 1/24,
  metrics uAP/mAP.
- **Latent subtraction.** z_{F−B} = z_F − (1−α)·z_B, where z is the last-layer token average (MViT) or the last convolutional
  layer's features (ResNet), from **two passes of a shared-weight encoder**. α rises from 0 to 1 during training. The real
  background video is also needed **at test time**.
- Their headline numbers (Table 2, disjoint mAP):
  | model | baseline | linear α | exponential α |
  |---|---|---|---|
  | MViT-V2 | 35.29 | 39.04 | 39.25 |
  | 3D R50 | 30.89 | 36.31 | 36.36 |
- Background-only models reach 0.65–1.0 of the foreground model's AP (Table 1). Background is predictive of behaviour.
- B̃ (synthetic background) = SAM2 mask with the region filled by the mean pixel value. Real B beats B̃ when used for
  subtraction.
- The conclusion lists "the additional data and compute required for latent space background subtraction" as a limitation.
  That is the motivation for this study.

## 3. The construction

### 3.1 Background in feature space (no warp)
For a static camera, a location whose input never changes gives the same encoder output in every frame, whatever the
non-linearity. Stack the per-frame feature maps at one scale ℓ: D^ℓ ∈ R^{T × (C_ℓ·H_ℓ·W_ℓ)}. Then either:

- **temporal median per location** (the cheap baseline, the latent counterpart of the current pixel median), or
- **robust PCA**: D = L + S, minimising ‖L‖_* + λ‖S‖_1 (Candès, Li, Ma & Wright, JACM 2011). L, the low-rank part, is the
  empty-scene features (it absorbs illumination drift that a median cannot). S, the sparse part, is the subject.
  Since T ≪ C·H·W, compute the SVD via the T × T Gram matrix.

**Feature-space gate:** G_{t,p} = [ ‖S_{t,p}‖ > θ ], or [1 − cos(F_t(p), L_t(p)) > θ] as in the tracker.

### 3.2 Why this makes FGBG single-pass (closed form)
FGBG's pooling is linear, so with the last-layer tokens split as L + S:

  z_F − (1−α)·ẑ_B = α·pool(L) + pool(S),  with ẑ_B = pool(L)

**Latent background subtraction is a learned reweighting of background vs. foreground in the pooled feature.** One pass of
the foreground video provides both parts. The hypothesis to test is pool(L) ≈ E(B), the encoder's output for the real
background video. Two reasons it may fail:
- **Attention bleed.** At the last layer, attention has already spread the subject into every token, so S is not sparse.
  This is why earlier or higher-resolution stages (MViT-V2 stage 1–2, or SAM2 stride 8/4, both more local) may work better.
  The subtraction then happens mid-network, which is a variant of FGBG and not FGBG itself.
- **Few time steps.** FGBG's input gives 8 temporal token slices (16 frames, temporal patch 2). Low-rank estimation from
  8 rows needs the subject to be absent at each location in most of them.

### 3.3 Integral image in feature space (needs a warp)
Replace `enc(C)` with an aligned, gated, recency-weighted average of **cached** features:

  F̄^ℓ_p = Σ_t w_t G_{t,p} F^ℓ_t(p + Δ^ℓ_t) / Σ_t w_t G_{t,p}

Here Δ^ℓ_t is the tracker shift scaled to stride ℓ, with bilinear sampling for sub-cell shifts. F̄ is **not** meant to equal
enc(C), since the encoder is non-linear. It only has to give the (retrained) adapter the same information. It needs three
things:

1. **Shift equivariance:** E(shift_k x) ≈ shift_{k/ℓ} E(x). Hiera's absolute position embeddings and global attention
   blocks partly break it. Strides 4/8 come from earlier, more local stages, so they should behave better.
2. **Locality:** foliage next to a visible patch leaks into that patch's feature. This is the main risk.
3. **Majority visibility** per location (a median breaks down at 50% occlusion). The gate is what handles this, as in
   pixel space.

### 3.4 Hard limit, shared by every single-clip method
A subject that never moves is absorbed into L/B. In PanAf500, sitting is the most frequent behaviour (76k of about 200k
boxes). The pixel median has the same limit. A full fix needs other clips from the same camera; FGBG ships camera IDs.

## 4. Experiments, in order, each with a kill criterion (pre-register the thresholds before running)

**E0. Shift equivariance of SAM2.1 b+ features.** No training, minutes.
- Measure ‖E(shift_k x) − shift E(x)‖ / ‖E(x)‖ at strides 16/8/4, for integer-cell and sub-cell shifts, on NFO frames.
- Also measure locality: change the image outside a patch and see how much that patch's feature changes.
- **Kill (B)** if the relative error at stride 16 is comparable to the feature difference between person and foliage cells.

**E1. FGBG background-estimate oracle.** No training. Needs the FGBG data, whose availability is unknown.
- Take about 30 pairs and a frozen Kinetics MViT-V2.
- Compare, against the real z_B: (a) pool(L) from RPCA/median on the foreground tokens, (b) the do-nothing baseline z_F.
  Measure cosine distance, at the last layer and at 1–2 earlier stages.
- **Kill (A)** if (a) is not clearly closer to z_B than (b) at any layer.

**E2. FGBG training.** Only if E1 passes.
- One recipe (FGBG's MViT-V2 settings, disjoint split), varying only the source of ẑ_B: real B (paper), latent
  median/RPCA, none.
- Success: recover ≥ X% (pre-register, e.g. 75%) of the paper's disjoint mAP gain (+3.75 to +3.96 for MViT-V2) with a
  single pass.
- Compute warning: FGBG trained on 4×H200 with batch 336. Before full fine-tuning, try a frozen-backbone attentive probe.

**E3. NFO latent integral.** Only if E0 passes.
- Retrain the adapter with C16/8/4 := F̄^{16/8/4} and B := latent median/RPCA. Same seeds and protocol as the headline
  (seq2–4, 24 whole crossings, full-pass), all in one environment (cross-environment mask drift is about 5.5%).
- Success: within the seed spread of the current adapter on box AUC, hand J and AP50, with one encoder call per frame.
  Time both with `tools/timing.py`.

## 5. Must-do before building
- **Novelty pass (15–30 min)** on: deep or latent robust PCA and background subtraction in CNN/ViT feature space (e.g.
  "deep RPCA", "background modelling in feature space"), single-pass background-bias removal in action recognition, and
  feature-space multi-frame fusion. None of this has been searched yet. The PanAf session's literature pass covered SAM2
  occlusion and behaviour recognition only (`../PanAf20K/docs/lit_pass_2026-11-11_sam2_occlusion_behaviour.md`).
- **Expert pipeline review** (pre-authorised by the user) once the design is settled, before writing model or training code.
- Do not reopen the SAM2 memory variants (closed in this repo's CLAUDE.md).

## 6. Relation to the PanAf thread
The PanAf session tests whether the NFO adapter generalises to PanAf500. A GT box prompt is given on each ape's first
frame; the comparison is plain SAM2 vs. SAM2 + adapter, and the decisive test is tracking recall on about 30 hand-labelled
occluded test videos. That test needs an empty-scene estimate for camera-trap clips, where apes are often present from the
first frames, so the warm-up median is invalid. A clip-level median/RPCA B (§3.1, pixel or latent) unblocks it. **This is
the first concrete consumer of §3.1.**

## 7. Follow-up: super-resolved subject across frames and videos
- Generalise the integral image from "average of aligned frames" to multi-frame reconstruction with sub-pixel alignment.
  Within a clip, this is burst / video super-resolution: deformable feature alignment (BasicVSR, Chan et al., CVPR 2021)
  or deep burst SR (Bhat et al., CVPR 2021).
- Across videos, this needs the *same individual*. PanAf `ape_id` is intra-video only, so cross-video fusion requires
  individual re-identification first. It is a separate problem.
- Aligning a non-rigid ape is much harder than aligning NFO's mostly translating people. Expect a learned alignment rather
  than a Theil–Sen shift.
- Only worth starting once E3 shows the latent integral is lossless.
