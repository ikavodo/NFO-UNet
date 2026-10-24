# Plan: integral image/background fusion adapter → dense prompt for a frozen, memory-free SAM2 (2026-10-16)

Status: **live**, not started. Written for an implementing session. Data generation is out of scope: §3 lists only
what the data must provide; another session builds it.

Read first: `docs/vos_benchmark/archive/nfo_vos_pilot_findings_full.md` (last four sections) and
`docs/vos_benchmark/nfo_failure_log.md` (2026-10-16 lines).

## 0. Why (measured, 2026-10-16, NFO seq1, 11 windows with integral image history)

- **Integral image as SAMURAI init fails.** Two variants were tried:
  - planted prepend frame: ΔJ −0.073 [−0.170, −0.004];
  - integral image mask inserted as the t₀ memory: ΔF2 −0.001 [−0.072, +0.050].
  - Even the oracle t₀ mask (SAM2-ceiling) gives only ΔJ +0.036.
- **Why (`cp_vbar.show_logits`).** Under the box + p* prompt, SAM2 segments 3,600–6,900 px *in the empty scene*. This
  is the box's prompt prior latching onto background structure. A median 53% (up to 86%) of SAM2's mask on the
  integral image is that same background object. Multiplying the two (σ(l_C)·(1 − σ(l_B))) cancels the leak only when
  SAM2 picks the same object in both images, so it is not a general cleaner.
- **What works is per-frame and per-pixel against the empty scene.** V̂_anch raises recall in all 61 windows on
  both bases: ΔR +0.019 … +0.080 vs the silhouette label.
- **The tracker's point, not a box, is the right sparse prompt (user, 2026-10-16).** A wrongly tracked integral image
  then cannot drag the mask, because the point anchors it to the tracker's position at t.

**Hypothesis.** A small trained adapter that *fuses the integral image's and the background's SAM2 features* can produce
the dense mask prompt that a hand rule could not. That prompt should say where the person is (from the integral image),
where background structure is (from B, the cause of the leak), and how far to trust the integral image. A frozen,
memory-free SAM2 decoder, given that prompt plus the tracker point, then segments each frame. No memory attention
is involved.

## 1. Task

Per frame t (GT-free at test time), the inputs are:
- the frame I_t;
- the tracker point p_t (the blob tracker's merged centre, as in V̂);
- the integral image C_t (V̂'s GT-free weighted-temporal integral image of t−6 … t−1, aligned onto t by the tracker's
  Theil–Sen motion; `factorised_gtfree.composite`);
- the empty scene B (warm-up median).

Output: the modal mask V̂_t of the person in I_t.

SAM2.1 b+ is frozen throughout: image encoder, prompt encoder and mask decoder. Only the adapter is trained.

## 2. Mechanism

**Features.** F_X^ℓ = Enc(X) at level ℓ ∈ {16, 8, 4}, for X ∈ {C_t, B}:
- stride 16: `image_embed`, 256 ch;
- strides 8 / 4: `high_res_feats`, 64 / 32 ch;
- source: `sam2_image_predictor.py:127`.

Enc(B) is computed once per segment. Enc(C_t) costs one extra encoder pass per frame (0.074 s measured on NFO), the
same cost as V̂.

**Remove the integral image's background noise at the source (added 2026-10-16).**
- `composite_frames` (`factorised_gtfree.py:62-71`) normalises the person gate G per pixel: Σ w G A / Σ w G. Wherever
  G is small but non-zero, the result is still a weighted average of the *shifted* history. So the integral image's
  background is the scene blurred along the person's motion, and F_C − F_B ≠ 0 there even though no person is
  present.
- **Default input: background-consistent integral image.**
  C̃ = a·C + (1 − a)·B, with a = clip(Σ_i w_i G_i, 0, 1), the gate mass, i.e. the evidence that something
  non-static was seen at that pixel. Outside the person, a → 0, so C̃ → B and F_C̃ − F_B → 0 by construction. It costs
  nothing.
- **Ablation: null integral image.** B̃ = Σ_i w_i shift(B, Δ_i), the same alignment applied to the empty scene; then use
  F_C − F_B̃. Anything B̃ produces is alignment-induced and not person (a control variate). It costs one more encoder
  pass per frame.
- What remains is gait blur and alignment error on the person. These are learned away by the targets (§4), not
  removed by hand.

**SAM2's own objectness as input channels (added 2026-10-16).** The frozen SAM2 decoder is run with the point p_t
alone on C̃ and on B (no grad). This gives logits l_C and l_B at stride 4, which are concatenated into the adapter at
stride 4. These are the two ingredients of the hand rule σ(l_C)·(1 − σ(l_B)), which cancelled leaks only when SAM2
picked the same background object in both images. The adapter learns their combination instead.
- Cost: two decoder calls (0.047 s each).
- l_B also depends on p_t, so it is per frame.

**Fusion per level (Siamese difference, the change-detection form).**

  z^ℓ = [ P^ℓ F_C^ℓ ,  P^ℓ F_B^ℓ ,  P^ℓ (F_C^ℓ − F_B^ℓ) ,  cos(F_C^ℓ, F_B^ℓ) ]

Here P^ℓ is a 1×1 projection to d = 32, with one shared P^ℓ for all three terms (weight-tied Siamese).
- Ancestor: Fully Convolutional Siamese Networks for Change Detection (Daudt, Le Saux, Boulch, ICIP 2018,
  arXiv 1810.08462). Its concatenation and difference variants are the two fusion rows of the ablation (§6).
- Why the difference is given explicitly (derived, `training/plan_feature_layers_training.md` §9): with
  F = F_bg + Δ, the person evidence a template can carry is Δ_p = F_C − F_B. It is the matched-filter template.
  On the synthetic clips no *hand-made* template beat the energy detector, because template error ‖e‖² was large.
  A learned fusion can down-weight unreliable template parts. That is the bet.

**Conditioning (the bad-integral image mitigation).**
- **Point map.** h_t is a Gaussian at p_t (σ = 1 stride-16 cell), concatenated at stride 16. With it the adapter can
  test whether the integral image's person lies where the tracker is now. A integral image built from the wrong history
  (C ≈ B) or the wrong motion does not.
- **Reliability scalars** r, injected by FiLM (Perez et al., AAAI 2018, arXiv 1709.07871; id not re-verified this
  session): number of tracked history frames out of 6, and the Theil–Sen residual of the tracker chain. These are
  exactly the quantities whose failure broke the prepend.

**Decoder of the adapter.**
- Top-down path s16 → ×2 + s8 → ×2 + s4, each step a 3×3 conv, GroupNorm and GELU, width 64.
- Two heads at stride 4 (256 × 256):
  - prompt logits m;
  - gate g ∈ [0, 1].
- Expected size ≈ 0.3 M parameters (assumed; count and report).

**Injection into the frozen SAM2 (no architecture change, no memory).**

  dense = g ⊙ ↓(mask_downscaling(m)) + (1 − ↓g) ⊙ no_mask_embed

- `prompt_encoder.py:57` is `mask_downscaling`, `:66` is `no_mask_embed`, `:114-116` is `_embed_masks`.
- ↓ = average-pool to 64 × 64.
- The dense embedding is added to the image embedding at `mask_decoder.py:205` (`src = src + dense_prompt_embeddings`).
- Sparse prompt: ~~the point p_t only~~ **box + point** (revised after O1, §10: a dense prompt alone is weaker than a
  box, even when it is GT).
- **Zero init (revised 2026-10-16 after a unit check).** A sigmoid gate at bias −4 (g = 0.018) already moved the
  decoder logits by up to 18.5, because mask_downscaling(m) ≫ no_mask_embed. The implementation therefore uses
  dense = dense₀ + α·↓g·(mask_downscaling(m) − dense₀), with a learnable scalar α = 0 at init (the ControlNet
  zero-init trick; Zhang et al., ICCV 2023, arXiv 2302.05543).
  - At step 0 the output is exactly box + point SAM2 (measured: max |Δlogit| 0.0), and α receives gradient.
  - The public `mask_input` API is not used: a zero mask ≠ no mask.

**What the dense prompt can and cannot do (derived).**
- It enters at stride 16 (the downscaler ends at 64 × 64). So it is an **extent / position / which-object prior**,
  for initialisation and leak suppression.
- Sub-cell visible fragments (H2) must still come from the decoder's high-res path, `mask_decoder.py:224-225`, driven
  by F_t, which the adapter does not alter.
- The stride-16 oracle visibility AUC on sub-cell fragments was 0.612. So expect gains on wrong-object and leak
  frames, not on fragment recall.
- O1 below measures this before any training.

**Prior work (verified 2026-10-16).**
- AutoSAM (Shaharabany, Dahan, Giryes, Wolf, BMVC 2023, arXiv 2306.06370):
  - What it does: a trained network replaces SAM's prompt encoder, takes the image as input, and is trained by
    gradients through frozen SAM (BCE + Dice). This is the training scheme used here.
  - Difference: our adapter conditions on a *integral image/background pair from the same video* and keeps the point.
- PerSAM (Zhang et al., ICLR 2024, arXiv 2305.03048): a training-free one-shot reference prior for SAM (location prior
  and target-guided attention).
  - Difference: our reference is the tracker-aligned integral image of the same object, plus an explicit
    background-negative cue.
- No novelty pass has been run (deferred by the user). Do not claim novelty before one.

## 3. Data interface (what the data session must deliver; not how)

Per clip (GPJATK walkers, foliage occluder ≡ background, the `data/synthetic_nfo_design.md` setting), and per frame t:
- I_t, grayscale (NFO is grayscale);
- B, the empty scene, built by the NFO protocol (warm-up pixel median);
- S_t (full silhouette), V_t (modal visible), O_t (observable in ≥ 1 aligned history frame);
- GT box;
- the tracker outputs used at test time: p_t, the chain, and the shifts. Simulated or real, but **from the same
  integral image code path** (`factorised_gtfree.composite` with tracker-derived shifts), so that train and test
  integral images match.

Required corruption modes, each flagged per frame, at a total rate of about 25%:
- (a) shift error beyond the tracker's measured error;
- (b) history reaching before the person enters, so C ≈ B;
- (c) p_t on a non-person blob.

Splits are by subject. Val is for model selection; test is touched once.

## 4. Losses

- **Main:** focal + Dice on the decoder's output vs V_t, at SAM's 20 : 1 ratio.
- **Auxiliary:** 0.1 · BCE of m vs S_t ∧ O_t, so the prompt stays an interpretable observable-extent map. This is
  not amodal hallucination, because O_t excludes parts never seen.
- **Recall-weighted variant (ablation, pre-registered):** Tversky in place of Dice, α = 0.3, β = 0.7 (Salehi et al.,
  MLMI 2017, arXiv 1706.05721; id not re-verified this session).

## 4b. Training procedure

**One sample** is one frame t of a synthetic clip, with its 6-frame history. It is built exactly as at test time:
- C̃_t from the history, with shifts = GT shift + tracker-like error (calibrate to the NFO hybrid tracker: median
  7.7 native px vs GT shifts, rescaled by person height);
- B = the warm-up pixel median;
- p_t = an interior GT person point plus jitter, or a corrupted point (§3 mode c);
- targets V_t, S_t, O_t.

**Forward pass, three stages:**
1. Frozen SAM2, no grad:
   - Enc(I_t), Enc(C̃_t), Enc(B) (Enc(B) cacheable per clip);
   - decoder(point) on C̃_t and on B → l_C, l_B.
2. Adapter, trained: z per level → top-down path → m, g.
3. Frozen SAM2 decoder, with gradients flowing through its activations to the adapter but no parameter updates:
   - inputs: Enc(I_t) (`image_embed` + `high_res_feats`), sparse = point p_t, dense = gated mix (§2);
   - output: mask logits → loss vs V_t (+ the auxiliary loss on m).
   - Run with `multimask_output=False`; SAM2's IoU head is not trained.

**Optimiser:**
- AdamW, lr 1e-4, weight decay 1e-4, cosine schedule;
- batch 16 frames from ≥ 8 clips;
- frozen SAM2 in bf16, adapter in fp32;
- about 20 k iterations; early-stop on the val G2 metric;
- 3 seeds.
- These are starting values (assumed). Record wall-clock and peak memory.

**Caching.** Enc(I_t) and Enc(B) do not depend on the sampled jitter, so cache them per frame. Enc(C̃_t) does depend
on it: either encode online, or pre-sample 4 jitter draws per frame and cache those.

## 5. Gates (pre-registered; one end-to-end oracle first)

**O1, no training, the decisive kill test.** Run on 30 fixed val clips: frozen SAM2, point p_t, with dense prompts
built from GT (m = ±10 logits, g = 1):
- (i) S_t (extent);
- (ii) S_t ∧ O_t;
- (iii) V_t.

Compare against point-only, box + point, and the hand rule σ(l_C)·(1 − σ(l_B)) as the prompt. Report J, R and P on
V_t, overall and on the corruption strata.
- **Kill:** if the best GT prompt (i)/(ii) gains ≤ +0.02 J over point-only, the dense-prompt channel cannot carry
  this information. Stop and report.
- Prompt (iii) bounds the channel: modal truth injected at stride 16.

**G2, trained adapter on val:**
- ΔJ vs point-only > 0, with a paired CI excluding 0;
- at least 50% of the O1 (ii) gap;
- no harm on corrupted frames: ΔJ ≥ −0.01.

**G3, NFO zero-shot** (no NFO training), all seq1–4 window frames:
- references: the frozen silhouette label and the hand labels (seq1 polygons/brush, seq2–4 handAv with void
  ignored);
- arms: point-only SAM2 per frame, B0, T1, V̂_anch;
- report J, P, R, F2;
- the target is recall at matched precision vs T1.

## 6. Ablations (one variable per row, 3 seeds, same data and budget)

- fusion: Siamese difference (default) vs concatenation only vs + F_t as a third input;
- levels: s16 only vs s16 + s8 + s4;
- point map h_t off;
- reliability FiLM off;
- gate off (m through `mask_input` directly);
- Dice vs Tversky.

## 7. Cost per frame (NFO, measured components)

- frame encoder 0.074 s (needed anyway);
- integral image 0.081 s;
- Enc(C_t) 0.074 s;
- adapter (not measured; expected ms);
- decoder 0.047 s;
- Enc(B) once per segment.

That is a few ×0.1 s per frame, against about 1 s/frame plus depth for Diffusion-VAS.

## 8. Risks

- **The prompt overrides the image.** The extent mask at t₀ cost 0.09 precision when inserted into memory.
  Mitigations: the gate and the modal main loss; watch P on G2.
- **One foliage photo.** Hold out photo crops by region; if val ≫ test, add PercepTreeV1 images.
- **Domain gap from GPJATK to NFO** (person scale, sensor). G3 is the only arbiter; no NFO tuning.
- **Partial circularity on NFO.** The silhouette label and B share the empty-scene cue; the hand labels are the
  referee.

## 9. Steps for the implementing session (each gated on the previous)

1. A loader that yields §3 fields from the data session's output; unit-check V ⊆ S and O ⊆ S.
2. **O1** (`layers/prompt_oracle.py`): direct prompt-encoder / decoder calls with the gate; montage traces; stop if
   killed.
3. The adapter plus a training loop (frozen SAM2 in fp16/bf16, adapter fp32), with logged parameter count and
   wall-clock.
4. G2 on val, with ablations only after G2 passes.
5. G3 on NFO; append the results to `archive/nfo_vos_pilot_findings_full.md` and the failures to `nfo_failure_log.md`.

Working rules: code and .md only in git; commit messages end with the Co-Authored-By line; save visual traces in the
repo and let the user judge them; ask before dispatching subagents.

## 10. O1 result (measured 2026-10-16, `layers/prompt_oracle.py`, 30 val clips, flutter, 330 frames)

Frozen SAM2.1 b+ image predictor, memory-free. Point = centre of the jittered GT box (σ_a 3 px); box = jittered GT box.
J / P / R vs V_t, mean per clip; ΔJ vs point-only [clip bootstrap]; J on the bad-init stratum (J(pt) < 0.5, n 284 frames).

| prompt | J | P | R | ΔJ vs pt | bad-init J |
|---|---|---|---|---|---|
| point only | 0.277 | 0.578 | 0.365 | — | 0.181 |
| box + point | 0.533 | 0.564 | 0.900 | +0.256 [+0.216, +0.297] | 0.500 |
| hand rule on C, dense only | 0.242 | 0.556 | 0.312 | −0.035 | 0.186 |
| hand rule on C̃, dense only | 0.226 | 0.548 | 0.290 | −0.051 | 0.164 |
| GT S (extent), dense only | 0.397 | 0.633 | 0.518 | +0.120 | 0.345 |
| GT S ∧ O, dense only | 0.410 | 0.670 | 0.511 | +0.133 [+0.108, +0.159] | 0.355 |
| GT V (modal), dense only | 0.442 | 0.730 | 0.516 | +0.166 | 0.382 |
| box + hand rule on C̃ | 0.471 | 0.520 | 0.798 | +0.194 | 0.435 |
| **box + GT S ∧ O** | **0.621** | 0.643 | 0.949 | +0.344 | **0.600** |
| box + GT V | 0.718 | 0.758 | 0.937 | +0.441 | 0.699 |

- **Pre-registered rule: PASS** (best GT extent prompt +0.133 over point-only).
- **The comparator in the rule was too weak.** A dense prompt alone gives less than a box: even GT V alone (0.442) <
  box + point (0.533). SAM2 weights its mask channel lightly without sparse prompts, which is consistent with SAM's
  training, where the mask input is its own previous low-res logits plus clicks.
- **On top of the box the channel carries the information.** Box + GT S ∧ O (the adapter's auxiliary target, no
  hallucination) gives +0.088 J over box + point, and +0.100 on the bad-init stratum (0.500 → 0.600). Box + GT V
  bounds the channel at +0.185.
- **The hand fusion hurts** (box + hand rule −0.062 vs box), so learning is needed. Background-consistency (C̃) does not
  help the hand rule (C 0.242 vs C̃ 0.226).
- **Plan changes (from this result):**
  - sparse prompt = tracker box + point, not the point alone; the gate at 0 is then exactly box + point SAM2;
  - G2 control = box + point;
  - G2 bar = ≥ 50% of the box + GT S ∧ O gap, i.e. ΔJ ≥ +0.044 with a paired CI excluding 0.
- Trace: `images/benchmark/gpjatk_foliage/prompt_oracle.png` (bad-init frames: X_t, C, C̃, hand prompt, then
  point / hand / GT S ∧ O outputs; TP green, FP red, FN blue).

## 11. Approach A: residual feature fusion (approved 2026-10-16; supersedes the dense-prompt-only adapter, which becomes ablation C)

**Why.** A dense prompt is only an additive term on the stride-16 embedding (`mask_decoder.py:205`). The general form
is a zero-initialised residual on all three levels the decoder reads, F̂_t^ℓ = F_t^ℓ + α_ℓ·R^ℓ(F_t^ℓ, F_C^ℓ, F_B^ℓ),
ℓ ∈ {16, 8, 4}. Strides 8 and 4 reach the upsampling path (`mask_decoder.py:224-225`), where the oracle separates
sub-cell visibility (AUC 0.94–0.97 vs 0.61 at s16). The frame's own features F_t are an input, because visibility is the
comparison F_t ↔ F_C vs F_t ↔ F_B. Dropped as inputs: σ(l_C), σ(l_B) (leak-prone SAM2 outputs, redundant with F_C /
F_B, two decoder calls), and the point and box maps.

**Sparse token.** The frozen decoder needs ≥ 1 sparse token (O1: dense GT V alone 0.442 < box 0.533).
- Default: one learned prototype token, MLP(Σ a·F_C^16 / Σ a). No hand prompt; the tracker is used for alignment only.
- Ablation: the tracker box (jittered GT box), with exact box-only SAM2 at init (α = 0).

**Data.** Pixel integral image C + evidence mass a (V̂'s construction, test-time consistent); feature-space integral image is
approach B, later. Occlusion sampler per clip: ρ ~ U[0.2, 0.6], clump ∈ {12, 24, 12 ∪ 24}, flutter ~ U[0, 2]·NFO
calibration; the corruption modes kept. Per-clip batches: every target frame of a clip shares one empty-scene encode.

**Training.** Frozen SAM2.1 b+, bf16. AdamW 3e-4, weight decay 1e-4, cosine, 1,500 steps × 2 clips (≈ 24 frames),
validation every 250 steps on the 30 fixed val clips (occlusion drawn per clip, deterministic). Loss 20·focal + Dice vs
V. Controls on the same frames: box-only and box + point SAM2. Bar (G2): ΔJ vs box + point ≥ +0.044, CI excluding 0.
Report by mode, density bin and clump type.

**Tasks.** (1) data: occlusion sampler + per-clip batches, tests; (2) model: residual fusion + token, tests for shapes
and exact box-only equivalence at init; (3) training/eval loop; smoke run; (4) full run token default; (5) box ablation.

## 12. Review of approach A (2026-10-16) and what changed

1. **Target consistency.** Training target = synthetic V (exact, modal). Real-frame reference = the user's modal seq2–4
   handAv labels, void ignored (`layers/fusion_nfo.py`, every validation). The extent-leaning silhouette label and the
   seq1 polygons are *not* used to judge the adapter.
2. **Encoded integral image out of distribution.** `fusion_nfo.feature_diff_montage` saves ‖F_t − F_C‖ and ‖F_t − F_B‖ per
   level on NFO frames (`images/benchmark/gpjatk_foliage/fusion_nfo_featdiff.png`). Ablation queued: F_C removed, a only.
3. **Token without evidence.** token = w·(point embedding at the a-centroid + zero-init MLP correction) + (1 − w)·learned
   fallback, w = 1 − exp(−Σa/κ), κ learned (init 500 cells). Test: empty a gives exactly the fallback.
4. **Prompt-space projection.** The token starts as a prompt-encoder output (point embedding), and the MLP correction is
   the learned projection; the fallback starts as the centre-point embedding.
5. **Mask selection.** multimask_output = False, with SAM2's own stability fallback (`build_sam.py:85`,
   `mask_decoder.py:149-150`). The decoder stays in eval mode, so the rule is identical in training and validation, for
   all arms.
6. **Zero init.** Scalar α replaced by zero-initialised final 1×1 convs (ControlNet zero conv). Test: features unchanged
   at init, and the zero convs receive gradient.
7. **Cost.** F_B is encoded once per clip in training and once per segment at test (static camera). Test-time cost:
   2 encoder passes per frame (frame + integral image). A moving camera would need F_B aligned: out of scope.
8. **Scaling of a.** a = Σ w G ∈ [0, 1] by construction (w sums to 1, G ∈ [0, 1]), with a fixed 6-frame history;
   resampled by area to 256² and avg-pooled to 128² / 64². Its dependence on foliage density is signal, not scale drift.
9. **Synthetic-only training.** 10% unoccluded clips (ρ = 0). Real-frame validation from step 0 (item 1).
   **Person scale:** s ~ U[0.5, 1] zoom-out per clip (GPJATK person 0.53 of frame height, NFO 0.33 measured); the border
   is the reflected empty scene.
10. **Ablations (after the main run, one variable each, same budget):** −F_B; −F_C (a only); −difference terms; token
    vs box vs token + box; stride 16 only vs all three.

## 13. Closing the synthetic → NFO gap (2026-10-16), pre-registered before the v3 run

**What failed so far (measured):**
- v1, old data: synthetic val J 0.75; NFO seq2–4 adapter J 0.575, R 0.59 (vs GT-box SAM2 0.722).
- v2, blob tracker in training: seq1 selection adapter J 0.326, R 0.34 (vs box + point 0.566).
- Diagnosis of v2:
  - the blob tracker covered only 48% of synthetic frames, so 35% of training targets had no integral image (a = 0);
    on NFO 0% do;
  - NFO evidence mass is about 4× lower (median Σa 6,265 vs 23,293 cells).

**v3 data path (fusion_data.build):** the stride-16 kernel tracker in training and on NFO (fusion_nfo.items), one
alignment code path.
- Measured on synthetic: centre error 0.092 person heights, 0% lost frames at every density up to 0.6.
  On NFO: 0.105.
- Whole-clip batches.
- Backgrounds: CanaTree100 50/50 with the template, every 5th image held out for val.
- Integral image parameters randomised per clip: n_h 4–8, τ 2–6, z0 2–4, widening the evidence-mass range.

**Sanity checks, printed at every validation (every 100 steps):**
1. **Tracker parity.** Kernel-tracker error on synthetic vs NFO (above). If synthetic is much easier, the adapter
   learns to trust a integral image that NFO cannot deliver.
2. **Input parity.** Median Σa and mean a on the labelled person, synthetic val vs seq1. Large mismatch means
   out-of-distribution inputs at test.
3. **Integral image use.** The `adapter-noC` arm (C := B, a := 0) on seq1. If it scores like `adapter`, the integral image is
   ignored and any gain is not our idea.
4. **GT-free control.** `trk-box`: SAM2 with the kernel tracker's box, the fair comparison for the box-free token
   adapter. GT-box SAM2 remains the oracle reference.
5. **Gap in the quantity that matters.** GAP = (adapter − trk-box) on synthetic vs on seq1. The domain gap is
   closed when the real-frame advantage approaches the synthetic one.

**Pre-registered "gap closed" (TEST, seq2–4 handAv, touched once, checkpoint chosen on seq1):**
- adapter J − trk-box SAM2 J > 0, with a frame-bootstrap CI excluding 0;
- adapter R ≥ trk-box R − 0.02;
- `adapter-noC` J < adapter J − 0.02, i.e. the integral image matters.
