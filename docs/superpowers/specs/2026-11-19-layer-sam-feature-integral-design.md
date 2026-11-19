# Layer-SAM with a learned feature-space integral image (design, 2026-11-19, revision 3)

Branch `exp/layer-sam-feat` (from `exp/ape-stage1`).

Status: **design, awaiting approval. No code yet.** Revision 2 folds in the pre-implementation expert review (§9); revision 3 the user's answers of 2026-11-19 (no hand-set logit priors, stage 0, bilinear shift, 40 ms, background track).

## 1. Intent

Today's adapter (`adapter/inference.py::masks_from_arrays`) runs the SAM2.1-b+ image encoder **twice per frame**:
- once on frame t;
- once on the pixel integral image C (`Cf = enc(C)`, inference.py:125).

That measures 61 ms/frame, against 53 ms for SAM2 video (CLAUDE.md). The goal is a variant that:

1. **Uses one encoder pass per frame.** C is built from **cached encoder features**: a ring buffer of F16/F8/F4 for the
   last n_h frames. There is no SAM2 memory encoder and no mask in the buffer.
2. **Learns the aggregation weights.** They start from a feature-space version of today's recency × change-gate
   weighting, so step 0 reproduces today's construction in feature space.
3. **Has an intermediate loss** that makes the integral meaningful: Ĉ should approach the encoder features of frame t
   *without the occluder*, on the hidden-but-recoverable person cells.
4. **Leaves the rest unchanged:**
   - the Fusion adapter, its zero-init residuals, the token correction;
   - loss v3.4, step-200 selection on seq1;
   - the tracker.

The motivations are "learn, don't hand-set" (memory `learn-not-hand-set`) and real-time Layer-SAM. A learned
background is needed for apes, where PanAf clips have no empty warm-up; §5 has the open decision on it.

## 2. Goals, non-goals, anti-goals

- **G-speed.**
  - Must: below SAM2 video (53 ms/frame) at batch 1 on the laptop RTX PRO 4000 (`tools/timing.py`).
  - Target: real time at NFO's 25 fps, i.e. ≤ 40 ms (user, 2026-11-19).
  - Derived expectation is 34–37 ms. The arithmetic: 61 = 2 × 24.8 (encoder) + 3.2 (adapter) + 2.1 (decoder) + ≈ 6 ms
    of per-frame overhead (tracker cosine map at native size, gate, resize); removing one encoder pass and the pixel
    gate/shift leaves ≈ 36 ms.
  - First measure the floor: the `trk` arm (SAM2 + tracker prompts, no adapter). If the floor is > 30 ms, the 30-fps
    target needs overhead work (native-size upsampling), not this design.
- **G-acc:** no real loss against the current pixel adapter on the headline protocol: seq2–4, 24 whole crossings,
  full pass, 3 seeds. Decision rule in §7.
- **Non-goals:** SAM2 memory (closed); changing the tracker or the Fusion architecture; the on/off switch. The switch
  reads pixel-gate statistics (`dual_expert.py:243`) and is refitted only if this arm wins.
- **Anti-goals:**
  - Any tuning or selection on seq2–4.
  - Cross-environment comparisons. Masks differ by ~5.5% of pixels between laptop and cluster, so all arms are dumped
    and scored in one environment.

## 3. Why the measured negative does not transfer directly (user, 2026-11-19)

In the earlier GPJATK test (2026-10-06; `master_thesis/results/gpjatk_em_viability/fusion_location/SUMMARY.md`),
feature fusion decoded worse than pixel fusion:
- amodal IoU 0.692 vs 0.777 with tracker motion;
- pixels won 22 of 23 targets.

There, the averaged features went **straight into the frozen mask decoder**, which put them off its input manifold.
Here only the adapter reads Ĉ. The decoder still sees F_t + R, R being the zero-initialised residual.

**Hypothesis H:** feature-space aggregation, learned and supervised by clean features, gives the adapter at least the
information that enc(C) gives it.

## 4. Construction

### Symbols
- F_k^s: encoder features of frame k at stride s ∈ {16, 8, 4}.
  - `F16/F8/F4` in `fusion_data.build`, already computed once per frame there.
  - F16 includes SAM2's constant `no_mem_embed` (inference.py, `set_image_batch`), so **every cosine below is taken
    on features centred per channel**.
- Δ_k: tracker shift of frame k relative to t, x-only Theil–Sen (`hist_warps`, `theil_at`).
  - In feature cells: Δ_k^s = Δ_k · 1024/W / s.
  - One stride-16 cell is 12.5 native px.

### Alignment
- **Bilinear sub-cell shift** of key k by Δ_k^s (`grid_sample`). This is the default.
  - The integer roll's only motivation was the 2026-11-13 rule "no interpolated features". That rule was about the
    decoder's input manifold, and by §3 it no longer applies, since only the adapter reads Ĉ.
  - The roll's rounding error is up to ±6.25 native px at stride 16.
- G0 also reports the roll, at no extra cost. A bilinear shift is a 2-tap low-pass in feature space, so it is not free.
- Box-relative RoPE is the fallback only if G0 fails.
- Key cells shifted in from outside the image are masked (logit −∞).

### Aggregator `Agg` (≈ 10–20k parameters; logits at stride 16 only). Revision 3: no hand-set logit terms
- **Keys:** frames t−n_h..t−1, shifted, plus a **null key** ∅ whose value is B^s.
  - The null key is a learned "abstain" slot, cf. DETR's no-object class (Carion et al., ECCV 2020, arXiv 2005.12872).
  - Without it, the softmax must take a weighted average of foliage at cells where no past frame saw the person.
- **Logit of key k at cell p:** ℓ_k = h([P F_k, P(F_k − F_t), P(F_k − B), e_age(k)]).
  - P is a 1×1 conv projection to 32 channels; e_age is a learned age embedding.
  - h is a 1×1 conv, GELU, a 3×3 conv and a 1×1 conv.
  - ℓ_∅ is a learned bias.
  - Recency and change-gating are *not* imposed. h sees age and F_k − B, so it can learn both.
- **Why no fixed prior** (user, 2026-11-19, "bitter lesson"). An additive prior plus an unbounded residual loses no
  expressivity, but it can lose optimisation:
  - A gate term log σ(z − θ) ≈ −5 to −10 on cells today's gate rejects scales the softmax gradient there by
    α_k(1 − α_k) ≈ e^{−5}–e^{−10} (derived). So h cannot reopen keys that the gate wrongly closed.
  - With checkpoint selection at step 200, h would barely move from the prior, so "learned" would go untested.
- **Weights:** α = softmax over keys ∪ {∅}, computed at stride 16 and bilinearly upsampled (in each key's aligned
  frame) to strides 8 and 4.
- **Output:** Ĉ^s = Σ_k α_k F_k^s + α_∅ B^s.
- **Evidence mass:** a := 1 − α_∅ at stride 4 (256 × 256, the shape `Fusion` reads; it feeds `z()` and the token
  prototype). It is the learned counterpart of today's Σ w G.
- **Stage 0 (replaces the hand-set initialisation).**
  - Pre-train Agg alone on L_C, with Fusion and the decoder absent. Only cached encoder features are needed.
  - A few hundred steps; stop when the val L_C plateaus.
  - Then train Fusion as today, with Agg fine-tuned jointly (its own parameter group).
  - Without stage 0, Fusion's first steps would see a uniform-average Ĉ, smeared by articulation. GPJATK: uniform-ish
    P_vis 0.674 vs P_recency 0.800 amodal IoU (reported, fusion_location/SUMMARY.md).
- **Fusion reads** C16/8/4 := Ĉ, B16/8/4 := B, a := 1 − α_∅ (fusion_adapter.py:74, :104), without other changes.
- **Mode 2** (5% of frames, `P_MODES`): today C := B. Here it becomes "all keys masked", so α_∅ = 1. This is the
  existing missing-input dropout, cf. ModDrop (Neverova et al., TPAMI 2016, arXiv 1501.00102).

### Clean-feature loss

  L = L_v3.4 + λ_C Σ_s mean_{p ∈ Ω_t^s} [1 − cos(Ĉ^s_p, E(X̃_t)^s_p)]

- **Target X̃_t:** frame t rendered with the **same sway and the same photometric noise** but without the occluder,
  using the layers of `make_layered_clip`.
  - With this target, the only gap between Ĉ and the target is the occluder: noise, gain and JPEG are not something
    a convex combination of keys can remove.
- **Ω_t^s:** cells with area coverage > 0.5 of O_t ∧ ¬V_t, no dilation. These are the person pixels hidden at t but
  visible in at least one aligned history frame.
- **Reported floor:** 1 − cos on the visible person cells.
- **λ_C = 1, fixed.** Agg gets its own parameter group. The reason λ is not tuned: Fusion's `out` is zero-initialised,
  so the mask loss sends no gradient into Agg at step 0. Early training of h is driven by L_C alone, and AdamW is
  scale-invariant per parameter, so λ cannot be calibrated by matching loss values.
- **Training cost per clip:** today's n encoder passes on C are replaced by n encoder passes on X̃_t. The memory peak
  stays near today's 9.8 GB, because logits are computed at stride 16 only.

## 5. Background B

**Decided 2026-11-19 (user):**
- The decisive arms use (a): the per-cell median of the encoded warm-up frames.
- The user also wants:
  - a **cold-start case**: the subject is present from frame 0 and there is no empty warm-up;
  - a background model that **learns robustly over time**.
  - "I don't care how background is modelled, but it should be intelligent enough."

**Background track**, run separately from the adapter arms and evaluated without adapter training:
- **Metric:** 1 − cos(B̂_t, E(B̃)) over time t, on cells the subject has crossed, in cold-start synthetic clips (no
  warm-up). The causal estimate B̂_t must not use future frames.
- **Candidates:**
  - **B-med** (zero parameters): a causal per-cell median over a sparse buffer of 16 encoded frames, with a
    conservative update (cells whose change score is high are not written). This is ViBe's update rule (Barnich & Van
    Droogenbroeck, IEEE TIP 2011) applied to features.
  - **B-rec** (learned): the same Agg module, unaligned and recurrent. Its keys are {B̂_{t−1}, sparse past frames}; its
    target is E(B̃) noise-matched. Trained on L_B alone, with features cached.
- **Not chosen: RPCA** (Candès et al., J. ACM 2011, arXiv 0912.3599). It is a batch method (non-causal over the window).
  For a moving subject its robustness is median-like, and online variants add an iterative solver per frame. It stays an
  offline analysis baseline only.
- **End-to-end check:** the winning B̂ replaces the warm-up median for the `feat-perlin` checkpoint, on NFO seq1 windows
  with the warm-up withheld.

The earlier options were (a) the warm-up median and (b) a learned B from frames that contain the subject. B-rec is (b)
made recurrent.

The review finding:
- On NFO, the keys for B are **empty** warm-up frames, all equally valid.
- A learned B there has no signal to learn from: it can only learn to average.
- It therefore cannot transfer to apes, where the clip itself contains the subject.

Options:
- **(a) NFO arm: B = per-cell median of the encoded warm-up frames (8 of the 20), not learned.**
  - Zero parameters, one variable fewer.
  - The learned B moves to the ape branch.
- **(b) Learned B from frames that contain the subject.**
  - Keys: 8 frames sampled from the clip window. Their features are already computed in training.
  - Same Agg module, unaligned, with a use embedding; target E(B rendered with the same noise).
  - Needs no warm-up, so it transfers to apes.
  - It is a second variable against the pixel control (whose B comes from the empty warm-up), so it needs its own
    ablation arm.

## 6. Gates before training (thresholds fixed before running; 30 synthetic clips)

**G0 — shift equivariance under subject motion.**
- Setup:
  - Paste the clean person at x and at x + Δ over a fixed B.
  - Δ is continuous in native px: {3, 6.25, 12.5, 25, 50}.
  - Compare E(at x + Δ) with shift_Δ(E(at x)), for both roll and bilinear, on person cells at each stride.
- Metric:
  - ε = 1 − cos, centred, reported as median and p90;
  - against δ = 1 − cos(F_person, F_B), the same cells against the empty scene.
- **Pass:** p90 ε_16 < ½ median δ_16, for the chosen shift method.
- Fail for both → the RoPE fallback (a design change, back to the user).

**G0 result (measured 2026-11-19, 30 val clips, `tools/shift_equivariance.py`, commit 81424e2): FAIL by the rule, only at
sub-cell shifts.** Columns: ε = equivariance error, ε0 = no-shift baseline, δ = person-vs-empty-scene contrast, all
1 − cos on person cells, centred.

| stride | Δ (s16 cells) | mode | median ε | p90 ε | median ε0 | median δ |
|---|---|---|---|---|---|---|
| 16 | 0.25 | bilinear | 0.157 | 0.339 | 0.167 | 0.919 |
| 16 | 0.5 | bilinear | 0.268 | **0.516** | 0.376 | 0.919 |
| 16 | 0.5 | roll | 0.376 | 0.700 | 0.376 | 0.919 |
| 16 | 1 / 2 / 4 | either | 0.037 / 0.034 / 0.043 | 0.137 / 0.155 / 0.169 | 0.686 / 0.849 / 0.930 | 0.92 |
| 8 | 0.25 (half an s8 cell) | bilinear | 0.350 | 0.811 | 0.425 | 0.637 |
| 4 | 0.25 (one s4 cell) | either | 0.018 | 0.083 | 0.335 | 0.573 |

- Reading:
  - The features are close to **equivariant for whole-cell shifts** at every stride: ε ≈ 4% of δ at s16.
  - A **half-cell phase offset** changes a person cell's s16 feature by ≈ 30% of the person-vs-background contrast
    (median), and by > 50% at p90.
  - Bilinear interpolation recovers only part of it (0.376 → 0.268).
  - This is the sub-stride aliasing known for CNNs/ViTs (Zhang, ICML 2019, arXiv 1904.11486; Azulay & Weiss, JMLR 2019,
    arXiv 1805.12177).
- **Not fixable by positional encoding.** The feature at the shifted phase is not in the cached frame, so the RoPE
  fallback cannot help.
- Consequence: about half of the history keys carry a phase error > ¼ cell at s16. G1 measures what that costs the
  integral, end to end at the representation level.

**G1 — representation, after stage 0 (the cheapest test of "learned beats hand-set").**
- Setup: on Ω (hidden recoverable cells), held-out synthetic clips, compare cos(·, E(X̃_t)) for:
  - Ĉ_learned, after stage 0;
  - Ĉ_fixed: recency τ = 3.5 × the feature-change gate, no learning;
  - enc(C), the pixel integral;
  - F_t, do nothing.
- **Kill** if Ĉ_learned ≤ F_t at stride 16.
- Ĉ_learned − Ĉ_fixed is reported, with a clip-bootstrap CI.
  - If it is ≤ 0, drop the learned part and run `feat-fixed` as the main arm.

**G1 result (measured 2026-11-19; synthetic val clips 0–29, 16 targets each; stage 0 = 400 steps, best at step 400,
4.7 s/it on the laptop; `adapter/feat_stage0.py`, commit after 81424e2): verdict LEARNED>FIXED.**

Mean cos(·, E(X̃_t)) on hidden-recoverable person cells, per clip then averaged:

| stride | learned (stage 0) | uniform (step 0, untrained) | fixed (recency × feature gate) | pixel enc(C) (today) | F_t (do nothing) | floor: F_t on visible cells |
|---|---|---|---|---|---|---|
| 16 | **0.493** | 0.446 | 0.290 | 0.425 | 0.319 | 0.709 |
| 8 | 0.502 | 0.504 | 0.446 | 0.410 | 0.355 | |
| 4 | 0.588 | 0.581 | 0.494 | 0.467 | 0.335 | |

Clip-bootstrap 95% CIs at s16:
- learned − pixel **+0.068 [+0.046, +0.090]**;
- learned − F_t +0.174 [+0.145, +0.202];
- learned − fixed +0.203 [+0.177, +0.229].

Val L_C fell 0.4896 → 0.4724.

Reading:
1. Even uniform averaging of the cached, shifted features (with the null key) is closer to the clean features on hidden
   cells than today's re-encoded pixel integral, at every stride.
2. Stage-0 learning adds +0.047 at s16 over uniform and ≈ 0 at s8/s4. The weights are computed at s16 and upsampled,
   so finer strides inherit them.
3. **The fixed control is weak.** The feature-space port of the adaptive gate abstains on most cells, so its Ĉ ≈ B,
   below F_t at s16. "Learned > fixed" therefore says little. The informative comparisons are learned vs uniform and
   learned vs pixel.
4. **Caveat (GPJATK 2026-10-06):** a higher representation cosine did not imply a better decode there. Only G2 (adapter
   training, seq1 masks) answers the actual question.

## 7. Arms and decisions

| arm | what changes vs `pix-perlin` | seeds |
|---|---|---|
| `pix-perlin` | none: existing checkpoints (control) | 3 (exist) |
| `feat-perlin` | feature integral, learned + L_C | seed 0 first; 3 if G2 passes |
| `feat-fixed` | feature integral Ĉ_fixed (recency τ = 3.5 × feature-change gate, no learning, λ_C = 0). Separates "feature space" from "learned" | seed 0 |
| `pix-mix` | occluder mix only | seed 0 first |
| `feat-mix` | both; the final model, only if `feat-perlin` and `pix-mix` both pass | 3 |

**Occluder mix.**
- Per clip, mode ∈ {perlin 0.5, stack 0.25, depth 0.25}. Twigs are excluded (they broke the visible-part labels,
  failure log 2026-10-15).
- 'depth' mode's ρ is a depth quantile, not a visible fraction. So ρ is resampled per mode until the **visible-fraction
  histogram matches perlin's**. Otherwise the mix also changes the occlusion level.
- Rationale for testing it at all:
  - Occlusion Copy & Paste (BMVC 2022) reports that variety beats realism (from a search summary).
  - Against it: depth occluders were measured as unlike NFO (T1 precision 0.555 vs 0.887, failure log 2026-10-15).

**G2, first training.**
- `feat-perlin` seed 0, 1000 steps, checkpoint 200, scored on seq1 only.
- Continue to 3 seeds if seq1 hand J and box AUC are ≥ `pix-perlin` seed 0 − 0.02.

**Headline (seq2–4, 3 seeds).** Paired window bootstrap of the seed-mean difference `feat-perlin` − `pix-perlin`.
**Pass** if the lower 95% bound is > −0.02 on **both** box AUC and hand J. Recognition conf and AP50 are reported.

## 8. Inference

- Ring buffer {F16, F8, F4} × n_h = 6 (fp16 ≈ 8 MB/frame, derived).
- Per frame:
  1. enc(x_t), shared by tracker, Agg, Fusion and decoder;
  2. push it to the buffer;
  3. Agg;
  4. Fusion;
  5. decoder.
- The tracker keeps enc(pixel median) once per window, ≈ 0.6 ms/frame amortised (derived).
- Frames with < n_h history use the tracker prompts, as today.
- **Parity test:** the training `build()` and inference produce identical Ĉ and a for the same clip.

## 9. Changes from the expert review (2026-11-19)

1. **Step-0 mass.** a = 1 − α_∅ without a gate prior was uniform at ≈ 0.71 everywhere: 1/(1 + Σ_{k=1..6} e^{−k/3.5})
   = 0.287 = α_∅ (derived). That made the token prototype a global mean. Fixed by the feature-gate prior.
2. Added `feat-fixed` to separate the representation from the learning.
3. **The target was unreachable.** It was noise-free and un-swayed while the keys are noisy and swayed, and Ω was
   dilated onto foliage. Now the target is noise-matched and Ω = hidden-recoverable cells.
4. **The learned B on empty warm-up frames had no learning signal.** Now §5, a user decision.
5. **G0 was redesigned:** subject motion over a static scene instead of a whole-image shift; continuous Δ; p90;
   roll vs bilinear. Reasons: Hiera's windowed attention (window 8 at stride 4, 14 at stride 16) and its absolute
   position embeddings, plus the 800 → 1024 rescale.
6. **Memory:** logits at stride 16 only.
7. **λ:** fixed at 1, with its own parameter group; cosines on centred features.
8. **Speed:** the expectation is corrected to 34–37 ms, and the `trk` floor is measured first.
9. **The switch:** not claimed compatible; refitted only if this arm wins.
10. **The decision rule** is a bootstrap lower bound on box AUC and hand J.
11. **Mix:** ρ matched by visible fraction; the cost is now stated as `feat-mix` being a third training.
12. **Minor fixes:**
    - `clip_raw` and `rescale_clip` must carry X̃ and the warm-up frames;
    - τ₀ is randomised as τ is today;
    - G1 reports Ĉ vs enc(C).

## 10. Left to the builder
- Module and file names: Agg in `adapter/feat_integral.py`, imported by `fusion_data.build` and `inference`.
- Batch layout of the keys (pad n_h to 8, mask).
- Sbatch flags: `FEAT=1`, `FEAT_FIXED=1`, `OCC=mix`. `remote_smoke` must cover the feature path.
