# One-pass adapter: feature-space composite with a learned gate (design, 2026-10-20)

**Status (2026-10-20): not pursued (user decision).** Kept as a record; the plan in `docs/superpowers/plans/2026-10-20-one-pass-adapter.md` is not executed.

## Intent

The current per-frame adapter (`layers/fusion_adapter.py`, "adapter-pf", v3.4 loss, step 200) beats SAMURAI on
tracking on NFO seq2–4:
- box AUC +0.072 ± 0.004 over 3 seeds;
- hand J and recognition on par.

For the paper, the user wants the same method with:
1. **one SAM2 image-encoder pass per frame** (today: frame + pixel composite, plus the empty scene);
2. **fewer hand-crafted inputs**: no explicit feature differences, no hand-made evidence mass (z-score gate, q90
   threshold, fixed τ).

The poster keeps the current adapter. This is paper work, built this week, trained on the cluster.

## Goals, non-goals, anti-goals

- **Goal:** a "one-pass" adapter that keeps at least **half** of the composite's measured gain on seq2–4 (user, 2026-10-20).
  The composite's gain is current adapter − no-composite arm: box AUC +0.018 [+0.013, +0.022], recognition conf
  +0.022 [+0.010, +0.033].
- **Goal:** runtime per frame ≈ one encoder pass + adapter + decoder.
  - Measured on the laptop RTX PRO 4000: encoder 24.8 ms, adapter 3.2 ms, decoder 2.1 ms (`benchmark/nfo_vos/timing.py`).
  - SAM2 video, for reference: 53 ms.
- **Non-goal:** changing the loss, the synthetic data generator, the tracker, or the prompt scheme.
- **Non-goal:** SAM2 memory, which is closed for this project (`nfo_failure_log.md`, 2026-10-20).
- **Anti-goal:** tuning anything on seq2–4. It is the single held-out test.
- **Anti-goal:** an apples-to-oranges comparison. Same training budget, seeds and evaluation path as the current adapter.

## Decision rule (pre-registered)

All on seq2–4 (46 windows), masks dumped on the cluster, mean over 3 seeds, recognition in one YOLO run (N = 7) with
all arms.

1. **Adopt for the paper** if both hold:
   - one-pass − **one-pass-noC** (same checkpoint, history := empty scene and m := 0 at inference), paired per seed and
     averaged over the 3 seeds: box AUC and recognition conf each ≥ **half** of the current adapter's composite gain,
     i.e. ≥ ½ · mean over seeds of (adapter − adapter-noC). Seed 0 measured +0.018 box AUC and +0.022 conf;
     adapter-noC is dumped for seeds 1 and 2 as well, so the threshold is a 3-seed mean fixed before any one-pass mask
     exists;
   - one-pass − current adapter: hand J ≥ −0.010.
2. **Otherwise** the paper reports that pixel integration beats feature integration under fragmented occlusion, keeps
   the two-pass adapter, and uses this comparison as the evidence.

## Constraints (fixed, so only the input design changes)

| thing | stays as | where |
|---|---|---|
| empty scene B | median of person-free frames, encoded once per sequence / clip | `fusion_data.build`, `video_adapter.per_frame_inputs` |
| feature tracker | `kernel_track` on 1 − cos(F16_t, F16_B); x-only Theil–Sen motion over [t−12, t] | `layers/track_check.py`, `factorised_gtfree.theil_at` |
| prompts | tracker box (t0 size) + tracker point; point token corrected by δ | `Fusion.forward` |
| output | zero-initialised 1×1 conv residuals added to F_X at strides 16/8/4 | `Fusion.out` |
| loss | v3.4: 20·focal_W(α 0.67) + Focal Tversky (FP 0.3, FN 0.7, γ 0.75), layer weights W | `fusion_adapter.loss_fn` (LOSS=v34) |
| data | GPJATK + foliage clips, `clip_raw` / `sample_occlusion`, mode 2 (5%: history = empty scene) | `fusion_data.py` |
| training | **STEPS=1000** planned (cosine T_max = 1000), AdamW 3e-4, 32 targets per clip, eval every 100 steps; **primary checkpoint step 200** | `fusion_adapter.main` |
| frozen SAM2 | SAM2.1-b+ image predictor; encoder, prompt encoder, mask decoder frozen | `run_t4.image_predictor` |

## What changes (decided)

### 1. Feature-space composite in place of the pixel composite

For target frame t, history k = 1…K, with **K = 6 fixed** in training and at test. Today training draws n_h ∈ 4..8 and
NFO uses 6; fixing K = 6 is *my call*, so the history tensor has a fixed shape.

- **Shift.** Δx_k = fx[t] − fx[t−k] (native px), from the same Theil–Sen fit used today. Each past feature map
  F^s_{t−k} (s ∈ {16, 8, 4}) is shifted horizontally by Δx_k · (1024/W) / s grid cells: bilinear, sub-cell, border
  replicate. No new encoder pass: the F_{t−k} already exist (training: `build` encodes every frame once; test: a ring
  buffer of the last 6 frames' features).
- **Order: gate first, then shift.** As in the pixel version (`prompt_oracle.composite_and_mass`), the gate is
  computed on the **unshifted** past map against the unshifted empty scene. The gate map and the feature maps are
  then shifted by Δx_k. Shifting first would compare misaligned background with F_B, and the gate would fire on the
  background.
- **Learned gate (replaces the z-score gate and a).**
  - One gate map per past frame, computed at stride 4 so that slivers narrower than 16 px survive:
    g_k = σ(φ([P⁴(F⁴_{t−k}), P⁴(F⁴_B)])). φ = 1×1 conv → GELU → 3×3 conv → 1 channel. Inputs concatenated, no explicit
    difference.
  - Average-pooled to strides 8 and 16.
- **Recency weights.** w_k = exp(−k/τ) / Σ_j exp(−j/τ), with τ learned (log-parameterised, init 3.5).
- **Composite and mass** per stride:
  Ĉ^s = (Σ_k w_k g^s_k P^s(F^s_{t−k}) + ε Σ_k w_k P^s(F^s_{t−k})) / (Σ_k w_k g^s_k + ε), with ε = 1e-3.
  The ε term is the pixel version's "plain mean where the mass is ≈ 0".
  m^s = Σ_k w_k g^s_k replaces a.

### 2. Adapter input per stride, without hand-made extras

z^s = [P^s(F^s_t), Ĉ^s, P^s(F^s_B), m^s].

- Then the same blocks as today: coarse-to-fine 16 → 8 → 4, GroupNorm/GELU, then the zero-init output.
- Dropped: the explicit differences F_X − F_C and F_X − F_B. They are linear in the concatenation after the shared 1×1
  projection, so no information is lost (`docs/related_work_minimal_adapter.md`).
- P^s stays the shared 1×1 projection per stride.

### 3. Token correction

- **Prototype:** the gate-weighted composite of the **raw** 256-channel stride-16 past features under m¹⁶ (64×64, cheap).
  Today it is the C16 prototype under a.
- **δ = MLP(proto):** zero-initialised, scaled by 1 − exp(−Σm¹⁶/κ). The tracker-anchored point token is unchanged.

### 4. Empty history (mode 2 in training, noC at inference)

The history is replaced by the empty scene **and the mass is forced to m := 0** (so δ's scale is 0), exactly as the
current code sets a := 0 in both places (`fusion_data.build` comp(), `per_frame_inputs` noC). This keeps the
one-pass noC matched to the old noC, and mode 2 stays a pure fallback, not a gate training signal. Frames with fewer
than 6 predecessors fall back to per-frame SAM2 with tracker prompts, as today.

### 5. Step-0 behaviour

The output residuals and δ stay zero-initialised, so at step 0 the one-pass adapter equals per-frame SAM2 with tracker
prompts ("trk"), as today. The gate starts at σ(bias 0) = 0.5 everywhere.

## Arms and evaluation

| arm | what | status |
|---|---|---|
| adapter-pf / -s1 / -s2 | current adapter, seeds 0/1/2, step 200 | masks exist (cluster) |
| adapter-pf-noC / -s1-noC / -s2-noC | current adapter with C := B, a := 0 at inference | seed 0 exists (cluster); seeds 1, 2 to dump |
| **onepass-s0/s1/s2** | this design, seeds 0/1/2, step 200 | to build |
| SAMURAI (t1-c2), SAM2 video (b0-c2) | baselines | masks exist (cluster) |

- **Metrics:** hand J/P/R (23 frames), box AUC/P@20/P_norm (46 windows), recognition AP50 + conf at N = 7. Paired
  window bootstrap and segment-cluster bootstrap, as in `downstream/report.py`.
- **Declared secondary (reported, never used for selection):** step 600 for onepass and for the current adapter
  (seeds 0/1 have it600). Answers "does the gate need longer training?" without test-set selection.
- **Synthetic val J (30 clips):** reported as a sanity check only. It is not a selection metric: on 2026-10-20 it
  ranked the losses opposite to real data.
- **Runtime:** a lean timing of the one-pass inference path in `benchmark/nfo_vos/timing.py`, same laptop GPU. The
  target is ≤ 35 ms per frame.
- **noC of the one-pass adapter** (history := empty scene) at inference, as the matched no-composite control. Then the
  decision rule can also be read within the new architecture.

## Compute

- **Training:** per clip, every frame is encoded once (as today) and the pixel-composite encode is dropped, so a step
  should be cheaper than today's ~3–5 s. That is *my estimate*.
- **Memory:** the stride-4 history is 32 targets × 6 × 32 ch × 256² ≈ 0.8 GB at fp16. If a 16 GB GPU runs short,
  reduce targets per clip to 16 and record the change.
- **Cluster:** `fusion_adapter.sbatch` with a new `MODE` (e.g. `onepass`); any node except the exclude list, AMP auto
  (fp16 on Turing, logged).
- **Masks:** new arms in `dump_arms.py` via `ARM_CKPT`, plus a one-pass inference path.

## Review gates before the first real run

1. Builder check of this document (fresh reviewer).
2. Expert pipeline review: one reviewer subagent, pre-authorised by CLAUDE.md. It checks:
   - target/label consistency and zero-init;
   - train/test parity: synthetic `build` vs the NFO inference path must produce the same history tensors, shift
     convention and gate inputs;
   - empty-history and mode-2 handling, and the ablation list.
3. A smoke test, local and on the cluster (`remote_smoke.py` extended with one onepass training step).

## Left to the builder

- Module and function names. Whether onepass is a `mode` of `Fusion` or a sibling class. Keep the old path intact:
  the current adapter's checkpoints must still load.
- How the ring buffer of past features is organised in the NFO inference path.
- The exact φ width (suggest 32 channels) and gate-head initialisation, as long as the gate starts at 0.5 everywhere.
- Unit tests:
  - the shift: an integer shift equals `torch.roll` with border replicate;
  - the gate fallback;
  - at step 0, onepass output == trk output.

## Known deviations (declared)

- **Learning-rate schedule.** Current-adapter seeds 1 and 2 ran with STEPS=1000. Seed 0 (2026-10-16) was killed at
  ~625 steps and its planned length is unconfirmed (1500 is the default). With T_max 1500, its learning rate over the
  first 200 steps was up to ~6% higher than with 1000. All one-pass seeds use 1000.
- **Precision.** Current seeds 0 and 1 trained in bf16 (laptop), seed 2 in fp16 (Turing, node8). One-pass seeds run
  on any cluster node with AMP auto. The dtype is logged per seed and reported; it is not pinned.

## Calls I made (for the user to check)

- Gate before shift (pixel-composite order); empty history forces m := 0 (old a := 0 convention).
- Decision rule compared within the architecture (one-pass − one-pass-noC, per seed); threshold = half the current
  adapter's 3-seed composite gain, so adapter-noC is dumped for seeds 1 and 2 first.

- K = 6 fixed history in training (was n_h ∈ 4..8 randomised).
- Gate computed once at stride 4 and pooled, not one gate per stride.
- Token prototype from raw stride-16 composite features under the learned mass.
- Primary checkpoint step 200 (protocol parity); step 600 declared secondary.
- Recency τ learned, init 3.5.

## Expert pipeline review (2026-10-20): changes adopted

**Blocking, fixed in the design.**
- **B1 One history function, two callers.** `history(P_feats[K], gates[K], F_B, dx_native, W_native) → (Ĉ^s, m^s, proto)`
  is written once and called by both the synthetic `build` path and the NFO inference path (ring buffer, from
  `video_adapter.fast_masks`). The grid scale is taken from the clip's native width, never from `S1`, and both paths
  store features in the same dtype. Parity test: one NFO window through both paths must give allclose Ĉ, m and proto.
- **B2 Conditional arm for the failure branch.** If decision rule 1 fails, run one more arm: a feature composite with
  the **old hand-made gate** (pixel z-score gate, q90 threshold, area-pooled to strides 16/8/4, fixed τ 3.5), one seed.
  It costs no extra encoder pass. This separates "features vs pixels" from "learned vs hand-made gate". Without it, a
  failure is reported only as "this one-pass design lost", never as "pixels beat features".

**Should-fix, adopted.**
- **S1 Gate collapse.** No supervision is added. At every eval the run logs, for synthetic val and seq1: Σm, m on the
  person, the gate's spatial std, and the per-input feature norms (step 0).
  - Collapse is pre-declared as either of:
    - gate spatial std < 0.05 on synthetic val at step 200;
    - seq1 median Σm outside [0.5×, 2×] the synthetic median.
  - A collapsed run is reported as such and not used for the decision rule.
- **S2 Token δ.**
  - κ is rescaled to the stride-16 grid: init 500/16 ≈ 31 cells.
  - The prototype is the m¹⁶-weighted mean of the composite **inside the tracker box** (box_trk), not over the whole frame.
- **S3 Memory.**
  - The 1×1 projection P and the gate g are computed **once per clip frame**, not per target × k.
  - Shifting happens on the **projected** features (64/32/32 channels). P commutes exactly with a bilinear,
    border-replicate shift, because the interpolation weights sum to 1.
- **S4 Precision.** Ĉ = (…)/(m + ε) and the gate sigmoid are computed in fp32, outside autocast.
- **S5 Absolute bar.** Added to decision rule 1: one-pass − SAMURAI box AUC > 0, with its window-bootstrap CI excluding 0.

**Nice-to-have, adopted.**
- **N1 History dropout.** The tensor shape stays K = 6. Per clip, n_h ∈ {4, 5, 6} is drawn, and the w_k of the dropped
  oldest frames are zeroed and renormalised. This keeps the old history-length augmentation.
- **N2** log τ and log κ are excluded from weight decay.
- **N3 Unit tests.**
  - An integer shift equals a roll with border replicate.
  - A sub-cell shift of a pixel map, then area-pooled, matches pooling then shifting within tolerance. The stride-16
    sub-cell blur is inherent and declared.
  - At step 0, one-pass equals trk.
- **N4** Assert dx is finite. On a NaN or unset tracker, fall back to the trk arm for that frame.
