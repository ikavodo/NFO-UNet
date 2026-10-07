# Port A/4 to the NFO benchmark, gated by SAM2's confidence

Status: design, 2026-10-07. Branch `vos-benchmark-pilot`, run root `results/benchmark/rr`.
Goal: **beat B0/T1 on the benchmark headline (modal J&F)**, without losing on frames where the
baseline is already right. Amodal extent is the secondary target.

## 0. Evidence this rests on (measured)

**NFO full run** (56/61 starts per method, preliminary; `results/benchmark/rr/`):
- B0 J&F 0.831, T1 0.830. Init failures 16/56 and 14/56, concentrated in seq1, seq2 and seq4.
- 13–14% of person-present frames fail (J < 0.5). That is the headroom.
- **SAM2's predicted IoU flags failing frames:** AUC 0.853 (B0) / 0.872 (T1); Spearman with J
  +0.78 / +0.79. The object score is weaker (AUC 0.77–0.79).
- Cover/backup twins agree at r = 0.94 / 0.91, so outcomes are set by position in the scene.

**GPJATK** (`master_thesis/results/gpjatk_em_viability/fusion_location/SUMMARY.md`, Addenda 3, 6, 7;
one sequence, one prompt, no CIs):
- The composite is the **full-frame recency-weighted mean**, w = exp(−age/3.5), of the 7 frames
  ending at t, aligned by constant velocity. No occluder mask (Addendum 1).
- **A/4** = one SAM2 memory bank, with composite C_t inserted just before raw R_t at every 4th t;
  the same prompt on C_t0 and R_t0. At d80: full extent 0.392 (best of all designs), visible part
  0.441 vs 0.471 for raw alone. **Fixed A/4 costs about 0.03 on the visible part.**
- Inserting at every t (A) collapses at d80, because composites crowd raw frames out of the 6
  memory slots.
- SAMURAI's memory-admission gate (predicted IoU > 0.5, object score > 0) does not help:
  "memory composition is not the bottleneck" (Addendum 6). That gates *admission*, not *insertion*.
- Output rule: full extent = composite mask ∪ raw mask; visible part = raw frames.

**Why gate insertion (argued):** fixed A/4 trades visible-part accuracy for extent everywhere. The
NFO headline is the visible part, and failures are concentrated (14% of frames, mostly after
failed inits). Inserting composites only when SAM2's confidence drops leaves confident stretches
identical to the baseline, and puts the extent cue exactly where the baseline is losing the person.

## 1. Arms (one base: SAMURAI, the stronger base with composites in the earlier pilot; B0 versions optional)

| arm | sequence fed to one memory bank |
|---|---|
| T1 | raw frames (baseline, already run) |
| T1-A4 | C_t before R_t at every 4th t (ported design, fixed) |
| **T1-G(τ)** | C_t before R_t **only if the baseline's predicted IoU at t−1 < τ** (and at t₀: if the baseline's predicted IoU on the prompted frame < τ, insert C_t0 and prompt both C_t0 and R_t0) |

**Open-loop gate:** the trigger uses the baseline run's logged predicted IoU, not the gated run's
own (SAM2 cannot add frames mid-propagation, and skipping preloaded slots would shrink its memory
window). It is causal (uses only frames ≤ t−1) and GT-free. **If nothing triggers, the frame
sequence equals the baseline's, so the output is identical.** A closed-loop gate (react to its own
confidence) is a later refinement, if the open-loop gate works.

## 2. Construction (port; reuse, do not rewrite)

- **Composite C_t:** recency-weighted mean (exp(−age/3.5)) of the 7 native frames ending at t,
  each shifted onto frame t by the **GT-free hybrid tracker velocity**
  (`composite.estimate_shifts(..., 'hybrid_x')`, 7.7 native px median error). Full frame, no mask,
  no cut-out. Port `recency_fusion` from `master_thesis/.../fusion_location.py:493`, cite it.
  Frames before t₀ are allowed in the buffer (causal; the tracker has seen them).
- **Runner:** generalise `run_sam2_video.run_trial`'s `prepend` into an insertion schedule
  `[('c', t) | ('r', t)]`. Stage frames in that order, prompt the first raw frame (and C_t0 if
  inserted), map raw-frame outputs to the scored masks and composite-frame outputs to a separate
  `extent` array. Keep the predicted-IoU capture.
- **Outputs per trial:** `masks` (raw frames → modal, the headline), `extent` (C ∪ R where a
  composite exists) and `inserted` (bool per t).

## 3. Tuning and evaluation (pre-registered)

- **Development set: seq1** (15 starts; already used for every pilot choice). τ ∈ {0.5, 0.6, 0.7,
  0.8} is chosen by mean modal J&F on seq1 only. A/4's period stays 4, ported, not tuned.
- **Test set: seq2–seq4** (46 starts), evaluated **once** with the chosen τ.
- Report: J&F, init failures, DRE/NRE, the share of frames with an inserted composite, and the
  per-sequence table. ΔJ&F vs T1 with the 95% cluster-bootstrap CI (twin groups, stratified by
  sequence). Secondary: amodal box IoU of `extent` vs `groundtruth.txt` boxes.

**Anticipated (before running):**
- T1-A4 < T1 on modal J&F (GPJATK: −0.03 on the visible part); T1-A4 > T1 on amodal extent.
- T1-G(τ*) ≥ T1 on modal J&F. Gains come from starts whose baseline predicted IoU is low early
  (failed inits); confident starts are unchanged by construction.

**Decision rules (test set):**
1. **Method beats baseline:** ΔJ&F(T1-G − T1) > 0 with the 95% CI excluding 0.
2. **Partial:** Δ > 0 but the CI includes 0. Report it as "no worse, gains on failed starts"
   (per-start table) and run the closed-loop gate next.
3. **Fails:** Δ ≤ 0. The extent cue does not repair visible-part failures on NFO; the A/4
   contribution stays amodal (secondary metric).

## 4. Tasks (TDD, one commit each)

1. **Composite builder.** `recency_composite(trial, t)` → native frame (tests: weights sum to 1 and
   favour the newest frame; with zero shifts and a static scene it reproduces the input).
2. **Insertion schedules.** `schedule_fixed(T, k=4)` and `schedule_gated(baseline_pred_iou, τ)`
   (tests: k = 4 inserts at t ≡ 0 mod 4; with τ below every predicted IoU nothing is inserted;
   the gate at t uses only predicted IoU ≤ t−1).
3. **Runner.** `run_trial(..., schedule)` stages the interleaved sequence (test: an empty schedule
   gives masks identical to the plain run, bit for bit; outputs map back to raw frames).
4. **Build composites** for the insertion points needed (all t for A/4; gated only where it
   triggers on the grid of τ), cache them under `results/benchmark/rr/composites_a4/`, and commit
   only small traces.
5. **sbatch:** variants `a4` and `gated:<τ>`; seq1 first (dev), choose τ, then seq2–4.
6. **Analysis:** add the arms, the insertion share and the extent metric; write the result against §3.

## 5. Risks

- Open-loop vs closed-loop: once a composite rescues the track, the baseline's low confidence can
  keep triggering insertions. Watch the insertion share.
- Run starts were hurt by the earlier prepend composite (7 × stride-2 span). A/4's span is 7
  consecutive frames, so check runs separately.
- The B0/T1 comparison environment is the cluster one (official SAM2 `2b90b9f`, SAMURAI `76ba195`,
  no flash attention), so all arms must run there too.
