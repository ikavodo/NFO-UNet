# Synthetic NFO: a controllable, NFO-matched GPJATK + foliage dataset — experiment design (2026-10-15, DRAFT)

Status: draft for sign-off. **Sign-off needed on §2 (hypotheses), §6 (pass/fail) and §4 (label convention).**
Nothing below has been run unless it says "measured".

## Phase 0 results (measured 2026-10-15, T1, 6 clips × 39 frames each; `sanity_recall.py` with BACK / CLUMP)

| setting | person occluded | V area share by size bin (cells) | T1 vs V: J / P / R | T1 size-binned recall |
|---|---|---|---|---|
| lab, clump 12, d45 | 0.45 | 0.020 / 0.037 / 0.245 / 0.698 | 0.534 / 0.588 / 0.852 | 0.25 / 0.48 / 0.68 / 0.92 |
| **photo back**, clump 12, d45 | 0.45 | same matte | 0.525 / 0.587 / 0.844 | 0.23 / 0.29 / 0.66 / 0.93 |
| lab, **clump 24**, d45 | 0.45 | **0.012 / 0.011 / 0.115 / 0.861** | 0.716 / 0.791 / 0.890 | 0.16 / 0.58 / 0.63 / 0.90 |
| lab, **clump 48**, d45 | 0.40 | 0.002 / 0.005 / 0.026 / 0.968 | 0.840 / 0.893 / 0.940 | 0.19 / 0.65 / 0.56 / 0.94 |
| lab, clump 24, d75 | 0.80 | 0.048 / 0.072 / 0.372 / 0.508 | 0.216 / 0.228 / 0.530 | 0.34 / 0.47 / 0.31 / 0.61 |
| NFO (handAv, seq2–4) | — | 0.013 / 0.009 / 0.108 / 0.870 | 0.750 / 0.887 / 0.819 | 0.13 / 0.39 / 0.67 / 0.84 |

**Verdicts.**
- **HB is killed.** The back layer does not change precision (0.587 vs 0.588 at d45; 0.106 vs 0.126 at d75).
- **Occluder scale relative to the 16-px cell is what controls SAM2's behaviour.**
  - With clump 12 px, occluders are about 1 cell wide, and SAM2 bridges them: its mask lies 92% inside S, i.e. it
    behaves amodally.
  - Coarser clumps make SAM2 modal.
- Clump 24 / d45 matches NFO's fragment-size distribution within 0.01 per bin (an input statistic; found by
  probing, so it counts as calibration, not as a held-out test).
- At that setting, T1 is still more generous than on NFO (P 0.79 vs 0.89, R 0.89 vs 0.82).
- This is consistent with HT: NFO's fine foliage (the void) is missing, and rung C adds exactly that layer.
- **Reconciles the user's "high density".** NFO's *coarse* occlusion is moderate (≈ 45% of the person). The total
  is high once the fine layer is added, since void is 47.6% of (person ∪ void).

**Revised ladder.** HB is dropped, so rung B now means: rung A with the matte at clump 24 / d45 and the photo
texture. Rung C adds the fine branch layer, with void exactly as in handAv.

## 0. Context

**Goal.** A synthetic dataset with exact layers (person S, occluders, visible part V), close enough to NFO that
SAM2/SAMURAI *behave* on it as they do on NFO. Layer-SAM2 is trained there (architecture v2,
`layered_sam2_architecture.md`).

**Why the first attempt failed** (measured, 2026-10-15):
- Setup: lab scene with an occ_perlin matte filled with the photo.
- At high density SAM2 is *precision*-limited. The box prompt fills a person-sized region onto the foliage.
- NFO is recall-limited instead: T1 P 0.887, R 0.819.

**What "as close as possible" means here.** We do not want an image that *looks* like NFO. We want a dataset on
which:
1. the input statistics that set the H2 mechanism match NFO (fragment sizes in cells, contrast, person scale);
2. the tracker behaviour matches NFO on statistics that were *not* used to fit the generator.

## 1. Mechanism

**H2 mechanism, stride-16 tokens** (sketch, derived from SAM2's architecture, not fitted).
- A visible fragment covering a fraction φ of a 16-px token cell contributes roughly φ of that token's feature.
- The rest of the cell is occluder and back-layer content.
- With person/clutter feature contrast Δ_P and Δ_C, the token's person evidence scales like
  φ·Δ_P − (1 − φ)·Δ_C.
- Predictions:
  - recall rises with fragment size in cells and with person contrast;
  - recall falls with clutter contrast in the cell, which depends on what lies *behind* the gaps.

**Why the lab scene breaks this** (hypothesis, consistent with the measured precision collapse).
- Behind the gaps lies a plain lab wall, so Δ_C is near 0 for the back layer.
- Every textured pixel inside the box is "not wall", so SAM2 groups person and front foliage together.
- On NFO, foliage also lies behind the person. The gaps beside a fragment show more foliage, Δ_C is large, and
  SAM2 keeps only confident person tokens. That makes it careful and recall-limited.

**The human label convention** (from the user's void definition, 2026-10-13, and handAv):
- Annotators *remove* substantial foliage (modal with respect to coarse occluders).
- They *ignore or void* thin, complex foliage (amodal-ish with respect to fine occluders).
- Measured on handAv: void = 47.6% of (person ∪ void).
- So NFO's occluder has two scales, and the "generous" label is approximately m ∧ ¬coarse. This replaces the
  morphological stand-in T_r = V ∪ (closing_r(V) ∧ S) with an *exact* label.

## 2. Hypotheses

**HB (back layer).** Planting the person in front of a foliage back layer, instead of the lab, turns SAM2 from
precision-limited to recall-limited at fixed density.
- Boundary: at low density (≤ 0.3) both settings are dominated by large fragments and should agree.
- Kill: at density 0.45 and 0.75, T1 precision vs V with the photo back layer is not higher than with the lab by
  at least 0.15.
- (Phase 0; running.)

**HT (two-scale occluder).** Adding a fine branch layer (occ_branch, 1–3 px wide, thinner than half a cell)
reproduces NFO's label/behaviour pair:
- a large void share;
- T1 recall vs H = m ∧ ¬coarse of about 0.8, with P about 0.9;
- size-binned recall rising from about 0.1 (sub-cell) to about 0.85 (≥ 4 cells).

Boundary: if fine-layer pixels are wider than about 1 cell, SAM2 sees them as real occluders, and the
fine/coarse split stops mirroring the human convention.

Kill: no setting of (coarse density, fine density) inside the fitted range puts *all* held-out behaviour
statistics within tolerance (§6).

**HE (easy baseline is enough; the user's prior).** occ_branch matte + perlin texture, lab scene, one occluder
scale, already passes §6. Predicted to fail, given today's result. It is kept as rung A so the claim "the layered
generator is needed" is tested, not assumed.

## 2b. Prior work (verified 2026-10-15)

- **Fitting generator parameters so synthetic data matches a target set:** Meta-Sim (Kar et al., ICCV 2019,
  arXiv 1904.11621). We use the low-dimensional version: grid or bisection over 3–5 parameters against measured
  statistics, no learned scene-graph network.
- **Distribution distance with small samples:** KID (Bińkowski et al., ICLR 2018, arXiv 1801.01401). Secondary
  check only. Grayscale crops break the ImageNet-feature assumption: flag this, do not gate on it.
- **Cut-and-paste compositing artefacts** (boundary halos a detector can exploit): the known issue is "Cut, Paste
  and Learn" (Dwibedi et al., ICCV 2017). The search did not resolve its arXiv id; treat it as unattributed until
  checked. Mitigation here: feather the silhouette with σ = 1 and keep the person's original pixels.

## 3. Baseline and the single variable per rung

The ladder changes one generator component per rung. Nuisance parameters are re-calibrated per rung to the
*input* statistics (§4) before the behaviour is scored.

| rung | back layer | coarse occluder | fine occluder | label scored |
|---|---|---|---|---|
| A (user's easy baseline) | lab | occ_branch matte, perlin texture | — | V |
| B | **photo crop** | occ_perlin matte, photo texture | — | V |
| C | photo crop | occ_perlin, photo | **occ_branch, 1–3 px, photo texture** | H = m ∧ ¬coarse; void = m ∧ coarse-free ∧ fine |
| D (optional) | photo crop, defocused, lower contrast | as C | as C | as C |

- Fixed across rungs: GPJATK clips and seeds, person scale (≈ 13 cells tall, NFO median), box + point prompt at
  t0, B0/T1 checkpoints, scoring code (score_void.py logic with void ignored).
- Do-nothing reference: NFO itself. Its between-sequence spread defines "as close as NFO is to itself" (§6).
- No oracle is involved.

## 4. Ground-truth audit

**GPJATK silhouettes.**
- Silhouettes come with the dataset. The extraction method is not stated in `GPJATK/Readme.txt` (a grep found
  no description). **Unverified.** If they come from background subtraction, they carry shadow/hole errors at
  the ~1-px boundary level.
- Acceptable for S: errors are far below the cell scale that H2 depends on.
- Caveat for V near boundaries: feathering hides a 1-px misfit, but the boundary band is not exact.
- Not a GT-as-source contamination risk: nothing we test is derived from the silhouettes.

**Occluder masks.** Exact by construction.

**Human-convention label H.**
- This is a *modelling choice*, grounded in the user's void definition, not measured.
- Validated only indirectly, by whether behaviour on H matches NFO's behaviour on handAv (§6).

**NFO targets** (measured today, seq2–4 handAv, n = 23 frames, one annotator):

| statistic | NFO value |
|---|---|
| person height | median 155–172 px of 600 → about 13 cells of 12.5 px |
| label area share by component size [0,.5) / [.5,1) / [1,4) / [4,∞) cells | 0.013 / 0.009 / 0.108 / 0.870 |
| person / (person + void) | 0.524 |
| person contrast | robust z ≈ 9 (seq1), ≈ 3 (seq2) |

For comparison, today's synthetic set had V area shares of 0.007/0.010/0.045/0.938 (d30) and 0.020/0.037/0.245/0.698
(d45). NFO lies between them.

**Limits.**
- One annotator. Tolerance bands must widen once the inter-annotator J exists.
- The foliage photo licence was cleared by the user.

## 5. Feasibility (toy)

- The generator, SAM2 runs and scoring already work end to end (18 clips in about 2 min, measured).
- Remaining toy checks:
  - the cut-out person composites without NaNs or halos (montage, user judges);
  - occ_branch at 1–3 px renders at the requested width after the blur.

## 6. Pre-registered pass/fail (to sign off before rungs A–D run)

**Fitted** (calibration targets; the generator's density, clump size, fine-branch count and contrast gain are
tuned to these on 6 calibration clips):
1. person height ≈ 13 cells;
2. H area share by size bin within ±0.05 of NFO in each bin;
3. void share 0.48 ± 0.10 (rungs C/D only);
4. person contrast median within [3, 9].

**Held-out behaviour** (never used for fitting; scored on 12 fresh clips, other subjects):
- T1 vs H, void ignored: P within [0.83, 0.95] and R within [0.75, 0.89] (NFO 0.887 / 0.819 ± about 0.07, the
  approximate seq2–4 per-sequence spread; to be recomputed exactly before running).
- T1 size-binned recall: sub-cell ≤ 0.25; [.5,1) in [0.25, 0.55]; [1,4) in [0.55, 0.80]; ≥ 4 cells ≥ 0.75.
- B0 − T1 |ΔJ| ≤ 0.03 (NFO: +0.005).

**Verdicts:**
- A rung **passes** if every held-out statistic is inside its band.
- The dataset is "NFO-close" at the first passing rung.
- If no rung passes, report the closest rung and the failing statistic. No post-hoc band widening.

## 7. Scope-cutting order (decided now)

1. Drop rung D.
2. Drop B0 (keep T1).
3. Reduce held-out clips from 12 to 6.
4. Never cut: the held-out/fit split, or rung C (it carries the label convention).

## 8. Execution order (each step gated on the previous one)

1. **Phase 0 (HB):** lab vs photo back layer, everything else fixed. *Running 2026-10-15.*
2. Re-score NFO's per-sequence spread exactly, and fix the bands in §6.
3. Rung A, then B, then C: calibrate, then run held-out behaviour, then decide. Stop at the first pass.
4. Montage per rung into `images/benchmark/gpjatk_foliage/`. The user judges.
5. Only then: the scaled dataset for Layer-SAM2 step 2.

## 9. Adversarial notes

**Strongest objection.** Matching SAM2's behaviour statistics does not imply that a model *trained* on the
synthetic set transfers to NFO. Behavioural matching is necessary, not sufficient. The real test is step 3:
a model trained on synthetic data, evaluated on NFO tier B.

**Second objection.** The label H = m ∧ ¬coarse encodes our *theory* of the annotator. If the theory is wrong,
§6 can still pass by tuning the fine/coarse split. Partial guard: the fine layer's width is fixed by the "< half
a cell" boundary *before* fitting, and only its count is fitted.

An idea-critic pass could plausibly catch a missing ancestor in synthetic-occlusion datasets for VOS (e.g. Kubric
/ MOVi, synthetic amodal sets). Ask before dispatching.

## Occluder ≡ background (user proposal, 2026-10-15; rule of MovingMNIST-OcclusionBench `occluders.py:450`)

**Model.** Occluded person pixels show the scene background itself: I = a·B + (1 − a)·I_person.

**Derived.** For a static camera and static foliage, this is the exact image formation. A pixel shows the person
iff the person is in front of whatever is first visible there in the empty scene. So:
- B̂ = B, as with NFO's warm-up median;
- occlusion outside the person's trajectory changes no pixel. Restricting it to the trajectory matters only for
  bookkeeping the coverage.

**Measured** (T1, d45, 6 clips):

| background B | clump | J / P / R vs V | size-binned recall |
|---|---|---|---|
| lab | 12 | 0.756 / 0.786 / 0.957 | 0.29 / 0.88 / 0.94 / 0.98 |
| lab | 24 | 0.856 / 0.900 / 0.948 | 0.20 / 0.85 / 0.86 / 0.94 |
| photo (person planted) | 12 | 0.563 / 0.620 / 0.865 | 0.17 / 0.37 / 0.70 / 0.94 |
| photo (person planted) | 24 | 0.672 / 0.740 / 0.890 | 0.16 / 0.48 / 0.58 / 0.91 |
| NFO | — | 0.750 / 0.887 / 0.819 | 0.13 / 0.39 / 0.67 / 0.84 |

**Reading.**
- A plain lab wall in the holes makes the task too easy: mid-size recall is 0.85–0.88 against NFO's 0.39–0.67.
- The photo background gives an NFO-like size profile, but precision is still 0.74.
- Voiding thin occluders (opening with a 7-px disk) changes nothing: perlin blobs have almost no thin parts
  (void share 0.1–0.9% against NFO's 47.6%).
- The fine structure must come from elsewhere:
  - a depth-consistent mask taken from the photo itself (z-test against a monocular depth map);
  - or an added twig layer.
