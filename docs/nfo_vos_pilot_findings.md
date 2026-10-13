# NFO VOS pilot: what worked (2026-10-05)

Pilot: seq1 segments 4–5, 21 starts, 51-frame windows, pseudo-GT at 224 (spec
`docs/superpowers/specs/2026-10-02-nfo-vos-benchmark-pilot-design.md`, rev. 4). Two segments: all
numbers descriptive.

## The benchmark protocol works and the baselines are not straw men
- DAVIS-style headline (frames t₀+1…t₀+49), init failure rate (J(t₀) < 0.5), VOTS DRE/NRE, LaSOT
  P_norm, frame-fixed-effects init model. Scoring at 224 vs native changes J by median 0.006
  (621 real mask pairs, r = 0.999).
- B0 (SAM2.1 b+) J&F 0.779, 5/21 init failures; T1 (SAMURAI) 0.773, 5/21: motion-aware memory does
  not fix fragmented occlusion. B0's failures are at the prompted frame itself (same prompt → same
  t₀ mask for B0/T1), not propagation drift; after a good start it recovers from brief losses
  within ≤ 3 frames.

## Integration helps AMODAL extent, not modal masks
- Amodal box IoU vs the human `groundtruth.txt` boxes (frames t₀+1…t₀+49):
  integrated blob union (tracker alignment, 7 consecutive frames, no SAM2, ~3 ms/frame) 0.652 vs
  frame-t blob box 0.546; vs the frame-t box optimally enlarged (best-of-grid, favours control)
  0.558 → **+0.094, 17/21 starts**. Oracle alignment adds only +0.004 → the tracker's alignment is
  not the bottleneck for this task.
- Versus B0's modal box optimally enlarged (1.3× width): +0.014, 8/21 → a tie with SAM2 video at
  a fraction of the compute.
- Why (derived): integration recovers person pixels visible at t−k but occluded at t (A∖V_t).
  That is exactly amodal extent, and exactly what modal GT penalises.

## Native SAM2 input is the dominant fix for the per-frame path
- With the identical GT prompt at t₀, SAM2's image predictor on the native full frame matches B0
  (J 0.700 vs 0.703, 5/21 init failures each). On the 224 frame it reaches J 0.497 with 11/21
  failures (`diag_init_resolution.csv`).
- T4-1n (the tracker's prompt → SAM2 image on the native frame, no memory): J&F **0.700**, against
  T4-1 (224 crop) 0.580 → **+0.120, better on 21/21 starts**. Masks are right-sized (area ratio
  0.95, precision 0.90).
- T4-1n vs B0: −0.079 on average (better on 3/21), but it wins exactly the starts where B0's
  memory locks onto the wrong object: t1154 +0.13, t1164 +0.25, t1184 **+0.70**. Re-prompting every
  frame cannot lock in. Memory is more accurate after a good start.
- Complementarity headroom: choosing the better of B0 / T4-1n per frame (oracle) gives 0.842
  (+0.063 over B0). A GT-free switch based on the tracker box does not capture it (0.755):
  the switch signal is the bottleneck.

## Composite-prepend memory init helps both SAMURAI and SAM2 (GT-aligned, causal)
- Port of master_thesis composite-prepend: the integrated image of 7 past frames (stride 2, ≤ t₀,
  aligned with **GT** box centres) is cut out by SAM2 and pasted onto the warm-up background.
  This frame is prepended and prompted, so it becomes the conditioning memory.
- T1-cp J&F **0.843** vs T1 0.773 (**+0.070, better on 15/21**); B0-cp 0.824 vs B0 0.779
  (+0.045, 16/21). DRE (drift onto the wrong object) 0.07 → 0.015. The largest gains are on B0/T1's
  lock-in starts (t1184: 0.02 → 0.72 T1-cp, 0.63 B0-cp), with small gains on good starts and one
  loss (t1174). Unlike master_thesis tum3, SAM2 does not collapse with the composite here.
- Caveat: the alignment is GT-oracle (plausibility only). The decisive next test replaces it with
  the tracker's own causal alignment.

## GT-free (tracker) alignment keeps about half of the composite gain for SAMURAI
- Hybrid velocity (track OLS if ≥ 3 points, else chained Theil–Sen, x-only; 7.7 native px median
  error vs GT shifts), full 7-frame causal horizon. Run on the cluster (official SAM2 `2b90b9f` /
  SAMURAI `76ba195`, Quadro RTX 6000, no flash attention). The local baselines differ in
  environment, which is a known caveat.
- T1-cp-hybrid J&F **0.806** vs T1 0.773 (**+0.033, better on 15/21**), vs GT-aligned T1-cp 0.843
  (−0.037). On 19/21 starts it matches the GT-aligned composite within ±0.04, and it keeps the
  lock-in rescue (t1184: 0.02 → 0.68). Almost the whole gap to GT comes from **t1154 and t1164**
  (0.16 / 0.25, worse than plain T1). These are the segment-start trials, where the tracker's
  7-frame history reaches before the person is in view (frames before the GT segment), so the
  composite is built from the wrong content.
- B0-cp-hybrid (cluster SAM2 `2b90b9f`; local baseline `sam2 1.1.0`, a caveat): J&F 0.787 vs B0 0.779
  (+0.008, 14/21). It matches GT-aligned B0-cp on the other 19 starts, but **collapses** on t1154/t1164
  (0.08 / 0.00, empty masks, NRE 0.115), where SAMURAI only degrades. This is the same
  appearance-anchor-needs-motion-gating pattern as master_thesis tum3. The user judged the composite
  content there to be the right person, heavily occluded (not a bug).
- Failure signature (descriptive, n = 2): the SAM2 cut-out on the integrated image is **oversized**,
  S/expected-area = 2.85 and 3.45 vs a median of 1.47 (expected = κ·|box at t₀|). Gating the
  prepend on S/expected ≤ 2.5 would give T1 0.815, B0 0.816. The threshold is fitted on these 2
  starts, so pre-register it for the full run, and fix κ as a constant (0.39) there, because the
  pilot's κ is GT-derived.

## Round-robin pilot (rr, seq1, 15 starts, W = 40; cluster env) — 2026-10-07
- **The twin design works:** cover vs backup twins (same position, different pass) give J&F
  r = +0.99 (B0) / +1.00 (T1), mean |Δ| 0.05 / 0.04. Outcome is set by position in the scene,
  which supports the schedule's premise.
- Baselines: B0 J&F 0.754 (5/15 init failures), T1 0.758 (3/14; one T1 task, array id 21, missing).
- **The hybrid composite does not replicate on the broader pilot:** T1-cp-hybrid 0.733 (−0.017, better
  on 8/15), B0-cp-hybrid 0.684 (−0.071). Walks are about neutral (Δ −0.005 / −0.022). **Runs are
  harmed** (Δ −0.057 / −0.205: t617 B0 0.92 → 0.02, t1835 T1 0.91 → 0.60), except one big gain
  (t2050 +0.16/+0.18). The old pilot's +0.033 came from 2 walking segments and a lock-in
  rescue (t1184) that is not a start here. Composites are also less reliable across twins
  (r 0.64 / 0.89).
- **The oversized-cut-out gate (S/expected ≤ 2.5) fails out of sample:** no rr composite exceeds
  2.5. The losers do have the largest S (2.18–2.44), so the direction holds, but any threshold
  would be refitted on this pilot.
- Hypothesis for the run failures (untested): a fixed span of 7 × stride 2 frames covers about
  1.6× more travel at run speed, which means more velocity error and articulation in the
  integrated image.

## Full baselines and the A/4 dev run (seq1) — 2026-10-07
- Full rr run, 61 starts: B0 J&F 0.828 (17 init failures), T1 0.831 (15). T1 − B0 = +0.003,
  95% CI [−0.003, +0.011]. Cover/backup twins r = 0.94 / 0.91. SAM2's predicted IoU flags failing
  frames (J < 0.5): AUC 0.860 (B0) / 0.863 (T1).
- Init failures concentrate in the first two windows of a segment (28–31% vs 8% in the third) and
  at low v(t₀) (median 0.27 vs 0.70).
- **A/4 dev (seq1, T1 base, partial: 13–15 of 15 starts per arm), modal J&F Δ vs T1:**
  fixed A/4 −0.017 (two large losses: t853 −0.166, t1154 −0.128); gated τ = 0.5 / 0.6 / 0.7
  +0.003 / +0.003 / +0.004 (CIs include or touch 0); τ = 0.8 −0.006. Gating removes A/4's losses
  but rescues little: the worst start (t11, J&F 0.08) gains +0.001 despite 10 composites. Its
  history is as occluded as t₀, so the composite has nothing to add.
- **Amodal extent (secondary), on inserted frames:** composite ∪ raw vs T1's mask, box IoU against
  the amodal GT boxes: fixed A/4 0.619 vs 0.581 (**+0.038**, better on 11/14 starts); gated τ = 0.7
  +0.027 (11/14). This reproduces GPJATK: A/4 helps full extent and costs visible-part accuracy;
  gating keeps the visible part at baseline.
- Cluster determinism: identical inputs on different GPU nodes differ by ≤ 5·10⁻⁴ J&F.

## A/4 pre-registered test (seq2–4, T1 base, τ = 0.7 fixed on seq1) — 2026-10-07
- Visible-part ΔJ&F vs T1 (final, 46 starts: gated −0.0025 [−0.0078, +0.0017], fixed −0.0058 [−0.0123, −0.0007]; earlier 45-start values: 95% cluster-bootstrap CI stratified by sequence):
  **gated −0.003 [−0.008, +0.002]** (better/worse 14/18); **fixed A/4 −0.006 [−0.012, −0.000]**.
  **Decision rule 3 fires:** the composite's extent cue does not repair visible-part failures.
- **Full extent (secondary) holds on unseen scenes:** box IoU vs amodal GT on inserted frames, gated
  0.687 vs 0.609 (**+0.078**, 130 frames); fixed 0.719 vs 0.663 (**+0.056**, 450 frames).
- Implication: on NFO, A/4 is a full-extent method, not a visible-part one. To beat the baselines on
  the visible part, the lever is initialisation. Next: an oracle GT-mask-init ceiling run
  (`--init-mask`, methods b0-om/t1-om).

## Oracle-init ceiling (GT mask at t₀ instead of box + p*) — 2026-10-07
- **Final, all 61 starts:** B0 0.828 → 0.858 (+0.030 [+0.007, +0.055]); T1 0.831 → 0.861
  (+0.030 [+0.012, +0.053]). On each method's own init-failure starts: B0 0.714 → 0.806 (+0.092,
  n = 17), T1 0.709 → 0.803 (+0.094, n = 15). By scene (T1): seq1 +0.071, seq4 +0.036, seq2/seq3 ≤ 0.01.
- Preliminary (57–58 starts) numbers below, kept for the record:
- B0 0.831 → **0.861 (+0.030, 95% CI [+0.006, +0.056])**; T1 0.831 → **0.863 (+0.032, [+0.012, +0.057])**.
- The gain sits on the init-failure starts: B0 0.713 → 0.813 (+0.101, n = 15), T1 0.727 → 0.811 (+0.084,
  n = 17). Other starts gain +0.005 to +0.010. By scene: seq1 +0.082, seq4 +0.03, seq2/seq3 ≈ 0.
- Drift almost disappears (DRE 0.014 → 0.005 / 0.001).
- Reading: a better first mask is a real lever for the visible part, worth at most about +0.03
  overall and +0.08 to +0.10 on the hard starts. Even a perfect first mask leaves the hard starts
  below the rest (0.81 vs 0.85+), so propagation under heavy occlusion still loses some.
- Upper bound only: the GT mask is the 224 pseudo-GT mapped to native.

## Visible-mask decomposition at t₀ (extent ∩ background-difference) — 2026-10-08
- On the 17 hard starts (B0 or T1 init failures), J vs the GT t₀ mask: T1's own t₀ mask 0.292,
  B0 0.253. Extent ∩ foreground(t₀) (Otsu inside the extent): prompt box 0.322, A/4 composite
  cut-out 0.325, **oracle GT-aligned composite cut-out 0.335**. Far below the pre-registered
  ≥ 0.6 bar, so no SAM2 init run.
- With the oracle extent the result is no better, so the bottleneck is the **visibility cue**:
  a static-background difference does not separate visible person pixels under foliage.
  (`diag_init_mask.py`, `results/benchmark/rr/diag_init_mask.csv`, traces in
  `images/benchmark/rr_checks/init_mask_test/`.)

## Next (cheapest first)
0. **Do not miss:** SAM-PT and SAM-PD are required baselines for T4-1n in the full run (spec §8).
1. Decide the benchmark's question: modal VOS (B0 strong, integration does not help) vs amodal
   person extent/localisation (integration helps; scored on human boxes, so no pseudo-GT bias;
   matches the original NFO-UNet localisation task).
2. If amodal: proper amodal baselines (an amodal VOS / amodal completion method), not enlarged
   modal boxes; then the feature-fusion SAM2 variant targets amodal extent.
3. Native resolution only matters for the SAM2-input asymmetry (T4 init failures 10/21 at 224
   vs B0 5/21 at native); the integration mechanism is scale-invariant (w/δ).

## T1's failures are over-extent, and the box prompt causes them at t₀ — 2026-10-12
- **Failure decomposition** (rr, T1, 333 frames with J < 0.5, `per_frame.csv`): 72% cover the person
  (recall ≥ 0.5) with precision < 0.5; median recall 0.95, precision 0.33, predicted/GT area 2.68.
  Only 11% are off the person (recall < 0.1, location drift). SAM2's predicted IoU q(t) is not
  uniformly low: median 0.81, 23% of frames below 0.7; P(J < 0.5) is 0.92 at q < 0.3, 0.02 at q ≥ 0.85.
- **The t₀ prompt causes the over-extent on hard starts** (`diag_reinit.py t0cands`, SAM2 image
  predictor on CPU, J vs GT t₀ mask). The shared prompt = amodal GT box + p* (deepest GT pixel).
  - Box + p*, 3 candidates: SAM2's own pick 0.235, best of 3 0.254. All 3 candidates are
    over-extended, so no candidate choice can fix it.
  - p* alone: SAM2's pick **0.525**, best of 3 0.655.
  - Easy starts: 0.717 → 0.768 (SAM2's pick, same setup).
- **Video sanity, 2 hard starts** (CPU, SAMURAI fork, `--point-only`; CPU T1 reproduces cached GPU
  T1 within 0.02 J&F):
  - J&F T1 → T1-pt: seq2_s1_t315 0.849 → 0.862, seq3_s5_t1283 0.782 → 0.865.
  - This is n = 2: a sanity check, not evidence.
- **Caveat for the paper:** a point-only prompt is a prompt-protocol choice that is available to
  every baseline. If it holds on all 61 starts, B0-pt/T1-pt become the baselines to beat. The
  finding is "the amodal box prompt is mismatched with a modal target", not a new method.
- **Next:** B0-pt and T1-pt on all 61 starts (`POINT_ONLY=1 sbatch run_sam2_video.sbatch`), with
  cluster bootstrap vs B0/T1.
- **Result on all 61 starts (cluster, 2026-10-12): the point-only prompt does not hold up over the
  window.**
  - ΔJ&F vs box + p*: T1-pt − T1 **−0.016 [−0.034, +0.005]**; B0-pt − B0 −0.019 [−0.045, +0.009].
  - Hard starts: T1 +0.005, B0 +0.018 (init failures 15 → 12 and 17 → 12).
  - Easy starts: −0.023 / −0.034.
  - Twin test-retest r drops from 0.87–0.92 to 0.51–0.54, so the point prompt is unstable.
  - Large losses come with a good or similar t₀ (seq4_s2_t466: J(t₀) 0.40 → 0.89, yet J&F −0.18). The
    box constrains propagation, even though it over-extends t₀.
  - Choosing the prompt per start: the oracle reaches 0.853, but a GT-free pick by SAM2's predicted
    IoU gets 0.830–0.833, i.e. none of it.
  - **The protocol stays box + p*.** The n = 2 CPU sanity check was selected on hard starts and
    did not predict this.

## Layered-memory option C on NFO: the static layer carries the visibility signal — 2026-10-12
- Plan: `docs/Layered Memory for SAM2 Test Plan.md`, option C step 1. Score inside the GT box at t₀,
  14 hard starts. Features are frozen SAM2 stride-16 backbone features of t₀−6…t₀−1, aligned by the
  GT shift. Motion over the history is 1.5–5.4 cells. AUC, visible vs not:

  | Score | AUC |
  |---|---|
  | cos(f_t, D) | 0.39 (inverted) |
  | the plan's s = cos(f_t, D) − cos(f_t, S) | 0.75 |
  | cos(f_t, D) − cos(f_t, S_bg) | 0.89 |
  | **−cos(f_t, S_bg)** (S_bg = features of the person-free warm-up median) | **0.94** |
  | pixel background difference | 0.85 |

- For −cos(f_t, S_bg): precision at recall 0.9 is 0.46 (pixel background difference 0.28). The best J
  with a per-start oracle threshold is 0.51 (T1's t₀ mask 0.29).
- Next: gate T1's masks on every frame by −cos(f_t, S_bg), with τ tuned on seq1; J&F vs T1 over 61
  windows. `LAYERED=1 python -m benchmark.nfo_vos.diag_composite_diff`.
- **Cleaned dynamic layer (D_clean) adds nothing** (14 hard starts). Each history cell is weighted by
  its own sigmoid(z − 3) static test before alignment. cos(f_t, D_clean) AUC 0.47, and r = −cos S_bg +
  cos D_clean gives 0.91, below −cos S_bg alone (0.94). Its P@R.9 is 0.37 and best J 0.44, against
  0.46 / 0.51 for −cos S_bg alone. The recall-refinement arm is dropped.
- **Static-layer gate over full windows** (`static_gate.py`; output-only, T1's memory untouched;
  θ = 6 on z tuned on seq1; plateau at θ 5–6, +0.020 J on seq1):

  | Comparison (all 61 starts) | ΔJ&F | Trials better |
  |---|---|---|
  | gate − T1 | +0.001 [−0.017, +0.016] | 39/61 |
  | gate − area-matched erosion | **+0.015 [+0.007, +0.024]** | 41/61 |
  | gate − pixel background gate | **+0.021 [+0.001, +0.035]** | 58/61 |

  - Held out (seq2–4, 46 starts): gate − T1 is −0.007 (J −0.013, F +0.007). The seq1 gain (+0.024)
    does not transfer.
  - By start type: hard starts +0.031, easy starts −0.011.
  - Reading: the feature cue removes the *right* pixels relative to blind shrinking or the pixel cue.
    But cutting on every frame costs the 86% of frames where T1 is already fine.
  - Next (pre-register before running): gate only on frames where SAM2's predicted IoU q(t) is
    below τ (the failure detector, AUC 0.86), with θ and τ tuned on seq1 and evaluated on seq2–4.

## PRE-REGISTERED (2026-10-12, written before running): confidence-gated static-layer gate
- **Hypothesis.** The feature gate removes the right pixels (beats erosion by +0.015), but it hurts
  frames where T1 is already right. Applying it only where SAM2's predicted IoU q(t) is low should keep
  the gain on failing frames and avoid the loss elsewhere.
- **Method `t1-gateq`.** On frame t, if q(t) < τ: M′ = M ∧ (z > θ); otherwise M′ = M. Output-only.
  q(t) is T1's own logged predicted IoU for frame t, available causally.
- **Tuning.** On seq1 only, by mean J over frames t₀+1…t₀+38. Grid θ ∈ {3, 4, 5, 6, 8},
  τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.
- **Control `t1-erodeq`.** Area-matched erosion applied on exactly the same frames.
- **Test.** seq2–4 only (46 starts). ΔJ&F with 95% cluster-bootstrap CI (twin groups, stratified by
  sequence).
- **Pass:** gateq − T1 ≥ +0.005 with CI lower bound > 0, AND gateq − erodeq CI lower bound > 0.
  **Kill:** gateq − T1 CI includes 0 → output-only gating is not a method on NFO (one failure-log
  line). Hard/easy split reported, but not used for the decision.
- **RESULT (pre-registered test above): KILL.**
  - Tuned on seq1: θ = 6, τ = 0.9. That τ gates ~80% of frames, so the variant is nearly ungated.
  - seq2–4 (decision set): gateq − T1 **−0.006 [−0.027, +0.008]** (better on 31/46). gateq − erodeq
    +0.010 [+0.006, +0.015].
  - All 61 starts: +0.001 [−0.014, +0.015].
  - Reading: the cue removes the right pixels relative to erosion, but output-only removal at stride-16
    cells does not beat T1. Neither does restricting it to low-confidence frames.

## Multi-level background cue (2026-10-12, 17 hard starts, t₀, inside the GT box)
- z of cos(frame features, warm-up background features) at each SAM2 decoder level:

  | Score | AUC | Precision at recall 0.9 | Best J |
  |---|---|---|---|
  | stride 4 | 0.864 | 0.263 | 0.327 |
  | stride 8 | 0.868 | 0.276 | 0.310 |
  | stride 16 | 0.946 | 0.462 | 0.520 |
  | z4 + z16 | 0.948 | 0.482 | 0.524 |
  | pixel difference | 0.862 | 0.283 | 0.399 |

- The fine levels behave like the pixel cue and add little to stride 16. For the adapter: inject the
  cue at stride 16 (dense prompt). Fine-level injection is an ablation only; boundaries come from
  SAM2's own stride-4/8 features. `MULTI=1 python -m benchmark.nfo_vos.diag_composite_diff`.

## PRE-REGISTERED (2026-10-12, written before running): uncertainty-weighted logit fusion (ViperSAM-style)
- **Method `t1-lgfuse`.** On T1's own logits L (T1-lg run, 300×400):
  L′ = L + λ·R·ν, with
  - R = binary entropy of σ(L) / ln 2: SAM2's per-pixel uncertainty, highest at boundaries;
  - ν = tanh((z − θ)/2): z is the stride-16 static-layer z map, bilinearly upsampled; ν is negative
    where the cell looks like the empty scene.

  Mask = L′ > 0 at native resolution. It can remove AND add pixels, but only where SAM2 is uncertain.
  Output-only.
- **Control `t1-lgconst`.** Same rule with ν ≡ −1 (shrink uncertain pixels everywhere, no cue). λ is
  tuned separately.
- **Tuning.** seq1 only, by mean J. θ ∈ {3, 4.5, 6}, λ ∈ {2, 4, 8, 16}.
- **Test.** seq2–4, ΔJ&F with cluster-bootstrap CI vs T1-lg (same run's masks).
- **Pass:** fuse − T1-lg ≥ +0.005 with CI lower bound > 0, AND fuse − const CI lower bound > 0.
  **Kill otherwise.**
- **RESULT (pre-registered logit fusion): KILL.**
  - Tuned on seq1: θ = 4.5, λ = 2, the bottom of the λ grid. seq1: +0.011 J over T1-lg.
  - seq2–4 (decision set): fuse − T1-lg **−0.009 [−0.012, −0.007]** (better on 11/46).
    fuse − constant-shrink control −0.010 [−0.015, −0.004].
  - All 61: −0.005. Hard and easy starts are equal (−0.005 each).
  - T1-lg (local rerun) vs cluster T1: −0.001 [−0.002, +0.000] on seq2–4.
  - Three training-free output-level uses of the static cue have now failed held out: gate,
    confidence gate, uncertainty-weighted logit fusion. The cue is real at t₀ (AUC 0.95) and picks
    better pixels than erosion, but post-hoc edits of SAMURAI's output do not convert it into J&F. What
    remains is learned decoder-level integration (master_thesis
    `docs/superpowers/plans/2026-10-12-static-cue-dense-prompt-adapter.md`), where the no-cue adapter
    control is decisive.

## PRE-REGISTERED (2026-10-12, written before running): adding missed visible pixels with a cleaned composite
- **Question.** Can frame-t evidence, compared against two references (the empty scene and a
  person-aligned cleaned composite of t−6…t−1), recover visible person pixels that T1 missed, without
  amodal extent? The framing is the layered model of Wang & Adelson (IEEE TIP 1994): each pixel is
  assigned to the layer that explains it better, either the static background layer (exact per pixel,
  static camera) or the moving person layer (the composite, with a small search window for alignment
  and limb error).
- **Data.** All 61 windows; frames t₀+6, +9, …, +36 (11 per window); T1 = cached cluster masks.
  Alignment = GT box centres (oracle; tracker alignment is a later arm).
- **Candidates.** Pixels in a 10 px ring around T1's mask, inside the GT box at t (dilated 10 px).
  Labels: pseudo-GT visible mask.
- **Composites.** raw recency; GT-free cleaned (each view weighted by sigmoid(z_pix − 3) against the
  warm-up background); oracle-cleaned (pseudo-GT visibility).
- **Scores** (higher = add):
  - z16 (stride-16 empty-scene cue);
  - z4 (the same on SAM2 stride-4 features);
  - corr4_C = max over a 7×7 cell window of cos(f₄,t, f₄,C);
  - layer_f4_C = log(1 − cos(f₄,t, f₄,bg)) − log(1 − corr4_C);
  - layer_pix_C = log(SSD_bg) − log(min over |δx| ≤ 6, |δy| ≤ 3 of SSD_C), with SSD over 5×5 grey patches;
  - pix_bg = log SSD_bg (background residual only).
- **Metrics.** Pooled precision at 50% reach (recall of missed GT pixels in the ring), and AUC.
  Mask-level: J of T1 ∪ {ring ∧ s > τ}, with τ per score tuned on seq1 frames. ΔJ vs T1 on seq2–4
  frames, cluster bootstrap by twin group, stratified by sequence. J at native resolution.
- **Pass:** the best GT-free score has precision ≥ 0.30 at 50% reach, AND ΔJ (seq2–4) > 0 with CI
  lower bound > 0. **Kill:** the oracle-cleaned layered score stays < 0.30 precision at 50% reach (no
  fine person evidence in the composite).
- **RESULT (pre-registered fine-addition test): KILL.**
  - 61 windows, 667 frames, 3.4 M ring pixels, 6.8% of them missed visible person.
  - Precision at 50% reach:

    | Score | Precision |
    |---|---|
    | best overall: z4 (stride-4 empty-scene cue, no composite) | 0.143 |
    | composite correspondence corr4 (raw / GT-free / oracle) | ≈ chance (AUC 0.47–0.51) |
    | layered, pixel space | 0.115–0.127 |
    | layered, stride-4 features | 0.110–0.117 |

  - Oracle cleaning moves the layered pixel score from 0.115 to 0.127, nowhere near the 0.30 bar.
  - Mask-level ΔJ on seq2–4: −0.0014 to 0.0000 for every arm. τ tuned on seq1 adds almost nothing.
- **Why (measured, interpretation not verified):** missed GT pixels hug T1's boundary: 51% lie within
  2 px of T1's mask, 66% within 4 px, 82% within 10 px. The pseudo-GT is 224 px upsampled, about
  3.6 native px per GT pixel. So most "missed visible pixels" are consistent with boundary
  quantization between T1's native mask and the coarse pseudo-GT, not real uncovered body parts.
  Hand masks at native resolution (the earlier hand-annotation proposal) would settle it.
- `diag_fine_add.py`.

## Native-resolution pseudo-GT and scoring — 2026-10-12
- **Pseudo-GT regenerated at native 800×600** (`gen_nfo_pseudo_masks --native-only`; 224 files kept).
  Native vs upsampled 224: IoU median 0.95 (p10 0.91), same area. GT loaders now prefer the native
  files. **Scoring is native by default**; the old 224 scores are in `per_*_224.csv`.
- **Native results** (61 starts):

  | Method | J&F native | J&F 224 | Mean J (both) | Init failures native / 224 |
  |---|---|---|---|---|
  | B0 | 0.816 | 0.828 | 0.761 | 17 / 17 |
  | T1 | 0.818 | 0.831 | 0.763 | 16 / 15 |
  | T1-om | 0.849 | 0.861 | 0.789 | 0 / 0 |

  - Per-frame J does change (mean |ΔJ| 0.010, max 0.40); the means cancel.
  - F drops by about 0.025: the boundary is now measured finely.
  - **All conclusions are unchanged:**

    | Comparison | ΔJ&F at native |
    |---|---|
    | T1 − B0 | +0.002 [−0.003, +0.011] |
    | oracle init (T1) | +0.031 [+0.013, +0.054] |
    | oracle init (B0) | +0.031 |
    | point-only, T1 | −0.016 |
    | point-only, B0 | −0.019 |

    T1 failures: 73% over-extent, 11% off-person; precision-filter ceiling 0.765 → 0.904.

## RAN (2026-10-13): depth ordering as a label-free gap map — FAILED the pre-set bar (AUC 0.672, +0.000 over z; diag_depth.py)

Z-buffer model, static camera, static occluder: x shows the person at t iff S_t(x) = 1 and Z_bg(x) > Z_p(t),
where S_t = full silhouette (extent), Z_bg = depth of the empty scene, Z_p = person depth. Motion is
almost purely horizontal (fronto-parallel), so Z_p(t) ≈ Z_p per segment, and the visible-capable region is
one threshold on one static map: G = {x : Z_bg(x) > Z_p}. It is the label-free version of the gap map Ŵ
in docs/hand_annotation_plan.md.
- Depth: Depth Anything V2 (relative disparity δ = a/Z + b, affine unknown per image). Ordering survives,
  but δ from different images is not comparable: align each frame's δ_t to δ_bg by a least-squares
  affine fit on pixels outside the GT box before comparing. δ_p = median δ_t over pseudo-GT pixels.
- Per-frame variant (replaces unknown S_t): visible_t(x) = G(x) ∧ |δ_t(x) − δ_p| < ε.
- The motion-aligned composite is not needed for δ_p (only a scalar is needed, and the composite is out of
  distribution for a depth net).
- Kill test (seq1 seg 4, inside GT boxes, the 31 hand frames once they exist; pseudo-GT before that):
  pixel AUC of δ_bg for visible vs not-visible. Continue only if AUC ≥ 0.85 AND it adds to the
  empty-scene z (logistic on z alone vs z + δ_bg, leave-frames-out, ΔAUC ≥ 0.02).
- Rhyme: a z AUC of 0.94 did not turn into J gains when gating T1 (−0.007 on seq2–4). Use depth for
  annotation auto-completion and as an adapter input, not as another output gate on T1.
- Risk: depth nets blur thin structures, so narrow gaps may inherit foliage depth.

## LATER (logged 2026-10-13): camera-jitter variant of the benchmark

Robustness test for every static-scene cue (empty-scene z, gap map, depth gap).
- V-ESAI (from the REDIR paper) applies one random affine per sample group: a viewpoint augmentation that
  leaves static-scene methods intact. We need **temporal** jitter: a smooth random trajectory per frame
  (e.g. an Ornstein–Uhlenbeck process on t_x, t_y, θ, log s), with magnitudes taken from real handheld
  footage. Warp frames, masks and boxes identically, and crop to the region valid in all frames.
- Limit: a 2-D warp simulates camera rotation and zoom only. Real camera translation adds parallax
  between near foliage and the person, which no single registration can undo. Flag this in the paper.
- Arms: raw jitter, and jitter + stabilisation (registration to the warm-up frames) as the do-nothing
  compensator. Expected: B0/T1 roughly unchanged; static cues degrade unless stabilised.

## First hand masks (seq1 seg 4, 20 frames, semi-modal) vs pseudo-GT — 2026-10-13

- 20 frames annotated in CVAT (16 drawn without prefill, 4 edited from the prefill: +438..826 / −185..1073 px).
  The annotator ignored thin branches over the body, so the masks are **semi-modal**.
- Pseudo-GT vs hand (mean over frames): J 0.527 / P 0.744 / R 0.652 on frames < 1230; J 0.693 / P 0.783 /
  R 0.855 on frames ≥ 1230.
- Disagreement goes both ways: 14.4k px only in hand, 20.7k px only in pseudo-GT. Of each, ~11% is in the feet
  band (bottom 1/6 of the box), ~36–41% within 3 px of the pseudo-GT edge, ~48–53% farther.
  Figure: images/benchmark/rr_checks/hand/hand_vs_pseudo.png.
- Not yet known: the annotator's own test–retest agreement (noise floor).

## Exemplar silhouettes as labels (exemplar_gt.py) — 2026-10-13, seq1 seg 4, NOT held out

Retrieve clear-frame silhouettes of the same person by partial evidence (pseudo-GT = visible; far + unchanged
= background through a gap), blend the top 3 by signed distance, remove near depth (> 0.2), union pseudo-GT.
vs brush hand masks (20 fr): J 0.582 / P 0.644 / R 0.850 (pseudo-GT 0.618 / 0.766 / 0.764).
vs polygon hand masks (3 fr): J 0.579 / P 0.759 / R 0.717 (pseudo-GT 0.500 / 0.852 / 0.555).
Caveat: the depth threshold (0.2 vs 0.3) was chosen on these same frames. A verdict needs held-out hand frames.
Walker polygon (rolling_polygon.py) and mean template (template_gt.py) were worse: J 0.41 / 0.33 vs brush.
v2 (head-complete exemplars, 4-band sideways alignment to evidence, positive evidence G = far-gap pixels changed
> 25 grey levels vs the empty scene), fixed before running, same frames (not held out):
| label | vs brush J / P / R (20 fr) | vs polygon J / P / R (3 fr) |
| pseudo-GT P | 0.618 / 0.766 / 0.764 | 0.500 / 0.852 / 0.555 |
| P u G | 0.636 / 0.744 / 0.814 | 0.551 / 0.845 / 0.623 |
| v2 (S - depth>0.2) u P u G | 0.594 / 0.648 / 0.881 | 0.604 / 0.747 / 0.758 |
30% of clear-frame pseudo-GT silhouettes lack the head (top 12% of box < 2% of mask). G alone: precision 0.87 vs
brush, covers 0.47 of it. G is the pixel empty-scene cue, so labels containing it favour cue-based methods.

## PRE-REGISTERED (2026-10-13, written before running): automatic extent-leaning labels vs polygon hand labels

Question: does a simple automatic label agree with the annotator's polygon labels better than the pseudo-GT, with
recall prioritised?
- Reference: polygon hand labels, drawn as convex hulls of clicked points (one hull per part: head+torso+arms,
  each leg; union). The 3 earlier free-hand polygons (1183, 1214, 1263) are a side check only.
- Tuning frames (seq1 seg 4, R walk): 1164, 1183, 1194, 1214, 1224, 1244, 1263, 1274, 1294.
- Held-out frames, not inspected before the freeze: seg 5 (L walk) 1490, 1520, 1550, 1580, 1610; seg 0 (R walk)
  30, 70, 110.
- Method (exemplar_gt.py v2 without G):
  1. bank = pseudo-GT of seq1 frames with visibility v >= v_min and head present (top 12% of box >= 2% of mask),
     cropped to the GT box, mirrored to one direction; excluded within ±10 frames of the target; hand masks never
     enter the bank;
  2. retrieve the top 3 by |S n P| / |P| - |S n N| / |S| (P = pseudo-GT; N = disparity < 0.1 and
     |frame - empty scene| < 8), aspect within 15%, shift ±10% of box width; blend by mean signed distance;
     4-band sideways alignment (±15%, bands with >= 20 evidence pixels only);
  3. remove pixels with empty-scene disparity > θ;
  4. label = cleaned silhouette ∪ P.
- Tuned on the tuning frames only: θ ∈ {0.15, 0.2, 0.3, none}, v_min ∈ {0.6, 0.8, clear frames}; max F₂.
- Primary criterion: F₂ (recall weighted 2×) vs the polygon labels, pooled mean over held-out frames.
  PASS: F₂(label) ≥ F₂(pseudo-GT) + 0.05 AND precision ≥ 0.60. J, P, R also reported.
- Ablations on the held-out frames (one change each): no band alignment; no depth cleaning; v_min = clear frames.
- Context: brush vs free-hand polygon J 0.58–0.69 on 3 frames = scale of the annotator's own variation.
AMENDMENT (2026-10-13, before running): reference = free-hand polygon labels (CVAT, mask type; *_handpoly.png),
not point hulls (three hulls were 1.3–1.7× the polygon area on 3 frames). Empty scene for N = median of the target
segment's warm-up frames (seg 0/4: 946–1113, seg 5: 1349–1435). Code: benchmark/nfo_vos/extent_label_test.py.
RESULT (2026-10-13): PASS by the pre-registered rule.
- Tuning (seg 4, 9 frames) selected theta = none (no depth cleaning) and v_min = clear frames (122 head-complete
  exemplars); F2 0.776 vs pseudo-GT 0.639.
- Held-out (seg 5 + seg 0, 8 frames), mean: label F2 0.798 / J 0.615 / P 0.686 / R 0.835; pseudo-GT F2 0.706 /
  J 0.641 / P 0.920 / R 0.682. ΔF2 +0.093, 95% bootstrap CI over frames [+0.012, +0.203]; precision 0.686 ≥ 0.60.
- The gain sits in heavily occluded frames (1610: 0.658 vs 0.228; 30: 0.557 vs 0.421; 1520: 0.821 vs 0.677);
  easy frames tie. J is 0.026 lower: the label trades precision for recall, as intended.
- Ablations on held-out: no band alignment F2 0.800; no depth cleaning and v_min = clear are identical to the
  frozen label (both were already its setting). Depth and band alignment contribute nothing measurable.
- So the simplest version suffices: label = signed-distance blend of the 3 best head-complete clear-frame
  silhouettes (retrieved by pseudo-GT coverage minus "background seen through a gap") ∪ pseudo-GT.
- Limits: 8 held-out frames (CI lower bound below the +0.05 margin), one annotator, one sequence.
  Figure: images/benchmark/rr_checks/hand/extent_label_heldout.png.

## Reference: humans outline occluded objects near-amodally, and consistently (logged 2026-10-13)

Zhu, Tian, Metaxas & Dollár, "Semantic Amodal Segmentation", CVPR 2017 (arXiv 1509.01329): annotators asked
to outline occluded objects draw amodal (full-extent) regions, with high agreement between annotators. This
supports treating our polygon labels as near-amodal and SAM2 pseudo-GT as modal. Our annotator tried to stay
modal but still drew extent-leaning outlines (polygon ⊇ brush, 86–94% containment, 10–46% larger).

## Label robustness: do conclusions depend on the label? (label_robustness.py) — 2026-10-13, seq1 only

Mean region J (native). Labels: pseudo-GT / silhouette label (L = (silhouette − disparity > 0.2) ∪ pseudo-GT),
over 15 seq1 windows; and on hand frames inside windows: polygons (n = 13) / brush (n = 14).
| method | pseudo-GT | silhouette | polygon | brush |
| b0 | 0.681 | 0.593 | 0.528 | 0.511 |
| t1 | 0.699 | 0.609 | 0.555 | 0.563 |
| b0-om | 0.757 | 0.640 | 0.567 | 0.580 |
| t1-om | 0.764 | 0.650 | 0.583 | 0.588 |
| b0-pt | 0.693 | 0.587 | 0.502 | 0.552 |
| t1-pt | 0.697 | 0.592 | 0.499 | 0.551 |
Differences (95% bootstrap CI over windows / frames; no twin clustering, indicative):
- T1 − B0: +0.018 [−0.001, +0.048] / +0.016 [−0.001, +0.039] / +0.027 [+0.000, +0.074] / +0.052 [−0.000, +0.122]
- T1-om − T1: +0.064 [+0.006, +0.145] / +0.041 [+0.004, +0.097] / +0.028 [−0.028, +0.098] / +0.025 [−0.045, +0.102]
- B0-om − B0: +0.076 [+0.008, +0.160] / +0.046 [+0.001, +0.104] / +0.040 [−0.013, +0.105] / +0.069 [+0.018, +0.133]
Reading: the sign of every main comparison is the same under all four labels (oracle init helps; T1 ≥ B0 by a
small margin). Absolute J drops by ~0.1 under the silhouette label and ~0.15 under hand labels. Gaps shrink under
the extent-leaning labels, and on 13–14 hand frames the oracle-init CIs include 0. Point vs box prompt is the one
ordering that flips (pt below box on pseudo-GT/polygon, above B0 on brush): small n, not interpreted.

### Where the gap to the hand labels sits: precision vs recall (2026-10-13, ad hoc, seq1 windows)

Same frames as the robustness table. Mean per frame of J / P / R / area(pred)/area(hand); "pseudo" = the pseudo-GT
itself scored against the hand label (= the label generator, non-causal multi-anchor SAM2).

| vs polygon (n=13) | J | P | R | area ratio | vs brush (n=14) | J | P | R | area ratio |
|---|---|---|---|---|---|---|---|---|---|
| b0     | 0.528 | 0.721 | 0.573 | 1.12 | | 0.511 | 0.609 | 0.700 | 1.86 |
| t1     | 0.555 | 0.754 | 0.629 | 1.13 | | 0.563 | 0.668 | 0.793 | 1.83 |
| b0-om  | 0.567 | 0.910 | 0.600 | 0.66 | | 0.580 | 0.850 | 0.662 | 0.80 |
| t1-om  | 0.583 | 0.908 | 0.618 | 0.68 | | 0.588 | 0.844 | 0.674 | 0.81 |
| b0-pt  | 0.502 | 0.935 | 0.526 | 0.57 | | 0.552 | 0.860 | 0.629 | 0.75 |
| t1-pt  | 0.499 | 0.933 | 0.522 | 0.56 | | 0.551 | 0.862 | 0.624 | 0.74 |
| pseudo | 0.606 | 0.889 | 0.653 | 0.73 | | 0.607 | 0.781 | 0.731 | 0.94 |

Reading: box-init errors are mixed (leaks: P 0.72-0.75 at ~polygon-sized area). Oracle init removes the leaks
(P -> 0.91) but does not raise recall (0.57-0.63 -> 0.60-0.62), hence the small J gain. Oracle-init methods sit within
0.02 J of the pseudo-GT itself; the remaining gap to the polygon is recall (~0.35-0.40 of polygon area), i.e. the
modal/near-amodal gap plus missed fragments (not separable without per-frame visibility). Area ratios are means of
per-frame ratios (skewed by a few leaking frames). seq1 only, one annotator.
