# NFO VOS pilot: what worked (2026-10-05)

> **Naming (2026-10-14):** "SAM2-ceiling" = the label formerly called pseudo-GT (gen_nfo_pseudo_masks.py: SAM2.1-L,
> GT box + centre point at confirmed-clear checkpoints, propagated both ways, union, clipped to the GT box, thin
> components removed; files `*_sammask*.png`). It is an oracle-prompted SAM2 reference, not ground truth.


Pilot: seq1 segments 4–5, 21 starts, 51-frame windows, SAM2-ceiling at 224 (spec
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
- Upper bound only: the GT mask is the 224 SAM2-ceiling mapped to native.

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
   person extent/localisation (integration helps; scored on human boxes, so no SAM2-ceiling bias;
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
  Labels: SAM2-ceiling visible mask.
- **Composites.** raw recency; GT-free cleaned (each view weighted by sigmoid(z_pix − 3) against the
  warm-up background); oracle-cleaned (SAM2-ceiling visibility).
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
  2 px of T1's mask, 66% within 4 px, 82% within 10 px. The SAM2-ceiling is 224 px upsampled, about
  3.6 native px per GT pixel. So most "missed visible pixels" are consistent with boundary
  quantization between T1's native mask and the coarse SAM2-ceiling, not real uncovered body parts.
  Hand masks at native resolution (the earlier hand-annotation proposal) would settle it.
- `diag_fine_add.py`.

## Native-resolution SAM2-ceiling and scoring — 2026-10-12
- **SAM2-ceiling regenerated at native 800×600** (`gen_nfo_pseudo_masks --native-only`; 224 files kept).
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
  affine fit on pixels outside the GT box before comparing. δ_p = median δ_t over SAM2-ceiling pixels.
- Per-frame variant (replaces unknown S_t): visible_t(x) = G(x) ∧ |δ_t(x) − δ_p| < ε.
- The motion-aligned composite is not needed for δ_p (only a scalar is needed, and the composite is out of
  distribution for a depth net).
- Kill test (seq1 seg 4, inside GT boxes, the 31 hand frames once they exist; SAM2-ceiling before that):
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

## First hand masks (seq1 seg 4, 20 frames, semi-modal) vs SAM2-ceiling — 2026-10-13

- 20 frames annotated in CVAT (16 drawn without prefill, 4 edited from the prefill: +438..826 / −185..1073 px).
  The annotator ignored thin branches over the body, so the masks are **semi-modal**.
- SAM2-ceiling vs hand (mean over frames): J 0.527 / P 0.744 / R 0.652 on frames < 1230; J 0.693 / P 0.783 /
  R 0.855 on frames ≥ 1230.
- Disagreement goes both ways: 14.4k px only in hand, 20.7k px only in SAM2-ceiling. Of each, ~11% is in the feet
  band (bottom 1/6 of the box), ~36–41% within 3 px of the SAM2-ceiling edge, ~48–53% farther.
  Figure: images/benchmark/rr_checks/hand/hand_vs_pseudo.png.
- Not yet known: the annotator's own test–retest agreement (noise floor).

## Exemplar silhouettes as labels (exemplar_gt.py) — 2026-10-13, seq1 seg 4, NOT held out

Retrieve clear-frame silhouettes of the same person by partial evidence (SAM2-ceiling = visible; far + unchanged
= background through a gap), blend the top 3 by signed distance, remove near depth (> 0.2), union SAM2-ceiling.
vs brush hand masks (20 fr): J 0.582 / P 0.644 / R 0.850 (SAM2-ceiling 0.618 / 0.766 / 0.764).
vs polygon hand masks (3 fr): J 0.579 / P 0.759 / R 0.717 (SAM2-ceiling 0.500 / 0.852 / 0.555).
Caveat: the depth threshold (0.2 vs 0.3) was chosen on these same frames. A verdict needs held-out hand frames.
Walker polygon (rolling_polygon.py) and mean template (template_gt.py) were worse: J 0.41 / 0.33 vs brush.
v2 (head-complete exemplars, 4-band sideways alignment to evidence, positive evidence G = far-gap pixels changed
> 25 grey levels vs the empty scene), fixed before running, same frames (not held out):
| label | vs brush J / P / R (20 fr) | vs polygon J / P / R (3 fr) |
| SAM2-ceiling P | 0.618 / 0.766 / 0.764 | 0.500 / 0.852 / 0.555 |
| P u G | 0.636 / 0.744 / 0.814 | 0.551 / 0.845 / 0.623 |
| v2 (S - depth>0.2) u P u G | 0.594 / 0.648 / 0.881 | 0.604 / 0.747 / 0.758 |
30% of clear-frame SAM2-ceiling silhouettes lack the head (top 12% of box < 2% of mask). G alone: precision 0.87 vs
brush, covers 0.47 of it. G is the pixel empty-scene cue, so labels containing it favour cue-based methods.

## PRE-REGISTERED (2026-10-13, written before running): automatic extent-leaning labels vs polygon hand labels

Question: does a simple automatic label agree with the annotator's polygon labels better than the SAM2-ceiling, with
recall prioritised?
- Reference: polygon hand labels, drawn as convex hulls of clicked points (one hull per part: head+torso+arms,
  each leg; union). The 3 earlier free-hand polygons (1183, 1214, 1263) are a side check only.
- Tuning frames (seq1 seg 4, R walk): 1164, 1183, 1194, 1214, 1224, 1244, 1263, 1274, 1294.
- Held-out frames, not inspected before the freeze: seg 5 (L walk) 1490, 1520, 1550, 1580, 1610; seg 0 (R walk)
  30, 70, 110.
- Method (exemplar_gt.py v2 without G):
  1. bank = SAM2-ceiling of seq1 frames with visibility v >= v_min and head present (top 12% of box >= 2% of mask),
     cropped to the GT box, mirrored to one direction; excluded within ±10 frames of the target; hand masks never
     enter the bank;
  2. retrieve the top 3 by |S n P| / |P| - |S n N| / |S| (P = SAM2-ceiling; N = disparity < 0.1 and
     |frame - empty scene| < 8), aspect within 15%, shift ±10% of box width; blend by mean signed distance;
     4-band sideways alignment (±15%, bands with >= 20 evidence pixels only);
  3. remove pixels with empty-scene disparity > θ;
  4. label = cleaned silhouette ∪ P.
- Tuned on the tuning frames only: θ ∈ {0.15, 0.2, 0.3, none}, v_min ∈ {0.6, 0.8, clear frames}; max F₂.
- Primary criterion: F₂ (recall weighted 2×) vs the polygon labels, pooled mean over held-out frames.
  PASS: F₂(label) ≥ F₂(SAM2-ceiling) + 0.05 AND precision ≥ 0.60. J, P, R also reported.
- Ablations on the held-out frames (one change each): no band alignment; no depth cleaning; v_min = clear frames.
- Context: brush vs free-hand polygon J 0.58–0.69 on 3 frames = scale of the annotator's own variation.
AMENDMENT (2026-10-13, before running): reference = free-hand polygon labels (CVAT, mask type; *_handpoly.png),
not point hulls (three hulls were 1.3–1.7× the polygon area on 3 frames). Empty scene for N = median of the target
segment's warm-up frames (seg 0/4: 946–1113, seg 5: 1349–1435). Code: benchmark/nfo_vos/extent_label_test.py.
RESULT (2026-10-13): PASS by the pre-registered rule.
- Tuning (seg 4, 9 frames) selected theta = none (no depth cleaning) and v_min = clear frames (122 head-complete
  exemplars); F2 0.776 vs SAM2-ceiling 0.639.
- Held-out (seg 5 + seg 0, 8 frames), mean: label F2 0.798 / J 0.615 / P 0.686 / R 0.835; SAM2-ceiling F2 0.706 /
  J 0.641 / P 0.920 / R 0.682. ΔF2 +0.093, 95% bootstrap CI over frames [+0.012, +0.203]; precision 0.686 ≥ 0.60.
- The gain sits in heavily occluded frames (1610: 0.658 vs 0.228; 30: 0.557 vs 0.421; 1520: 0.821 vs 0.677);
  easy frames tie. J is 0.026 lower: the label trades precision for recall, as intended.
- Ablations on held-out: no band alignment F2 0.800; no depth cleaning and v_min = clear are identical to the
  frozen label (both were already its setting). Depth and band alignment contribute nothing measurable.
- So the simplest version suffices: label = signed-distance blend of the 3 best head-complete clear-frame
  silhouettes (retrieved by SAM2-ceiling coverage minus "background seen through a gap") ∪ SAM2-ceiling.
- Limits: 8 held-out frames (CI lower bound below the +0.05 margin), one annotator, one sequence.
  Figure: images/benchmark/rr_checks/hand/extent_label_heldout.png.

## Reference: humans outline occluded objects near-amodally, and consistently (logged 2026-10-13)

Zhu, Tian, Metaxas & Dollár, "Semantic Amodal Segmentation", CVPR 2017 (arXiv 1509.01329): annotators asked
to outline occluded objects draw amodal (full-extent) regions, with high agreement between annotators. This
supports treating our polygon labels as near-amodal and SAM2 SAM2-ceiling as modal. Our annotator tried to stay
modal but still drew extent-leaning outlines (polygon ⊇ brush, 86–94% containment, 10–46% larger).

## Label robustness: do conclusions depend on the label? (label_robustness.py) — 2026-10-13, seq1 only

Mean region J (native). Labels: SAM2-ceiling / silhouette label (L = (silhouette − disparity > 0.2) ∪ SAM2-ceiling),
over 15 seq1 windows; and on hand frames inside windows: polygons (n = 13) / brush (n = 14).
| method | SAM2-ceiling | silhouette | polygon | brush |
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
ordering that flips (pt below box on SAM2-ceiling/polygon, above B0 on brush): small n, not interpreted.

### Where the gap to the hand labels sits: precision vs recall (2026-10-13, ad hoc, seq1 windows)

Same frames as the robustness table. Mean per frame of J / P / R / area(pred)/area(hand); "pseudo" = the SAM2-ceiling
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
0.02 J of the SAM2-ceiling itself; the remaining gap to the polygon is recall (~0.35-0.40 of polygon area), i.e. the
modal/near-amodal gap plus missed fragments (not separable without per-frame visibility). Area ratios are means of
per-frame ratios (skewed by a few leaking frames). seq1 only, one annotator.

### PRE-REGISTERED: pixel-level background suppression of T1 (2026-10-13, written before running)

T1 (SAMURAI, box start) masks with pixels where |frame - empty-scene median| < tau deleted. tau from {4, 8, 12, 16, 24}
chosen on seq1 by window J vs SAM2-ceiling; applied unchanged to seq2-4. PASS: seq2-4 J gain > 0, 95% window-bootstrap CI
excluding 0. Reference: oracle-init gain. Script: benchmark/nfo_vos/bg_suppress.py.

Decision (user, 2026-10-13): benchmark windows must not start on a confirmed-clear frame (11/61 currently do, grid
placement trials.py:161). Window redesign (starts only inside occluded stretches) pending; gains are additionally
reported on the 50 non-clear-start windows.

**RESULT: FAIL.** seq1 J falls monotonically with tau (0.699 plain T1 -> 0.657 / 0.614 / ... / 0.479 at tau 4..24), so
tau = 4 (smallest). seq2-4 (46 windows): J(T1+bg) - J(T1) = -0.040 [-0.048, -0.032]; non-clear-start windows (37):
-0.036 [-0.044, -0.029]. vs seq1 polygons: P 0.754 -> 0.775, R 0.629 -> 0.573. Per-pixel intensity change removes
person pixels (grayscale person ~ background intensity) faster than leak pixels. Does not kill the feature-level
per-cell change detector (features carry texture, not just intensity), but the pixel shortcut is dead.
Side result: on seq2-4, T1 - B0 = -0.003 [-0.008, +0.002] and oracle-init gain +0.015 [+0.003, +0.027]: much smaller
than on seq1 (+0.018, +0.064). The baselines barely separate outside seq1.

## Test (a), 2026-10-14: amodal video model (Diffusion-VAS) on SAM2-ceiling vs hand polygons — FAIL
- **Pre-registered** (in `amodal_dvas.py` docstring, before running): Diffusion-VAS amodal stage (Chen, Ramanan &
  Khurana, CVPR 2025, arXiv 2412.04623; code @ bbea113, weights kaihuac/diffusion-vas-amodal-segmentation), input =
  SAM2-ceiling modal masks + Depth Anything V2-L depth, 25-frame clips at stride 3 (25 fps footage, model trained at
  8 fps). A = prediction ∪ SAM2-ceiling at native res. Pass: J(A) > J(SAM2-ceiling) vs polygons (paired CI > 0) and P ≥ 0.75.
- **Result** (17 seq1 polygon frames × 3 seeds):

  | | J | P | R |
  |---|---|---|---|
  | SAM2-ceiling | 0.597 | 0.892 | 0.642 |
  | amodal A | 0.580 | 0.642 | 0.838 |

  ΔJ −0.017 [−0.061, +0.034], better on 6/17; seed spread 0.015 J. Only 32% of the added pixels lie inside the
  polygon; added area 1.64× SAM2-ceiling. Demo-faithful low-res union: J 0.577 (deviation irrelevant).
- Gain does not track how incomplete the SAM2-ceiling is (Spearman ρ = −0.22 between SAM2-ceiling recall and ΔJ, p = 0.41);
  big gains only where SAM2-ceiling nearly vanished (1610: R 0.19 → J 0.19 → 0.43).
- Cost: 23–25 s per 25-frame clip (25 denoising steps × CFG), 6.2 GiB peak.
- Fourth mask-widening approach with the same signature (recall up, precision down more). Visual: 
  `images/benchmark/rr_checks/amodal/dvas_vs_polygon.png`.

## Gap granulometry, 2026-10-14: is polygon ∖ SAM2-ceiling made of thin structures?
- Hypothesis (user): P ≈ S ∧ ¬γ_r(O), M ≈ S ∧ ¬O, so G = P ∖ M ≈ thin occluders (void candidates).
  Pattern spectrum f(r) = 1 − |γ_r(G)|/|G| (Maragos, TPAMI 1989). Pre-registered bar: f(3) ≥ 0.5.
- Result (17 seq1 polygon frames, 24 114 gap px): f(3) = **0.491**, just under the bar (not supported as stated).
  Split by the convex hull of M:

  | region | share of gap | f(1) | f(3) | f(6) |
  |---|---|---|---|---|
  | inside hull(M) (enclosed by the person) | 40% | 0.29 | 0.70 | 0.96 |
  | outside hull(M) (outline overshoot / missed parts) | 60% | 0.11 | 0.45 | 0.69 |
  | reference: M itself | | 0.00 | 0.02 | 0.11 |

- Reading: the enclosed gap is thin (consistent with the hypothesis); most of the gap is outside the person's hull,
  thinner slivers there are consistent with polygon-outline imprecision, and thick outside chunks (1164: f(3) 0.11;
  1610: 0.06) are whole parts the SAM2-ceiling missed. Not checked: whether enclosed thin gaps are branches or
  background between limbs; the SAM2-ceiling min-width filter (H2) also creates thin gaps from deleted thin person parts.
- `gap_granulometry.py`; visual `images/benchmark/rr_checks/gap_granulometry/gap_by_thickness.png`.

## H2 RESULT, 2026-10-14: SAM2 itself drops sub-cell visible fragments; our 14 px filter is not the cause
- Pre-registered in `label_generators_design.md:58-66`. Raw SAM2 union re-run on seq1 segs 0/4/5 with the
  thin-component filter off (`--min-width-scale 0 --native-tag sammask_nofilt_native`; production filter = bbox
  width < 14 native px). Filter effect isolated as U vs filter(U) from the same run (the re-run differs from the
  stored SAM2-ceiling by SAM2 run-to-run variation: 9/428 frames identical, mean |xor| 76 px).
- Per brush (modal) component, seg4, 20 frames, 94 components (74 detached); 1 cell = 117 native px²:

  | area (cells) | n | recall, filter off | recall, filter on |
  |---|---|---|---|
  | < 0.5 | 14 | 0.00 | 0.00 |
  | 0.5–1 | 5 | 0.33 | 0.26 |
  | 1–4 | 31 | 0.51 | 0.45 |
  | ≥ 4 | 44 | 0.79 | 0.77 |

  Slope of recall per doubling of area +0.086 [+0.078, +0.100] (frame-cluster bootstrap); detached sub-cell recall
  0.09 (n = 19). **Neither kill condition met: H2 supported.**
- vs polygons (17 frames): filter off J 0.608 vs on 0.599. The thin enclosed gap (person between branches,
  gap_granulometry.py red pixels, 6 768 px) is recovered 4% by raw SAM2: SAM2 never predicts it.
- Interpretation (consistent with, not proven): a resolution limit of the stride-16 decoder. Implication for any
  fix, including per-cell change detection at stride 16: it inherits the same limit; evidence must come from
  stride-4/8 features or pixels.
- `h2_fragments.py`; visual `images/benchmark/rr_checks/h2/filter_removed.png`.

## Composite gating tests, 2026-10-14 (`composite_tests.py`, bars fixed in its docstring before running)
- **Test 1 (fine evidence vs brush, 18 seg-4 frames; candidates = GT box ∖ SAM2-ceiling, 317 k px, 4.3% person,
  411 px in sub-cell brush components): PASS.** AUC all / sub-cell:

  | score | all | sub-cell |
  |---|---|---|
  | pix_bg (5×5 SSD vs empty scene) | 0.905 | **0.941** |
  | layer_pix_gtfree | 0.871 | 0.915 |
  | z4 (stride-4 features vs empty scene) | 0.848 | 0.892 |
  | layer_f4_gtfree | 0.830 | 0.871 |
  | z16 | 0.800 | 0.723 |
  | corr4 (composite correspondence), any composite | 0.46–0.50 | 0.39–0.52 |

  The same scores were killed on 2026-10-12 against the coarse SAM2-ceiling (precision 0.11–0.14 at 50% reach): consistent
  with that kill being a label artefact. The composite adds nothing beyond the empty-scene cue (correspondence at
  chance; layered < pix_bg). Not measured: precision at useful recall (4% positives; AUC alone can hide low precision);
  sub-cell positives come from few components.
- **Test 2 (SAM2 image predictor on the GT-free composite, box + centre point): FAIL.** vs polygons (17):
  SAM2-ceiling J/P/R 0.597/0.892/0.642; M_C 0.485/0.527/0.872; M ∪ M_C 0.512 (ΔJ −0.085 [−0.151, −0.014]);
  S1 prompt candidate M_C ∧ changed cells 0.493. vs brush ΔJ −0.172. The composite mask leaks (precision 0.53):
  same signature as the amodal box prompt (T4c-1). S1 in this form is dead.
- Implication: sub-cell evidence exists at pixel/stride-4 level from the empty scene alone; SAM2-ceiling-based scoring
  cannot reward recovering it (the SAM2-ceiling lacks those pieces), so S3 must be evaluated on human labels.
- **Test 2b (composite mask as extent envelope + empty-scene depth filter, `composite_envelope.py`): FAIL.**
  V̂ = (M_C ∧ disparity ≤ 0.2) ∪ M vs polygons (17): J/P/R 0.581/0.691/0.787 vs SAM2-ceiling 0.597/0.892/0.642,
  ΔJ −0.016 [−0.070, +0.040] (7/17 better). vs brush (18): ΔJ −0.069 [−0.114, −0.022]. The depth filter raises
  M_C's precision 0.53 → 0.67 but cuts recall 0.87 → 0.64. Containment R(M_C, polygon) 0.87; ≥ 0.9 on 5/17 frames.
- Pattern across today's and yesterday's labels vs polygons: SAM2-ceiling 0.597, Diffusion-VAS 0.580, V̂ 0.581,
  exemplar silhouette v2 0.604 (3 fr), filter-off SAM2 0.608: all J ≈ 0.58–0.61 with different P/R. Consistent with an
  annotator-noise ceiling against a single annotator; the inter-annotator J would test this.
- **λ₂ gate (frame-vs-composite match inside the envelope M_C, `composite_match.py`): FAIL.** 18 brush frames,
  150 k px inside M_C (47% visible person). Pooled AUC: z16 (frame vs empty scene) 0.823; match16 0.707; match8
  0.559; match4 0.602; far 0.712. Leave-one-frame-out logistic: z16 0.815, z16+match16 0.814, z16+far 0.841,
  z16+far+match16 0.838. The match carries no information beyond the empty-scene term (inside the envelope, "differs
  from the empty scene" ≈ "visible person": the person is the only moving thing). Not excluded: SAM2's learned q·k
  projections in memory attention using the composite differently than backbone cosine; tracker (non-oracle) alignment.

## V = S ∧ ¬O at pixel level, no learning: FIRST LABEL TO BEAT THE SAM2-CEILING vs POLYGONS — 2026-10-14
- `factorised_label.py` (fixed before running). S = M_C (SAM2 image predictor on the GT-free composite; oracle
  GT-box alignment and box prompt); ¬O = pixel change vs the empty scene (pix_bg = log 5×5 SSD), threshold τ chosen
  leave-one-frame-out on the brush frames (τ = 6.31 ≈ median of pix_bg inside M_C; LOFO range 6.31–6.99).
  V̂ = (M_C ∧ pix_bg > τ) ∪ M.

  | vs | label | J | P | R |
  |---|---|---|---|---|
  | polygons (17) | SAM2-ceiling M | 0.597 | 0.892 | 0.642 |
  | | M_C ∧ change (prompt alone) | 0.568 | 0.860 | 0.624 |
  | | **V̂** | **0.680** | 0.835 | 0.778 |
  | brush (18) | SAM2-ceiling M | 0.643 | 0.773 | 0.795 |
  | | **V̂** | 0.665 | 0.734 | 0.874 |

  ΔJ vs polygons **+0.083 [+0.036, +0.135], 13/17 better: PASS**; vs brush +0.022 [−0.002, +0.052], 9/18.
  Stride-4 features (z4) instead of pixels: +0.044 [+0.010, +0.093] vs polygons, +0.010 vs brush.
- Why it works where every widening failed: the envelope supplies extent (S), the empty-scene change carves the
  occluders (¬O) at pixel resolution; precision drops only 0.89 → 0.84 while recall rises 0.64 → 0.78.
- Caveats: one sequence, one annotator; τ tuned on seg-4 brush frames (polygon frames in segs 0/5 are cross-segment,
  not cross-sequence held out); composite alignment and SAM2 prompt use GT boxes (oracle); M is the SAM2-ceiling, not a
  tracker's mask. Next: replace M by T1's own mask and GT alignment by tracker alignment.
- Visual: `images/benchmark/rr_checks/composite_tests/factorised_added.png`.
- **V̂ on top of the trackers (`factorised_t1.py`, τ = 6.31 fixed from above, not re-tuned): PASS.**
  V̂_X = (M_C ∧ pix_bg > τ) ∪ X, seq1 hand frames inside trial windows:

  | vs | method | J | P | R |
  |---|---|---|---|---|
  | polygons (14) | B0 | 0.516 | 0.711 | 0.568 |
  | | V̂_B0 | 0.588 | 0.705 | 0.762 |
  | | T1 | 0.542 | 0.741 | 0.621 |
  | | **V̂_T1** | **0.602** | 0.720 | 0.779 |
  | | SAM2-ceiling | 0.590 | 0.884 | 0.637 |
  | brush (13) | T1 | 0.597 | 0.710 | 0.781 |
  | | V̂_T1 | 0.619 | 0.675 | 0.880 |

  ΔJ vs polygons: V̂_T1 − T1 **+0.060 [+0.031, +0.091], 12/14**; V̂_B0 − B0 +0.071 [+0.037, +0.107].
  vs brush: V̂_T1 − T1 +0.022 [+0.002, +0.044]; V̂_B0 − B0 +0.042 [+0.009, +0.080].
  Recall +0.16 for −0.02 precision. Still oracle: composite alignment and the SAM2 prompt on C use GT boxes.
  A SAM2-ceiling-scored benchmark would count these additions as false positives (the ceiling lacks them):
  evaluation of this method needs human labels on seq2–4.
- **CORRECTION (same day):** the GT-free table below was contaminated: on 7/19 frames the blob tracker's chain had a gap,
  giving a NaN anchor (no blob box, NaN blob point). Fixed (tracker's own prediction on gaps); corrected numbers in the
  next bullet. The T1-alignment/GT-box/T1-prompt rows were unaffected.
- **GT-free V̂_T1 (`factorised_gtfree.py`, `factorised_extras.py`): PASS.** Alignment of the composite from (a) the
  independent blob tracker (Theil–Sen on its chain, x-only; median |dx| error vs GT shifts 8.2 px) or (b) T1 mask
  centroids (4.6 px). Prompt on the composite: box or point from the blob tracker or T1. ΔJ(V̂_T1 − T1) vs polygons,
  12 frames with history inside the window (7 for blob_box, which exists on 9/12 frames):

  | alignment / prompt | vs brush | vs polygons |
  |---|---|---|
  | GT / GT box (oracle ref) | +0.023 | +0.062 [+0.030, +0.098] |
  | blob / GT box | +0.017 | +0.044 |
  | T1 / GT box | +0.009 | +0.050 |
  | **T1 / blob box (chosen on brush)** | +0.022 | **+0.053 [+0.025, +0.082]**, 7/7 |
  | blob / blob point | +0.018 | +0.036 [+0.011, +0.064] |
  | T1 / T1 box | +0.016 | +0.030 [+0.012, +0.050] |
  | T1 / T1 point | +0.012 | +0.028 |

  Fallback prompt (blob box, else blob point), all 12 frames: T1 alignment +0.031 [+0.011, +0.055] vs polygons,
  +0.016 [+0.005, +0.030] vs brush. Strata (pooled polygon+brush, GT ref vs chosen arm): good T1 init +0.032 / +0.030;
  bad T1 init +0.076 / +0.047 — larger gains where T1 starts badly.
- **Transferable threshold:** per-frame robust z of pix_bg, τ_z = 7.5 (chosen on brush, interior of a 0–12 grid):
  J 0.554 vs 0.556 with raw τ = 6.31 (vs polygons, 12 frames): equivalent; z is used for other sequences.
- `run_vhat.py`: GT-free V̂_T1 masks for all 61 windows (masks/t1-vhat/), configuration frozen before any seq2–4
  frame is scored.
- **Corrected GT-free results** (NaN-anchor fix), ΔJ(V̂_T1 − T1) vs polygons (12 frames): blob/blob point **+0.042
  [+0.018, +0.068], 9/12 (chosen on brush, +0.016 vs brush)**; blob/blob box +0.037; T1/blob box +0.039; T1/blob point
  +0.032; blob/GT box +0.044; T1/GT box +0.050; GT/GT box (oracle) +0.062. Strata (pooled): bad T1 init +0.052, good
  +0.021 for the chosen arm. Robust-z threshold for the chosen arm: τ_z = 7.5, J 0.560 vs 0.567 raw (equivalent),
  +0.035 [+0.012, +0.062] over T1. `run_vhat.py` re-frozen to blob alignment + blob point + τ_z 7.5 and re-run.
  Open: with the robust z, seg-0 windows (empty scene from frames 946–1113, far in time) get no additions.

## S1 sanity check (causal re-prompting of SAMURAI with V̂), 2026-10-14: the injected extent does not persist
- `s1_reprompt.py`: propagate T1 to frame j, V̂_j = (M_C ∧ pix_bg > 6.31) ∪ own mask (GT-free composite, blob
  tracker), add_new_mask(j) as a fully trusted non-conditioning memory (SAMURAI cannot take mid-video conditioning
  frames), continue. Every 20 (j = 20) or every 10 (j = 10, 20, 30) frames; seq1 windows with hand frames.
- **After the re-prompt the mask snaps back:** median area S1/T1 on later frames = 1.00 in 17/18 window-arms
  (one window 1.61×). Before the re-prompt the run reproduces stored T1 to 0.1–3.6% |xor| (one window 11.5%:
  run-to-run variation).
- On non-prompted hand frames after the re-prompt (n = 5–8): ΔJ S1 − T1 −0.063 / −0.004 vs polygons (every 20 /
  every 10), −0.128 / −0.062 vs brush; post-hoc V̂ is better than S1 in every comparison (CI upper bound < 0).
  Small n, differences dominated by run variation and one window; the clear result is the area ratio.
- Consistent with the resolution argument: the added pixels are mostly sub-cell fragments; a mask prompt and
  memory both act at stride 16, and SAM2 re-decodes each frame from its own image features. Training-free
  injection at prompt/memory level does not carry pixel-level additions; post-hoc addition (or a learned
  stride-4 path) does.

## PRE-REGISTERED seq2–4 hand-polygon frames (2026-10-14, chosen before any seq2–4 method output was looked at)
Rule: frame at local index 20 of each admissible window (history and the V̂ envelope available; independent of
any method's output). Tier A = every other window starting with the first, labelled first; tier B = the rest.
Upload folders: `data/cvat/upload/<seq>_poly_<tier>/images` (file name = real frame number).

| seq | tier | n | frames |
|---|---|---|---|
| seq2 | A | 8 | 52, 132, 335, 497, 577, 1163, 1338, 1722 |
| seq2 | B | 8 | 92, 295, 375, 537, 1123, 1203, 1540, 1921 |
| seq3 | A | 7 | 59, 287, 367, 615, 873, 1110, 1498 |
| seq3 | B | 7 | 99, 327, 575, 833, 913, 1303, 1694 |
| seq4 | A | 8 | 43, 123, 293, 486, 566, 717, 922, 1284 |
| seq4 | B | 8 | 83, 253, 333, 526, 677, 757, 1081, 1448 |

Second annotator: the same tier-A frames, imported with tag `handpoly2`. Polygon protocol as for seq1.
Import: `python -m benchmark.nfo_vos.cvat_io import data/cvat/exports/<task>/annotations.xml handpoly <seq>`.
Related work for the method: `docs/related_work_layered_sam2.md`.

## First seq2–4 human labels (tier A, 23 frames) and a learned change threshold, 2026-10-14 (`learned_thresh.py`)
Labels: user's brush masks, tier-A frames above, tag `handA` (exports `data/cvat/exports/seq{2,3,4}_poly_A_annot`).
Annotator note: drawn closer to amodal than modal, varying by frame; visibility was uncertain in places (no void
label used). The SAM2 tracker runs are T1 (SAMURAI) and B0 (SAM2.1); V̂ = (M_C ∧ add) ∪ T1, where M_C is the
SAM2 mask on the GT-free person composite (as in `run_vhat.py`).

**Plain scores, seq2–4 (n = 23), J / P / R:** T1 0.721 / 0.911 / 0.777; B0 0.736 / 0.927 / 0.783; SAM2-ceiling
0.701 / 0.927 / 0.743. ΔJ B0 − T1 +0.015 [−0.005, +0.038].

**Learned threshold.** Candidates are M_C ∧ ¬T1. Each frame gets GT-free anchors:
- μ_P = median of pb over T1, where pb = log(1 + 5×5 mean (R − B)²) is the pixel change against the empty-scene
  estimate B;
- μ_B, σ_B = median and 1.4826·MAD of pb far from (M_C ∪ T1).

Arms:
- **anchored**: u = (pb − μ_B)/(μ_P − μ_B) > u*. One scale-free parameter.
- **logistic**: on [pb, μ_B, σ_B, μ_P].

Results:
- **Pre-registered selection (seq1 LOFO on brush) picked logistic, and it FAILS on seq2–4:** ΔJ −0.042
  [−0.083, −0.002]; it leaks 0.48× T1 area.
- The anchored arm, not selected (n = 23), gives +0.009 [+0.001, +0.020], better on 14/23.
- **Leave-one-sequence-out (LOSO),** fitting on three sequences (seq1 brush + seq2–4 handA) and scoring the
  fourth:
  - anchored: u* = 0.75–0.90 in every fold, pooled ΔJ **+0.010 [+0.003, +0.018], better on 22/35**;
  - logistic: −0.012 [−0.032, +0.004]. Its σ_B weight changes sign between folds (−1.2 / +3.2), so frame-level
    covariates are not identifiable from 4 sequences.
- Conclusion: a single in-frame-normalised parameter transfers; anything that learns the normalisation needs many
  more scenes (synthetic, `layered_sam2_architecture.md`).

**H2 replicates on independent labels, but carries no area on seq2–4:**
- Hand-label components below half a stride-16 cell: n = 52, T1 recall 0.01.
- On seq2–4 these components hold 0.0–0.5% of label area. Adding all of them perfectly would raise T1's J by
  ≤ 0.005 (seq1: 2.7% of area, +0.013).

**Where T1's missing area is (% of label area; in brackets the share showing pixel change u > 0.85):**

| seq | missed in total | within 3 px of T1's edge | gaps inside T1's hull | outside the hull |
|---|---|---|---|---|
| seq2 | 12.7% | 8.0% [19%] | 1.7% [3%] | 3.0% [8%] |
| seq3 | 24.6% | 10.5% [31%] | 3.4% [8%] | 10.8% [34%] |
| seq4 | 29.9% | 14.5% [14%] | 4.7% [3%] | 10.7% [8%] |
| seq1 | 23.0% | 12.7% [54%] | 3.0% [27%] | 7.3% [54%] |

On seq2–4 most of the missing area is (a) a boundary band and (b) pixels that do not differ from the empty
scene. Those are consistent with amodal-ish labelling, and no visible-evidence cue can recover them.

**With a boundary ignore band of r px either side of the label edge** (a VOC-style void border; Everingham et al.,
IJCV 2010 — from memory, unverified), seq2–4 T1 J / R:
- r = 0: 0.721 / 0.777
- r = 2: 0.810 / 0.850
- r = 3: 0.829 / 0.866

**Implication for the goal (beat SAM2 on modal J by recall):**
- Under the current seq2–4 label convention, about half of the recall gap is boundary disagreement.
- Most of the rest is not visibly evidenced.
- The label protocol (void label for uncertain regions, a boundary band) has to be fixed before a recall method
  can win on seq2–4.

BSUV-Net 2.0 off the shelf: all four Google Drive weight links in `ozantezcan/BSUV-Net-inference@2dc3003`
README return 404 (2026-10-14). Only the training code is released (GPL-3.0); running it needs retraining on
CDnet-2014.

### Diffusion-VAS amodal vs the seq2–4 labels (agreement check, 2026-10-14; `amodal_dvas.py` with DVAS_SEQ, 1 seed)
Same setup as the seq1 test (a): SAM2-ceiling as the modal input, A = prediction ∪ modal. J / P / R against handA:
- seq2: SAM2-ceiling 0.735 / 0.937 / 0.773; amodal 0.752 / 0.815 / 0.899.
- seq3: SAM2-ceiling 0.705 / 0.870 / 0.777; amodal 0.698 / 0.779 / 0.871.
- seq4: SAM2-ceiling 0.664 / 0.966 / 0.682; amodal 0.728 / 0.848 / 0.839. ΔJ +0.064 [+0.015, +0.107].

The amodal additions are 40% / 39% / 59% inside the label. On seq1 polygons the same comparison gave ΔJ −0.017.

Reading: the seq2–4 labels sit between modal and amodal. seq4 leans amodal; seq2 and seq3 do not, which is
consistent with the annotator's own description ("depends on the frame"). This is a label-convention measurement,
not a method result.

## seq2–4 with void (modal convention), 2026-10-14 (`score_void.py`, labels `handAv`: 1 person, 2 void)
Void covers on average 47.6% of the (person ∪ void) area per frame. Void pixels are ignored in all of J, P and R.

**Scores, J / P / R (n = 23):**

| arm | J | P | R |
|---|---|---|---|
| T1 | 0.750 | 0.887 | 0.819 |
| B0 | 0.755 | 0.901 | 0.814 |
| SAM2-ceiling | 0.716 | 0.883 | 0.783 |
| V̂ anchored (u* = 0.85) | 0.749 | 0.868 | 0.835 |
| Diffusion-VAS amodal | 0.693 | 0.772 | 0.866 |

**ΔJ against T1:**

| arm | ΔJ − T1 | 95% CI |
|---|---|---|
| B0 | +0.005 | [−0.008, +0.017] |
| SAM2-ceiling | −0.034 | [−0.077, −0.003] |
| V̂ | −0.001 | [−0.009, +0.008] |
| amodal | −0.057 | [−0.089, −0.026] |

The amodal model now loses clearly, consistent with void having turned the labels modal.

**Post-hoc V̂ does not replicate its seq1 gain (+0.042) on modal seq2–4 labels.** Only 22% of T1's missed non-void
area shows change against the empty scene (seq1: 54%).

**Oracle headroom for T1 (J, seq2–4, void ignored):**

| oracle edit | J |
|---|---|
| none | 0.750 |
| + all sub-cell person components (3.1% of area) | 0.767 |
| + missed pixels within 3 px of T1 | 0.849 |
| − all false positives | 0.819 |
| + all missed pixels | 0.908 |

Dilating or eroding T1 by 1 px trades P against R almost evenly (seq2–4 J: r = −1 0.689, r = +1 0.757, r = +2
0.718; seq1 brush: 0.531 / 0.575 / 0.543). So the boundary error is mis-placement in both directions, not a
too-tight bias.

Open question: how much of the boundary band is annotator noise. This needs inter-annotator J on the same frames,
from the colleague's labels.
