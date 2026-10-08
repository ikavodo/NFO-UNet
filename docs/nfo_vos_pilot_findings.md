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
