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
- B0-cp-hybrid: only 1/21 trials came back from the cluster. Not yet evaluable.

## Next (cheapest first)
0. **Do not miss:** SAM-PT and SAM-PD are required baselines for T4-1n in the full run (spec §8).
1. Decide the benchmark's question: modal VOS (B0 strong, integration does not help) vs amodal
   person extent/localisation (integration helps; scored on human boxes, so no pseudo-GT bias;
   matches the original NFO-UNet localisation task).
2. If amodal: proper amodal baselines (an amodal VOS / amodal completion method), not enlarged
   modal boxes; then the feature-fusion SAM2 variant targets amodal extent.
3. Native resolution only matters for the SAM2-input asymmetry (T4 init failures 10/21 at 224
   vs B0 5/21 at native); the integration mechanism is scale-invariant (w/δ).
