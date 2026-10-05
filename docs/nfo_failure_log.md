# NFO failure log

One line per dead end: what was tried, why it failed.

- 2026-10-05 · Occluder layer = Otsu-dark pixels of the person-free median background, to score pseudo-masks by |M ∧ O|/|M|: O covers 58–70% of the frame (dark clothing, shade, ground also dark), main masks already overlap it at median 0.43–0.95 → no separation. Use background agreement |I−B|<τ instead (`check_occluder_overlap.py`).
- 2026-10-05 · Pixel-level occluder removal (`fix_ext_occluder.fix`) as a GLOBAL filter: also cuts camouflaged person pixels (seq3 main masks p10 IoU 0.58) → applied to hand-flagged frames only.
- 2026-10-05 · Same pixel fix on seq4 f1022–1030 left patch: patch only half-matches background (a(C) 0.42–0.52), so the fix keeps it → cut by static location instead (`REGION_CUTS`).
- 2026-10-05 · VOS pilot T4 prompt = merged-blob bbox centre: on the person only 58% of frames (in GT box 94%), SAM2 segments the occluder → T4 J&F 0.37. Use the distance-transform interior point (71%) + blob box.
- 2026-10-05 · VOS pilot: image-level temporal integration (warped blob vote / median reference + Otsu visibility) for MODAL masks loses to the single frame-t blob at every setting, incl. oracle GT alignment and stride 1–4 (J&F −0.04…−0.12; recall gain 0). Integration adds A∖V_t pixels, which modal GT counts as errors. Do not retry for modal output (`diag_integration.py`, `results/benchmark/pilot/diag_*.csv`).
- 2026-10-05 · Buffer stride to widen gap-filling span (w < N·δ·s/2): longer span makes integration worse even with oracle alignment (articulation / non-rigid motion), → not the binding constraint.
- 2026-10-05 · VOS pilot: longer integration buffer at stride 1 (N = 13/19/25) is worse for both modal J&F (−0.08…−0.12 vs frame-t blob) and amodal box IoU (+0.094 at N=7 → −0.001 at N=25). Translation-only alignment cannot absorb gait articulation over longer spans — same root cause as GPJATK (master_thesis project_outlook Part 3). Keep N≈7 (`diag_buffer_length.csv`).
