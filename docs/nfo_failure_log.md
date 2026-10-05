# NFO failure log

One line per dead end: what was tried, why it failed.

- 2026-10-05 · Occluder layer = Otsu-dark pixels of the person-free median background, to score pseudo-masks by |M ∧ O|/|M|: O covers 58–70% of the frame (dark clothing, shade, ground also dark), main masks already overlap it at median 0.43–0.95 → no separation. Use background agreement |I−B|<τ instead (`check_occluder_overlap.py`).
- 2026-10-05 · Pixel-level occluder removal (`fix_ext_occluder.fix`) as a GLOBAL filter: also cuts camouflaged person pixels (seq3 main masks p10 IoU 0.58) → applied to hand-flagged frames only.
- 2026-10-05 · Same pixel fix on seq4 f1022–1030 left patch: patch only half-matches background (a(C) 0.42–0.52), so the fix keeps it → cut by static location instead (`REGION_CUTS`).
