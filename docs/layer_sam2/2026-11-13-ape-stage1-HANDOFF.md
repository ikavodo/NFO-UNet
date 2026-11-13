# Handoff: Layer-SAM on apes, Stage 1 (2026-11-13, branch exp/ape-stage1); read first in a new session

**Goal (main paper, not the workshop).** Show that Layer-SAM can be trained on planted **real-foliage** occlusion,
using only clear or partial apes that **move**. Layer-SAM is the frozen SAM2.1 per frame, no memory, plus the
adapter and the integral image. Test: on real occluded videos it must beat SAM2.1 video (with memory), so memory is
not needed.

## Decisions (user, binding)
- **Real foliage only. The procedural occluders are not used**, not even as a first diagnostic.
- Foliage source: the near-depth layer (Depth Anything V2 Small) of high-`near_edge` scenes.
  - PanAf500 first; PanAf-FGBG background clips later.
  - A vegetation/fragmentation filter on the pool.
  - The **same patch cut from consecutive frames** of the source, which gives real sway.
  - Planted over the ape's trajectory, into the frames and B.
  - Accepted visually: PanAf20K `renders/occluders/examples.jpg`, `tools/occluder_examples.py`.
- One clip per source video: its most-moving 40-frame window with motion ≥ 0.5 box widths.
  `benchmark/nfo_vos/ape/clips.py` → 119 clips (76 clear, 43 partial).
- **Gray** (pseudo-IR). B = the whole-video median. Track = GT boxes + jitter. GT = PanAf-SBR masks, fill-in
  included (the adapter should complement what SAM2 misses).
- Leakage: `benchmark/nfo_vos/ape/leakage.py`. DINOv2 median-frame similarity, threshold calibrated on FGBG camera
  ids (0.1% false links, 50% same-camera recall). 101 videos excluded.
- Eval: PanAf20K `results/panaf/eval_subset.csv` (74 / 74 / 74), occluded videos with moving apes. Layer-SAM vs SAM2.1
  video, same first prompt, gray, J. One seed first; 3 seeds if positive.

## Next task
The clip loader, `benchmark/nfo_vos/ape/data.py`.
- Input: a `clip_index.csv` row plus a foliage-pool entry.
- Output: the raw-clip dict of `adapter/fusion_data.clip_raw` (X, bg, V, S, boxes, bxj, cid, modes).
- `build()` / training stay unchanged, except two flags (default off, so NFO stays bit-identical, checked with
  `parity_snapshot`): the GT-box track, and an x+y Theil–Sen shift in `integral_image`.

Before that: the foliage pool (offline depth on the source frames, per-frame near-layer crops for patch tracking).

## Documents
- `docs/layer_sam2/2026-11-13-ape-stage1-data-design.md`: the design, the expert review and the user decisions.
- `docs/layer_sam2/2026-11-13-feature-integral-design-notes.md`: learned feature-space integral image. It comes after
  Stage 1; masked attention, no warping; literature pass included.
- PanAf facts and numbers: `../PanAf20K/docs/2026-11-13-handoff.md`.
