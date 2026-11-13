# Ape foliage pool Implementation Plan

> **For agentic workers:** executed natively (user: "A approved, go ahead"). Steps use `- [ ]`.

**Goal:** an offline pool of real near-layer foliage α stacks (40 consecutive frames per source) for planting over moving apes.

**Architecture:** one script, `benchmark/nfo_vos/ape/foliage.py`. Per PanAf500 source video (non-eval, non-leaked: 186):
pick the 40-frame window with the least dilated-ape-box coverage, run Depth Anything V2 Small on all 40 frames (one GPU
batch at the model grid 518×294), rank-normalise per frame, α = sigmoid((D − 0.8)/0.03), upsample α to 720×404,
save uint8. The loader re-decodes the RGB/gray frames itself (decode of 40 frames at 720×404 is cheap).

**Tech Stack:** `../samurai/.venv` (torch, transformers 5.2, cv2), HF `depth-anything/Depth-Anything-V2-Small-hf` (cached).

**Spec:** `docs/layer_sam2/2026-11-13-ape-stage1-data-design.md` §3 + HANDOFF; approach A chosen 2026-11-13 (per-frame α,
no stored pixels; rejected: pre-cut fixed crops, single-frame α).

## Global Constraints
- Near layer: rank01 per frame (PanAf20K `panaf/depth_cues.py:35`), α = sigmoid((D − 0.8)/0.03), model input (518, 294).
- Ape exclusion: PanAf500 GT boxes dilated 15% per side, unioned over the window = no-cut mask.
- IR flag: mean Lab chroma |a−128|+|b−128| < 4 (PanAf20K `tools/fgbg_change_check.py:58`).
- Split 90/10 train/val **by leakage cluster** (`results/ape/leakage.csv`), seed 0.
- Visual trace saved in the repo: `images/ape/foliage_pool.jpg`.

## Review Focus
- Source with apes in every frame → window still chosen (min coverage); coverage recorded, loader avoids no-cut.
- Video shorter than 40 frames or failed decode → skipped with a printed reason, not a crash.
- Non-720×404 videos (540-high ones exist) → α saved at the source's own H×W.
- Rank-normalised depth flicker → `flicker` column (mean |α_t − α_{t−1}| on near pixels), inspected before use.
- Ground as "near" → `frag` (boundary/area of α ≥ 0.5) recorded; loader filters crops on it.

---

### Task 1: pool builder + unit tests
**Files:** create `benchmark/nfo_vos/ape/foliage.py`, `benchmark/nfo_vos/tests/test_ape_foliage.py`.
**Produces:** `results/ape/foliage/<video>.npz` {`alpha` uint8 (40,H,W), `nocut` bool (H,W), `start` int};
`results/ape/foliage_index.csv` columns video, start, H, W, ape_cov, near_cov, frag, flicker, ir, cluster, split;
functions `best_window(boxes: dict[int, list], n_frames, H, W) -> (start, nocut)`, `near_alpha(D: Tensor[B,h,w]) -> Tensor`.
- [ ] Tests: `near_alpha` gives 0.5 at rank 0.8 and is monotone; `best_window` picks the ape-free window and its nocut covers the dilated box.
- [ ] Implement; tests pass (`$PY -m pytest benchmark/nfo_vos/tests/test_ape_foliage.py -q`).
- [ ] Run `$PY -m benchmark.nfo_vos.ape.foliage` on all 186 sources; print timing, flicker/frag quantiles; montage.
- [ ] User judges `images/ape/foliage_pool.jpg`; commit code + plan (data not in git).

### Task 2 (next): clip loader `benchmark/nfo_vos/ape/data.py` — separate plan after the pool is accepted.
