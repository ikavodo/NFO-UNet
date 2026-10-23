# Implementation plan: GPJATK + foliage testbed for Layer-SAM2 (written 2026-10-14 for a fresh session)

Read first:
- `docs/layer_sam2/archive/layered_sam2_architecture.md`, section "v2": the architecture and its ablation table.
- `docs/vos_benchmark/nfo_vos_pilot_findings.md`, last three sections: the NFO numbers this testbed must reproduce.
- `docs/vos_benchmark/nfo_failure_log.md`: what not to retry.

## 0. Why this exists (context in five lines)
- **Goal:** beat SAM2.1 (B0) and SAMURAI (T1) on modal person segmentation under fragmented foliage occlusion (NFO
  dataset). The win has to come from higher recall: humans label more "generously" than SAM2.
- **Post-hoc fixes after T1 failed on modal labels.** V̂ = T1 plus change-gated pixels from a person composite gives
  ΔJ −0.001 on NFO seq2–4 with the void regions ignored.
- **Next idea:** put the empty-scene comparison and the person composite *inside* SAM2 and train them ("Layer-SAM2",
  architecture v2).
- **Training needs exact layers:** visible person V, amodal person S, occluder matte a. Real NFO has none of these,
  so we build them synthetically: GPJATK walkers + foliage.
- **This plan is step 1:** build the testbed and check it is a valid proxy for NFO, with *no training*. Steps 2–3
  are outlined at the end.

## 1. Inputs on disk

| what | where | notes |
|---|---|---|
| GPJATK | `../master_thesis/GPJATK/Sequences/<pXsY>/` | 166 sequences, 4 cameras, 25 fps. `Images/c{1..4}_*.avi` (video), `Silhouettes/c{n}/*.png` (person mask, 540×960), `Background/c{n}.avi` (empty scene, per camera). Release agreement applies: cite Kwolek et al. (see `GPJATK/Readme.txt`) |
| foliage template | `images/veronica111886-yakushima-2336735.jpg` (NFO-UNet repo) | 4000×2667 RGB. The upper ~3/5 (rows 0–1600) is a foliage scene with depth; the lower part is not foliage. Not a GPJATK file: check its licence (Pixabay-style name) before any figure is published |
| existing testbed code | `../master_thesis/experiments/prototypes/gpjatk_em/em_severe_occlusion_testbed.py` | Reuse: `src.io.read_avi`, `read_frames`, `src.occluders.occ_branch(..., return_mask=True)`, walking-interval trimming by silhouette differencing |
| occluder generators | `../master_thesis/src/occluders.py` | 9 types (`occ_branch`, `occ_perlin`, `occ_worley`, `occ_fbm`, …). Each can return a `[T,H,W]` bool mask |
| SAM2 / SAMURAI runners | `benchmark/nfo_vos/run_sam2_video.py` (`build_predictor('b0' \| 't1')`) | Run with `../samurai/.venv/bin/python` (the shell's default `python` lacks sam2) |

Convention from master_thesis: synthetic frames go to the model in **grayscale, replicated to 3 channels**. NFO is
grayscale, and colour would let the model separate person from foliage trivially.

## 2. The generator: `benchmark/nfo_vos/synth/gpjatk_foliage.py` (new)

Per clip (one GPJATK sequence × camera × a contiguous walking window of N = 40 frames):

1. **Person + background layers.** The GPJATK frame I^g_t already equals m_t·P_t + (1 − m_t)·B, with m_t =
   silhouette > 0.5. Use the frame as is; do not cut the person out. B̂ = a·O + (1 − a)·median(`Background/c{n}.avi`):
   the empty scene *with the foliage in front*, as NFO's warm-up median also contains the foliage. Trim to the walking
   interval by silhouette differencing, with a relative threshold (0.5 × the clip's 90th percentile; the testbed's
   absolute 0.12 finds no walking frames on cameras c2/c4).
2. **Foliage texture O.**
   - Grayscale the template, keep rows 0 … 0.6·H_img.
   - Take a random crop with aspect 16:9, resized to 960×540.
   - The crop scale sets the leaf size. Sample the crop width from [960, 2667] px of the template; smaller crops
     give bigger leaves and larger fragments.
   - Fixed per clip, with a slow per-frame translation of ≤ 2 px (sway) and an optional small elastic warp.
3. **Occluder matte a_t** (the user's choice, 2026-10-15: the generated mask only *selects* where the grayscale
   template shows; the occluder functions' own light-canvas intensities are not used).
   - **Implemented** (`benchmark/nfo_vos/synth/gpjatk_foliage.py`): occ_perlin (thresholded bicubic-upsampled grid =
     leaf clumps with gaps), clump 12 px, static. occ_branch / occ_worley are not used: Worley borders give large
     visible cells (wrong regime), branches give anisotropic slivers.
   - **Density (user, 2026-10-15: high).** The sanity check (findings, 2026-10-15) shows that at density ≥ 0.6 SAM2 is
     precision-limited, not recall-limited, unlike NFO; the NFO-like regime is 0.30–0.45 with the box prompt, vs T_8.
     Density range for step 1: **open, the user's decision**.
   - a_t = mask from `occ_branch` / `occ_worley` / `occ_perlin`, parameters drawn per clip.
   - Soften the edges: Gaussian blur σ ∈ [0.5, 1.5] px, so a is in [0, 1] and edges are not perfectly hard.
   - Coverage target per clip: uniform in [0.3, 0.7] of person pixels occluded. Calibrate num_branches /
     threshold per clip by a few bisection steps; the old `calibrate_branch_num_branches` was deleted, see the note
     in the testbed file.
4. **Composite:** I_t = a_t·O_t + (1 − a_t)·I^g_t.
5. **Exact targets:**
   - V_t = m_t ∧ (a_t < 0.5): visible person (modal).
   - S_t = m_t: amodal person.
   - O-mask_t = a_t ≥ 0.5.
   - Human-style T_{r,t} = V_t ∪ (closing_r(V_t) ∧ S_t) for r ∈ {0, 4, 8, 12} px (disk), with closing_r = OpenCV
     MORPH_CLOSE using an elliptical kernel of size 2r+1.
6. **Tracker prompts (same protocol as the NFO benchmark):** box = bbox(S_{t0}); point = the distance-transform peak
   of V_{t0}, which must lie on a visible pixel. t0 = first frame of the clip.
7. **Save per clip** under `data/synthetic/gpjatk_foliage/<clip_id>/` (`data/` is gitignored):
   - `frames/00000.jpg …` (grayscale, 3-channel JPEG, quality 95, the format SAM2's video loader reads);
   - `V.npy`, `S.npy`, `occ.npy` ([N,540,960] bool), `T_r{r}.npy`;
   - `bg.png` (B̂);
   - `meta.json`: sequence, camera, frames, crop box, crop scale, occluder type and parameters, realised coverage,
     seed, prompts.
8. **Visual trace (mandatory):** a montage per 10 clips into `images/benchmark/gpjatk_foliage/`, ~10 in wide,
   dpi 80. Columns: I_t | V (green) / S − V (red) over I_t | T_8. The user judges these; do not Read them back.

Scale: ~50 clips × 40 frames (2000 frames) for step 1, from ≥ 10 subjects, all four cameras. Hold out 6 subjects
entirely for later test splits. Write the subject split into `meta` now.

Fragment-size control: report, per clip, the size distribution of V's connected components in stride-16 cells
(cell = 16·960/1024 = 15 px for GPJATK frames; NFO's is 12.5 px). It must span sub-cell (< 0.5 cell) to ≥ 4 cells,
like NFO.

## 3. Step 1 measurement: `benchmark/nfo_vos/synth/profile.py` (new)

Run B0 and T1 on every clip: prompts at t0, propagate forward, cache masks to
`data/synthetic/gpjatk_foliage/<clip>/masks_{b0,t1}.npz`. This is the expensive stable stage; cache it.

On frames 6…39 of each clip, compute the same breakdown as `benchmark/nfo_vos/score_void.py` and
`learned_thresh.py`, against V (and against T_r):
1. J / P / R of T1 and B0 vs V, and vs T_4, T_8, T_12.
2. Recall of V's components binned by size: [0, .5), [.5, 1), [1, 4), [4, ∞) cells.
3. T1's missed area split: within 3 px of T1's outline / inside T1's hull / outside the hull. Also the share of the
   missed area with change u > 0.85, where u = (pb − μ_B)/(μ_P − μ_B) is defined in `learned_thresh.py`.
4. Oracle headroom: J of T1 ∪ (missed ≤ 3 px), T1 ∪ (sub-cell components), T1 minus false positives.
5. Dilation test: J of T1 dilated / eroded by 1–2 px.
6. Person-vs-background contrast in units of the frame's robust z (`factorised_extras.pixz`), median over V.

**NFO reference values** (to compare against; all measured 2026-10-14):

| quantity | NFO value |
|---|---|
| T1 J / P / R (seq2–4, void ignored, n = 23) | 0.750 / 0.887 / 0.819 |
| recall by component size, T1, seq2–4 modal | [0,.5) 0.13; [.5,1) 0.39; [1,4) 0.67; [4,∞) 0.84 |
| same, seq1 (H2, SAM2-ceiling labels) | 0.00; 0.33; 0.51; 0.79 |
| T1 missed area, share within 3 px of T1's outline | 61% (10.9 of 18.1%) |
| share of missed area showing change u > 0.85 | 22% (seq2–4), 54% (seq1) |
| oracle J: none / + band ≤ 3 px / − FPs / + sub-cell | 0.750 / 0.849 / 0.819 / 0.767 |
| dilation +1 px | +0.007 J (P/R trade) |
| person contrast (robust z) | ≈ 9 (seq1), ≈ 3 (seq2) |

**Pre-registered verdict for step 1 (state it in the script docstring before running):** the testbed is a valid
proxy if, for T1 vs V:
- (a) sub-cell recall ≤ 0.25 and ≥ 4-cell recall ≥ 0.7;
- (b) J lies in [0.6, 0.85] for at least half of the coverage range;
- (c) person contrast spans the NFO range (3–9) across clips.

**Note on the outline band:** V is exact, so T1's boundary band vs V is *not* expected to match NFO's 61%. Part of
that band is human generosity, which T_r models. Report the band share against V and against T_8; do not use it
for the verdict.

If the proxy fails, fix the compositing before anything else: matte blur, leaf scale, contrast, noise, JPEG.
Record each fix as one line in the failure log.

## 4. Afterwards (outline only; do not start without the user)
- **Step 2, oracle injection, one location at a time.** Train only a zero-init adapter (~1–3 M parameters) with
  oracle inputs, true B and true S, at (i) the dense prompt (`mask_decoder.py:205`) and (ii) the high-res path
  (`mask_decoder.py:224-225`). If even oracle inputs give no gain at a location, drop that location.
- **Step 3, the ablation table of architecture v2 (rows 0–5).** One variable per row, same data, same parameter
  budget, 3 seeds. Test on NFO seq1 + seq2–4 tier A (dev); seq2–4 tier B with void is the final, untouched test.
- **The inter-annotator J on NFO** (colleague's labels) sets the noise floor; report it beside every NFO number.

## 5. Working rules for the implementing session (the user's standing preferences)
- Commit only code and docs. Add the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Check
  `git status` before each commit, because other sessions may be editing.
- Ask before dispatching subagents. Minimise turns and batch tool calls.
- Every image goes into the repo (`images/...`). The user judges visual checks; never claim one "looks right".
- Calibrated language (measured / derived / assumed). Paired bootstrap CIs. One variable per comparison.
- Two docs: a one-line entry in `docs/vos_benchmark/nfo_failure_log.md` per dead end; constructive results in
  `docs/vos_benchmark/nfo_vos_pilot_findings.md`.
- On first mention in any message to the user, re-explain each label (B0, T1, V, S, T_r, …) in one clause.
- No work on weekends. After 19:00, nag the user to stop.
