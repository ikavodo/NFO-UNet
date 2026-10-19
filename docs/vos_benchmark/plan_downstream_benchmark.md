# Downstream NFO benchmark (tracking boxes + composite recognition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score every method on NFO without the pseudo-GT. The scores are:
- hand-label J/P/R;
- one-pass tracking metrics (AUC, P, P_norm, failure rate) against the dense GT boxes and against depth-restricted
  "visible" boxes;
- YOLO person recognition on masked temporal composites built with GT-box motion.

**Architecture:**
- **Arms produce masks.** Every arm writes `results/benchmark/rr/masks/<arm>/<trial_id>.npz`, using the B0/T1 schema
  (`masks` [40,600,800] bool, `frames` [40]).
- **Scorers read masks.** They are CPU-only and never touch a model, except YOLO, which needs the GPU.
- **Parallel lanes:**
  - Lane R, the remote GPU, dumps the masks of the arms that are still missing.
  - Lane L, local, builds and validates the scorers on the B0/T1 masks that already exist.

**Tech Stack:** numpy, opencv, ultralytics 8.4.89 (`data/yolo11m.pt`), SAM2.1 b+, slurm.

**Spec:** this file, §0 (decisions agreed with the user, 2026-10-19).

## 0. Decisions (user, 2026-10-19)

**D1. No pseudo-GT as a headline label.**
- Segmentation is scored on hand frames only:
  - seq1 has 20 `*_handmask_raw.png`, 15 of them inside a window; seq1 is the selection set;
  - seq2–4 have 8/7/8 `*_handAv.png`, all inside windows; these are the single test.
- Frozen-silhouette numbers stay in the findings as history.

**D2. Tracking.**
- Metrics: OPE, LaSOT-style AUC, P@20 px, P_norm, plus a failure rate (share of frames with IoU = 0) in place of VOT
  robustness.
- EAO/A/R are excluded. They need VOT resets or anchors and many sequence lengths.
- Formulas are ported from pytracking `pytracking/analysis/extract_results.py` (`calc_err_center`,
  `calc_iou_overlap`, the threshold sets) and cited in a comment.

**D3. Two box references.**
- (a) The GT box, which covers the whole person.
- (b) A visible box: the GT box trimmed where the empty-scene depth shows near foliage.
  - Parameters are selected on seq1 hand frames and tested once on seq2–4 against the tight boxes around the hand
    masks.
  - (b) is used only if it beats (a) on that test.

**D4. Recognition composite.**
- N = 7 frames, aligned to frame t by the GT-box centre shift.
- Uniform masked mean Σ M_i·I_i / Σ M_i, filled with the empty-scene background where Σ M_i = 0.
- YOLO11m, person class.
- Scores: AP50 vs the GT box of t, and the confidence of the matched detection.
- Controls:
  - (i) no mask (plain aligned mean);
  - (ii) the GT box as the mask;
  - (iii) N = 1.
- Sensitivity: N = 3; recency weights w_i = 0.7^(t−i), reported as rank stability (Kendall τ of the method
  ranking).

**D5. Remote first for GPU work.** The remote has seq1. Local GPU work only when the remote queue is busy.

## Global Constraints

- **No caches** beyond the masks npz, which is ≈ 40 KB per window per arm (B0 is 2.5 MB for 61 windows, measured).
- **Visual traces** for every scorer go to `images/benchmark/downstream/` at dpi ≤ 90: one montage per arm on 3 fixed
  windows (seq1_s0_t11, plus the first seq2 and seq3 trials).
- **Bootstrap:** window bootstrap, 2000 resamples, seed 0. Paired differences vs T1 and vs trk.
- **Selection and test:** seq1 numbers are selection-only. seq2–4 is evaluated once per arm and reported as is.
- **Coordinates:** native 800×600 for every box. GT via `utils.bb_utils.parse_bbs` +
  `gen_nfo_pseudo_masks.gt_to_native`, as `benchmark/nfo_vos/trials.py:27-30,72,178`.

## Review Focus

1. **Empty mask** (the target is lost). `mask_box` returns None. It scores IoU 0 and center error ∞, counts as a
   failure, and adds nothing to the composite.
2. **Fragments far from the body** (a FP leaf blob at the image edge). The tight box inflates. Report that as is (the
   metric is meant to penalise it); do not clean it.
3. **Composite frames before the window start** (t−6 < first frame). Skip t; don't pad.
4. **Visible-box trimming that removes everything.** Fall back to the GT box and count the fallbacks.
5. **YOLO returns no person.** The composite counts as a miss: confidence 0, still included in AP as a FN.

---

### Task 1: Box metrics + ceiling test (lane L, ~3 h)

**Files:**
- Create `benchmark/nfo_vos/downstream/__init__.py` and `benchmark/nfo_vos/downstream/box_metrics.py`.
- Test: `tests/test_downstream.py`.

**Interfaces:**
- Produces:
  - `mask_box(m: np.ndarray[H,W] bool) -> np.ndarray[4] xyxy | None`;
  - `gt_boxes(seq: str) -> dict[int, np.ndarray[4]]` (native xyxy per frame);
  - `ope(pred: list[xyxy|None], gt: list[xyxy]) -> dict(auc, p20, pnorm, fail, iou: np.ndarray, cerr: np.ndarray)`;
  - `load_arm(arm: str, trial_id: str) -> (masks, frames)`.
- `main(arms: list[str], ref: str = 'gt')` writes `results/benchmark/rr/downstream/box_<ref>.csv` (per window × arm)
  and prints the per-sequence-group table with bootstrap CIs.

- [ ] **Step 1: Write the failing tests.**
  - `test_mask_box_empty`: all-False → None.
  - `test_mask_box_two_blobs`: blobs at (10,10) and (50,80) → [10,10,50,80] in xyxy, inclusive-max convention, same as
    pytracking's xywh conversion.
  - `test_ope_perfect`: pred == gt → p20 == 1, fail == 0, auc equal to the ported threshold grid's value for IoU = 1.
    The implementer computes that value from the ported thresholds and states it in the assert comment.
  - `test_ope_lost`: all None → auc 0, fail 1.
- [ ] **Step 2: Run `pytest tests/test_downstream.py -v`.** Expected: FAIL (ImportError).
- [ ] **Step 3: Implement.** Port the thresholds verbatim and cite the pytracking file and commit beside the block.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Ceiling test** (decides how much weight the GT-box metrics carry).
  - For every hand frame (43), compute IoU(mask_box(hand mask), GT box) and the centre error.
  - Print the mean and median per seq, and save `images/benchmark/downstream/hand_vs_gt_box.png` (the 6 worst
    overlays).
  - **Pre-registered reading:**
    - mean IoU ≥ 0.7: the GT box is a close-enough reference and D3(b) is optional;
    - mean IoU < 0.7: run Task 2 before reporting any box numbers.
- [ ] **Step 6: Run `main(['b0','t1','b0-pt','t1-pt'])`** on the existing masks. Check against hand frames that the
  B0 and T1 numbers are plausible (no window with all-None boxes unless its masks are empty).
- [ ] **Step 7: Commit** `box metrics (pytracking port) + hand-mask vs GT-box ceiling`.

### Task 2: Depth-restricted visible box (lane L, ~2 h; only if Task 1 Step 5 < 0.7)

**Files:**
- Create `benchmark/nfo_vos/downstream/visible_box.py`.
- Test: `tests/test_downstream.py`.

**Interfaces:**
- Consumes `gt_boxes`, `mask_box`, and the disparity maps `results/benchmark/rr/depth/seq{1..4}_bg_disp_1036.npy`:
  [600,800], 0..1, larger = nearer, Depth Anything V2 L on the empty-scene median.
- Produces `visible_box(gt: xyxy, disp: [600,800], q: float, rho: float) -> (xyxy, fell_back: bool)` and
  `visible_boxes(seq, q, rho) -> dict[int, xyxy]`.

**Rule.** Inside the GT box:
- the occluder set is O = {disp > τ}, with τ the q-quantile of disp inside the box. The quantile is used because
  disparity is only affine-relative;
- the support is S = box \ O;
- trim whole rows from the top and bottom, then whole columns from the left and right, while the row's or column's
  support fraction is < ρ.

- [ ] **Step 1: Write the failing tests.**
  - `test_visible_box_no_occluder`: constant disp → returns the GT box, fell_back False.
  - `test_visible_box_top_occluded`: disp 1 in the top half of the box, 0 elsewhere, q 0.5, ρ 0.5 → y0 moves to the
    box mid-row ±1.
  - `test_visible_box_all_occluded` → GT box, fell_back True.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Select on seq1 (20 hand frames).**
  - Grid q ∈ {0.5, 0.6, 0.7, 0.8}, ρ ∈ {0.1, 0.2, 0.3}.
  - Maximise the mean IoU(visible_box, mask_box(hand)).
  - Print the full grid and freeze (q, ρ) in a module constant with today's date.
- [ ] **Step 6: Test once on seq2–4 (23 frames).**
  - Paired ΔIoU = IoU(visible, hand box) − IoU(GT, hand box), frame bootstrap.
  - **Pass:** CI lower bound > 0. Otherwise drop D3(b) and log it in `nfo_failure_log.md`.
  - Save `images/benchmark/downstream/visible_box_seq234.png` (all 23 frames: GT, visible and hand boxes over the
    frame).
- [ ] **Step 7: Commit.**

### Task 3: Mask dump for the missing arms (lane R, write ~2 h, run on cluster)

**Files:**
- Create `benchmark/nfo_vos/downstream/dump_arms.py` and `benchmark/nfo_vos/downstream/dump_arms.sbatch` (pattern:
  `benchmark/nfo_vos/layers/fusion_adapter.sbatch`).

**Interfaces:**
- Consumes:
  - `video_adapter.per_frame_inputs(pred, net, t)`;
  - `video_adapter.run_window(vp, t, inputs, arm)`;
  - `fusion_adapter.decode(sam, f, token, point)` (fusion_adapter.py:114);
  - `run_sam2_video.build_predictor('b0')`;
  - the checkpoint `results/benchmark/rr/layers_fusion_adapter/token_seed0/fusion_it200.pt`.
- Produces `masks/<arm>/<trial_id>.npz` (`masks`, `frames`, plus `boxes` [40,4] for `trkbox`) for these arms:
  - `trkbox`: the kernel-f16 tracker box itself, no masks;
  - `trk`: per-frame SAM2 with tracker box + point, no memory;
  - `adapter-pf`: per-frame adapter v3.4 it200;
  - `vid-prompts`;
  - `vid-adapter-pre`;
  - `vid-adapter-noC`: adapter-pre with the composite input zeroed, the −composite ablation.
- CLI: `python -m benchmark.nfo_vos.downstream.dump_arms --arm A --trial-index I` (the slurm array index = trial
  index 0..60). It skips the window if the npz exists.

- [ ] **Step 1: SC (local, 1 window, seq1_s0_t11).**
  - The `vid-prompts` masks must match `video_check`'s prompts arm on the same window, 0 px differ.
  - `trk` on the hand frames in that window must match the `fusion_nfo.evaluate` box+pt arm.
- [ ] **Step 2: Write the sbatch.** Array 0-60, one arm per job via `ARM`. List the files the remote needs and git
  doesn't hold:
  - `fusion_it200.pt`;
  - `data/nfo_final/nfo_final/seq1..4`;
  - `data/nfo_processed/seq1..4_gt`;
  - `results/benchmark/rr/trials.json`;
  - the SAM2 checkpoint.
- [ ] **Step 3: Commit, push, and launch on the remote**, the 6 arms in parallel arrays. Pull the masks back with
  rsync, which is tiny.
- [ ] **Step 4: Diffusion-VAS on all window frames.** Currently it only exists on hand frames (`amodal_dvas*/`, 3
  seeds). It is a separate array job using the existing DVAS script with a frame-list argument. Without it, DVAS can't
  be scored on boxes or composites. Report it on hand frames only if it doesn't finish this week.

### Task 4: Masked composite + YOLO recognition (lane L build, run where the GPU is free; ~4 h)

**Files:**
- Create `benchmark/nfo_vos/downstream/recognition.py`.
- Test: `tests/test_downstream.py`.

**Interfaces:**
- Consumes `load_arm`, `gt_boxes`, `V.native(frame, seq)`, and the empty-scene background computed as in
  `video_adapter.per_frame_inputs` (median over the warm-up).
- Produces:
  - `gt_shifts(boxes: dict, frames: list[int], t: int) -> list[(dx, dy)]` (GT-box centre of t minus centre of i);
  - `masked_composite(imgs: list[HxWx3], masks: list[HxW bool], shifts, bg: HxWx3, weights: list[float] | None) -> HxWx3 uint8`;
  - `score_yolo(model, img, gt_box) -> dict(conf, iou, box)` (person class 0, conf threshold 0.01, top person by
    confidence);
  - `main(arms, N=7, weights='uniform')` → `results/benchmark/rr/downstream/recog_<N>_<weights>.csv`.
- AP50: ultralytics `ultralytics.utils.metrics.ap_per_class`, over all composites of a sequence group.

- [ ] **Step 1: Write the failing tests.**
  - `test_composite_no_mask_is_bg`: all-False masks → output == bg.
  - `test_composite_shift`: a single bright pixel at x = 10 in frame i with dx = +5 lands at x = 15.
  - `test_composite_uniform_mean`: two frames, values 0 and 100 under full masks → 50.
  - `test_gt_shifts_zero_at_t`.
- [ ] **Step 2: Run the tests.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use cv2.warpAffine with an integer-free sub-pixel shift, INTER_LINEAR, applied to both
  the image and the mask.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Controls first, on seq1.** Run arms `nomask`, `gtbox` and `t1` at N = 1 and N = 7.
  - Save `images/benchmark/downstream/composite_controls.png` (3 windows × 4 arms) for the user to judge.
  - **Reading:** if `nomask` AP50 ≥ every method's AP50, masks add nothing to recognition. Stop and report before
    running the rest.
- [ ] **Step 6: All arms**, N = 7 uniform, then the sensitivities N = 3 and recency 0.7. Print Kendall τ of the
  method ranking between settings.
- [ ] **Step 7: Commit.**

### Task 5: Results table + findings (local, ~1 h)

**Files:**
- Create `benchmark/nfo_vos/downstream/report.py`.
- Modify `docs/vos_benchmark/nfo_vos_pilot_findings.md` (new section) and `nfo_failure_log.md`.

**Interfaces:** consumes the CSVs of Tasks 1, 2 and 4, plus a hand-label J/P/R scorer reused from
`vhat_vs_silhouette.py` / `fusion_nfo` (the hand frames inside windows).

- [ ] **Step 1: One table**, seq1 (selection) and seq2–4 (test) as separate column groups.
  - Rows: arms.
  - Columns: hand J/R, box AUC/P/P_norm/fail (GT ref, and visible ref if Task 2 passed), recognition AP50 and mean
    confidence.
  - Paired CIs vs T1 and vs trk.
- [ ] **Step 2: Write the constructive findings section.** Lead with what holds on seq2–4. Failures go in one line each
  in the failure log.
- [ ] **Step 3: Commit and push.**

## Schedule (poster week)

| when | lane L (local) | lane R (remote) |
|---|---|---|
| Mon | Task 1 (+ ceiling), Task 3 Step 1 SC | — |
| Mon evening | Task 2 if needed | Task 3 arrays (6 arms), DVAS array |
| Tue | Task 4 on B0/T1 + controls | video-mode training stage 1 (separate plan, after the reviewer) |
| Wed | Task 4 all arms, Task 5 | — |

Video-mode training is not in this plan. It gets its own plan plus the pre-authorised reviewer, and runs on lane R from
Tuesday. Model selection moves from the pseudo-GT to seq1 hand J + seq1 box AUC.
