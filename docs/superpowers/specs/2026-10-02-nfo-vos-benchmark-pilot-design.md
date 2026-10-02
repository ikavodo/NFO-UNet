# NFO fragmented-occlusion VOS benchmark: pilot protocol

Status: design, revision 2, awaiting review (2026-10-02). Scope: the **pilot** only (one
back-and-forth of one sequence). Deferred items are listed at the end.

## 1. Question

**Does integrating several views of the person (aligning past frames to the person's motion)
improve *modal* segmentation under fragmented occlusion, compared with single-view-per-frame
memory-based propagation (SAM2, SAMURAI)?**

This motivates a later contribution: a SAM2 variant that fuses encoded features across warped
frames. The pilot checks whether integration at the image level already shows the effect.

Our method uses an amodal intermediate but outputs modal masks. The integrated reference collects
person pixels from several frames into one support S that is closer to amodal. S is then projected
back to frame t by a per-frame visibility step (`visibility_within_mask`,
`tracking/eval/gt_sam_gate.py:68`). We score that per-frame **modal** output against the modal
GT, never S itself.

## 2. Data

- **Pilot:** seq1, segment 4 = frames 1154–1308 and segment 5 = frames 1476–1621 (`find_segments`
  on `data/nfo_processed/seq1_gt/groundtruth.txt`). The two are consecutive, so they cover one
  crossing in each direction.
- **GT:** `{idx:05d}_sammask.png` (224×224), the pseudo-GT from `gen_nfo_pseudo_masks.py`. For the
  pilot it is treated as ground truth, and the label bias is accepted. `*_sammask_ext.png` frames
  are excluded.
- **Background prior (our method only, since it is the only one that uses it):** the 40 most
  recent person-free frames before t₀. A frame is person-free if it lies more than 40 frames from
  every GT segment. Pilot ranges: f946–f1113 (before segment 4) and f1349–f1435 (before segment 5).

## 3. Trials

- **Starts:** t₀ = start, start+10, start+20, …, continuing while t₀ + 50 ≤ the segment's last
  `_sammask.png` frame. Segment 4 gives 11 starts (1154…1254) and segment 5 gives 10 (1476…1566),
  21 in total.
- **Admissibility:** a start t₀ is kept iff D_{t₀}(p*) ≥ 2 px (224 space). Here
  p* = argmax_{x∈M_{t₀}} D_{t₀}(x) and D is `cv2.distanceTransform(M, DIST_L2, 5)`.
- **Prompt:** the GT box at t₀ (native px via `gt_to_native`, `gen_nfo_pseudo_masks.py:59`).
  Methods that accept a point also get p* (scaled to native). Nothing is given after t₀.
- **Propagation:** forward from t₀, strictly causal, no look-ahead.
- **Scoring:** every frame in [t₀, t₀+50] for every method.
- **Per-trial covariates:** v(t₀) = |M_{t₀}| / Ā, where Ā is the median |M| over the segment's
  `confirmed_clear_frames` (`nfo_visibility.py:116`). Also n_frag(t₀) = the number of connected
  components of M_{t₀}.

## 4. Methods

All SAM2-family calls use **`sam2.1_hiera_base_plus.pt`**, the only SAM2.1 checkpoint present
locally besides `small`. One backbone for all methods. It is also not the `large` model that
generated the pseudo-GT, which slightly reduces the shared-model advantage.

| ID | Method | Environment | Prompt at t₀ |
|---|---|---|---|
| B0 | vanilla SAM2.1 video predictor | `../master_thesis/.venv` (sam2 1.1.0) | box + p* |
| T1 | SAMURAI | `../samurai/.venv` (repo commit `76ba195`) | box |
| T4 | ours, one-shot causal | same as B0 (confirm NFO-UNet imports there) | box |
| T4-1 | ours, **integration off** (control) | same as B0 | box |

**T4, online definition:**
- Buffer: up to the last 7 **consecutive** frames (`NTH_FRAME = 1`), empty at t₀. While it holds
  fewer than 7 frames, the method still outputs every frame from what is there.
- Track at t₀: among the blob detections at t₀, pick the one whose centre lies inside the GT box
  (the largest overlap if several). The box also sets the initial position and the expected height.
- Velocity: only from the selected track's own past detections inside the buffer. Use the
  existing Kalman/Hungarian `track_blobs` plus the OLS over the in-buffer history
  (`position_from_track`, `tracking/core/track_window.py:7`) with readout at the newest frame. This
  replaces `build_gt_winner` (`tracking/eval/gt_integrated_image.py:38`), which used GT boxes and
  a whole-run OLS.
- If the track dies, re-acquire the blob nearest to the last predicted position.
- Segmentation: SAM2 image model (`segment_reference`, `tracking/eval/gt_sam_gate.py:60`) on the
  integrated reference, prompted with the point at the tracker position, then the per-frame
  visibility step. The output is the modal mask for frame t.

**T4-1 (integration off):** identical to T4, except the reference image is the current frame
alone (buffer length 1 for the reference; the tracker still uses its 7-frame buffer). This is the
one-variable control for the question in §1. Without it, a win for T4 over B0 could come from the
tracker, the image-vs-video SAM2 mode, or the visibility step, not from integration.

## 5. Metrics and analysis

- J (region IoU) and F (boundary) from the official DAVIS-2017 evaluation code, keeping its
  handling of empty GT and empty predictions. Report how many frames have empty GT.
- Metrics: J@Δt for Δt ∈ {10, 25, 50}, plus mean J&F over [t₀, t₀+50].
- Analysis: per method, a scatter of J@50 against v(t₀) with an OLS slope. With 2 segments the
  numbers are descriptive only; no confidence intervals in the pilot.
- Comparisons that answer §1: **T4 vs T4-1** (does integration help, everything else fixed) and
  **T4 vs B0/T1** (does it beat memory-based propagation).
- Record per method: wall-clock per frame and peak GPU memory. Hardware: RTX PRO 4000 Blackwell
  Laptop, 16 GB.
- Run record: NFO-UNet commit, method repo commit, environment and config in `run.json`.

## 6. Pilot decision rules (set before the run)

1. B0 J@50 ≳ 0.85 from every start: no failure to study, so stop.
2. T4 ≈ T4-1: integration is not what helps. Revisit the premise before building the feature-fusion
   SAM2 variant.
3. T4 > T4-1 and T4 > B0, T1, especially at low v(t₀): supports §1. Run the full benchmark.
4. T1 ≫ B0: the motion-aware memory already fixes a large part of the failure. SAMURAI becomes the
   baseline the future variant must beat.

## 7. Engineering

- Cache each propagation to disk as `.npz` masks per (method, trial). Analysis reruns must not
  re-infer.
- Visual traces: per trial, a 6-frame montage over [t₀, t₀+50] (prediction vs GT), about dpi 80,
  under `images/benchmark/pilot/`.
- Results under `results/benchmark/pilot/`.
- B0 and T1 run in separate environments: each writes masks to the cache, and one scorer reads all
  of them.

## 8. Deferred (keep in mind, not in the pilot)

- Ablation ladder decomposing the pseudo-GT reference (re-prompting at clear frames → GT-box clip
  → backward pass), to explain why modal memory-based variants fail.
- DAM4SAM (checkpoint already at `../DAM4SAM/checkpoints/`) and SAM2Long.
- Manual audit (labelme or CVAT, from scratch, modal, AI assist off).
- H-mem instrumentation (SAMURAI gating scores against v(t)).
- Full run: 32 segments, every admissible start, v-tertile edges computed on the pooled trials,
  segment-level bootstrap.
- Larger buffer for T4. Zero-shot/unprompted track. Amodal comparison (e.g. amodal VOS methods)
  only if S itself is ever reported. `_ext` frames.
