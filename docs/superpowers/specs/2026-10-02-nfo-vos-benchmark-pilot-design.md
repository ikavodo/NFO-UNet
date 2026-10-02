# NFO fragmented-occlusion VOS benchmark: pilot protocol

Status: design, awaiting review (2026-10-02). Scope: the **pilot** only (one back-and-forth of one
sequence). A section at the end covers the full 32-segment run, which is set after the pilot.

## 1. Question

Under fragmented occlusion, do SAM2 variants designed for occlusion (SAMURAI, DAM4SAM, SAM2Long)
beat vanilla SAM2 when all get the same single causal prompt? Where does each one fail, as a
function of how visible the person is at the prompt frame? Secondary questions:

- Which ingredient of the pseudo-GT reference closes the gap between reference and causal?
- Does a causal, one-shot version of our integrated-image method do better?
- Hypothesis H-mem: failures start where the memory-gating scores of SAM2-family methods drop
  during an occluder crossing.

## 2. Data

- **Pilot:** seq1, segment 4 = frames 1154–1308 (155 f) and segment 5 = frames 1476–1621 (146 f).
  The two are consecutive, so they cover one crossing in each direction. Both indices come from
  `find_segments` on `data/nfo_processed/seq1_gt/groundtruth.txt`.
- **Reference ("GT"):** `{idx:05d}_sammask.png` (224×224) from `gen_nfo_pseudo_masks.py`
  (sam2.1-hiera-large, `union_gt_outlier`). Each frame is labelled by modal segmentation, i.e.
  visible pixels only. `*_sammask_ext.png` frames are **excluded** (deferred).
- **Background prior (allowed for every method, declared):** the 40 most recent person-free frames
  before t₀. A frame is person-free if it lies more than 40 frames from every GT segment, because
  annotation stops while the person is still partly visible. Pilot ranges: f946–f1113 (before
  segment 4) and f1349–f1435 (before segment 5).

## 3. Trials

- **Starts:** t₀ ∈ {start, start+10, …} ∩ [start, end−50]. That gives about 10 per segment, ~20 in
  total.
- **Admissibility:** a start t₀ is kept iff D_{t₀}(p*) ≥ 2 px (224 space). Here
  p* = argmax_{x∈M_{t₀}} D_{t₀}(x) and D is `cv2.distanceTransform(M, DIST_L2, 5)`. No other
  filter.
- **Prompt:** the GT box at t₀ (native px via `gt_to_native`, `gen_nfo_pseudo_masks.py:59`).
  Methods that accept a point also get p*, scaled to native. This is the only information given
  after t₀ = 0.
- **Propagation:** forward from t₀ to the segment end. Strictly causal, latency 0.
- **Scoring:** every frame from t₀ to t₀+H for every method. No ramp-up exclusion.
- **Per-trial covariates:** v(t₀) = |M_{t₀}| / Ā, where Ā is the median |M| over the segment's
  `confirmed_clear_frames` (`nfo_visibility.py:116`). Also n_frag(t₀) = the number of connected
  components of M_{t₀}.

## 4. Methods and information budgets

| ID | Method | Prompt at t₀ | After t₀ |
|---|---|---|---|
| B0 | vanilla SAM2.1-hiera-large video predictor | box (+p*) | nothing |
| T1 | SAMURAI | box | nothing |
| T2 | DAM4SAM | box | nothing |
| T3 | SAM2Long | box (+p*) | nothing |
| T4 | ours, one-shot causal | box | nothing (+ background prior) |

Use the SAM2.1-hiera-large weights in every SAM2-family method that supports them; record any
method that can't. Before vendoring a method, check its official repository and commit. Port each
method's preprocessing exactly and flag every deviation.

**T4, ours, online definition:**
- The buffer is empty at t₀ and holds up to the last 7 sampled frames at stride 2, so it spans 13
  frames. These are the existing `SEQ_SIZE=7, NTH_FRAME=2`; they are kept so the method under test
  is unchanged.
- Output at every frame t ≥ t₀, from whatever the buffer holds; it is shorter during the first
  frames.
- The Kalman filter and Hungarian association (`track_blobs`) run causally. Velocity comes only
  from the filter state up to t. **No whole-run OLS** (replaces `build_gt_winner`,
  `tracking/eval/gt_integrated_image.py:38`).
- At t₀: pick the track whose blob centre lies inside the GT box (largest overlap if several).
  After that, follow its track id. If the track dies, re-acquire the blob nearest to the Kalman
  prediction.
- Segmentation: SAM2 *image* model on the integrated reference, with the point at the tracker
  position (`segment_reference`, `tracking/eval/gt_sam_gate.py:60`), then the existing per-frame
  visibility step.
- Readout is the newest frame. The `center` readout with 6-frame look-ahead
  (`tracking/stream/stream.py:14`) is not used.

## 5. Ablation ladder: decomposing the reference

Each rung adds one ingredient of the reference to B0:

| Rung | Adds | Causal? |
|---|---|---|
| A0 | = B0 | yes |
| A1 | + a GT box re-prompt at every geometric clear-corridor checkpoint t_c > t₀, applied when the stream reaches t_c (`geometric_checkpoints`, `nfo_visibility.py:130`) | yes (uses oracle boxes) |
| A2 | + clip each predicted mask to the GT box dilated by 3·(native/224) px, then a component width filter (same as `combine_checkpoint_masks_union_gt_outlier`) | yes (uses oracle boxes) |
| A3 | + backward propagation from each checkpoint, union-combined | no (≈ reference) |

## 6. Metrics and analysis

- J (region IoU) and F (boundary) from the official DAVIS-2017 evaluation code, with its handling
  of empty GT and empty predictions. Report how many frames have empty GT.
- Metrics: J@Δt for Δt ∈ {10, 25, 50} after t₀, plus mean J&F over [t₀, t₀+50].
- **Pilot analysis:** scatter J@50 against v(t₀) per method, with an OLS slope (robustness to the
  starting frame). Two segments are too few for a segment-level bootstrap, so pilot numbers are
  descriptive.
- Record wall-clock per frame, peak GPU memory and parameter count for each method.
- Run record: commit, method repo commit, GPU and config in a `run.json` next to the results.

## 7. Audit (manual, from scratch)

- 15–20 pilot frames: about a third each with low, mid and high v(t), across both segments.
- Annotate modal masks at native 800×600 in labelme or CVAT with AI assist **off**, without
  pseudo-mask pre-fill. Downsample with `scale_and_pad_img_to_square`.
- Report J/F between hand masks and pseudo-masks per v-bin. That is the noise floor.

## 8. Mechanism instrumentation (H-mem)

For T1 (and B0 where exposed), log the per-frame predicted IoU, the object score, and whether the
frame entered memory. Plot these against v(t) and against the first frame where J < 0.5.
H-mem is unsupported if collapses start without a change in the gate signal.

## 9. Pilot decision rules (set before the run)

1. B0 J@50 ≳ 0.85 from every start: no failure to study, so stop.
2. The J@50–v(t₀) slope is about 0: drop stratification.
3. A clear positive slope: keep v-tertiles for the full run.
4. T1–T3 ≈ B0: consistent with the claim; analyse H-mem.
5. Any of T1–T3 ≫ B0: narrow the claim before the full run.

## 10. Engineering

- Cache every propagation to disk, as `.npz` masks per (method, trial). Analysis reruns must not
  re-infer.
- Visual traces: per trial, a montage of 6 frames from t₀ to t₀+50 (prediction vs GT) at about
  dpi 80, saved under `images/benchmark/pilot/`.
- Results go under `results/benchmark/pilot/`.

## 11. Full run (after the pilot; parameters may change based on its results)

- All 32 segments (4 sequences × 8), with the same starting rule: about 300 trials.
- Every admissible start is used. v(t₀) is not used to pick starts, only to bin trials. The
  tertile edges are computed once over all pooled trials, so a bin means the same in every
  sequence.
- Segment-level bootstrap CIs (32 units).
- The audit extends to about 10 frames per sequence, stratified by v-tertile.

## Out of scope (for now)

Zero-shot/unprompted track (our method without a box; salient-motion baselines). `_ext` frames.
Look-ahead/fixed-lag variants.
