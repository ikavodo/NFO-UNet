# NFO fragmented-occlusion VOS benchmark: pilot protocol

Status: design, revision 4 (2026-10-05): DAVIS headline, init failure rate, T4 prompt fix. Rev. 3 (2026-10-05; rev. 3 adds the TRE/anchor framing, the
frame-fixed-effects init analysis, drift metrics, and the scoring resolution). Scope: the **pilot** only (one
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
  pilot it is treated as ground truth, and the label bias is accepted. Edge-extension frames are
  excluded from the pilot (segments 4–5 use none). **Full run:** wherever an extension mask is
  loaded, use `{idx:05d}_sammask_extfix.png` when it exists (occluder pixels removed by
  `gen_data/nfo_pseudo_masks/fix_ext_occluder.py`), else `{idx:05d}_sammask_ext.png`, the same
  precedence `tracking/visualize/nfo_sammask_videos.py:42-43` uses.
- **Resolution.** SAM2-large computed the pseudo-GT at native 800×600, but it is stored only at
  224×224, after `scale_and_pad_img_to_square` (`gen_nfo_pseudo_masks.py:346`). All methods are
  scored at 224. B0 and T1 run on the **native** frames (`data/nfo_final/nfo_final`, the frames the
  pseudo-GT came from), never on the 224 frames. SAM2 resizes every input to 1024
  (`image_size: 1024` in `sam2.1_hiera_b+.yaml`), so a 224 input would be upsampled about 4.6×,
  which would handicap the baselines. Their native masks are mapped to 224 with the same
  `scale_and_pad_img_to_square` call as the GT. T4 works in 224 space and is scored there directly.
  The cost: 1 px at 224 ≈ 3.6 native px, so fragments narrower than that are lost, for all methods
  equally. Moving to native resolution is planned for after the pilot (§8).
- **Background prior (our method only, since it is the only one that uses it):** **all**
  person-free frames before t₀ in the current inter-segment gap (minimum 20), used as the warm-up
  for the existing background model. Fixed by this rule, not tuned on pilot results. A frame is
  person-free if it lies more than 40 frames from every GT segment. Pilot ranges: f946–f1113
  (168 frames, before segment 4) and f1349–f1435 (87 frames, before segment 5).

## 3. Trials

- **Starts:** t₀ = start, start+10, start+20, …, continuing while t₀ + 50 ≤ the segment's last
  `_sammask.png` frame. Segment 4 gives 11 starts (1154…1254) and segment 5 gives 10 (1476…1566),
  21 in total.
- **Protocol lineage (no new protocol).** The design combines OTB's **TRE** (restart the tracker at
  different frames; Wu, Lim, Yang, CVPR 2013 / TPAMI 2015) with VOT2020's **anchor** protocol
  (initialise at several anchor frames, run without resets; Kristan et al., VOT2020 results,
  ECCVW 2020). OTB's **SRE** perturbs the initial box synthetically. Here the natural variation of
  v(t₀) across starts plays that role, so the perturbation comes from the real occlusion. v(t₀) is
  a measured covariate, not a design factor.
- **Why stride 10 (80% window overlap).** The overlap is required. Each frame is then scored by
  K = 50/10 = 5 trials that see identical content and differ only in their start, and this is what
  lets §5 separate the start's effect from the frame's difficulty. VOT's ~50-frame anchor spacing
  would give K = 1 and lose that. Overlapping trials are not independent: the resampling unit is
  the segment (segment-level bootstrap in the full run), never the trial.
- **No resets after failure.** VOT dropped the reset protocol in 2020 for the anchor protocol. A
  reset is also unfair here, for three reasons. (i) It injects GT at the moment of failure. This
  helps the methods that cannot re-acquire on their own (B0, T1) and gives nothing to T4, which
  re-acquires without GT. (ii) Failures happen in occluded frames, so the reset prompt itself has
  low v, which feeds the covariate back into the outcome. (iii) It needs a failure threshold on
  modal masks, which are tiny under heavy occlusion, so spurious failures would follow. Failure is
  measured (DRE/NRE, §5), never acted on.
- **Admissibility:** a start t₀ is kept iff D_{t₀}(p*) ≥ 2 px (224 space). Here
  p* = argmax_{x∈M_{t₀}} D_{t₀}(x) and D is `cv2.distanceTransform(M, DIST_L2, 5)`.
- **Prompt:** every method gets the **same** box + p* at t₀ (native px via `gt_to_native`,
  `gen_nfo_pseudo_masks.py:59`), sent through the same `add_new_points_or_box(points, labels, box)`
  call. SAMURAI's predictor keeps this signature
  (`../samurai/sam2/sam2/sam2_video_predictor.py:173`), although its released scripts pass the box
  alone (`scripts/main_inference.py:82`): **a flagged deviation from SAMURAI's protocol.** Ours uses
  the box for track selection; its SAM prompts come from the tracker. Nothing is given after t₀.
- **Propagation:** forward from t₀, strictly causal, no look-ahead.
- **Scoring:** every frame in [t₀, t₀+50] for every method.
- **Per-trial covariates:** v(t₀) = |M_{t₀}| / (κ·|B_{t₀}|), where B_{t₀} is the amodal GT box
  (224 px) and κ is the median fill ratio |M|/|B| over the segment's `confirmed_clear_frames`
  (`nfo_visibility.py:116`). That makes v the visible fraction of the expected full-body area at
  t₀'s own scale (`visibility`, `benchmark/nfo_vos/trials.py`). Rev. 2 used |M_{t₀}| / median|M|,
  which mixed apparent size with visibility: in segment 5 the box height falls from 83 to 51 px as
  the walker recedes, and that v reached 2.6. Also n_frag(t₀) = the number of connected components
  of M_{t₀}.

## 4. Methods

All SAM2-family calls use **`sam2.1_hiera_base_plus.pt`**, the only SAM2.1 checkpoint present
locally besides `small`. One backbone for all methods. It is also not the `large` model that
generated the pseudo-GT, which slightly reduces the shared-model advantage.

| ID | Method | Environment | Prompt at t₀ |
|---|---|---|---|
| B0 | vanilla SAM2.1 video predictor | `../master_thesis/.venv` (sam2 1.1.0) | box + p* |
| T1 | SAMURAI | `../samurai/.venv` (repo commit `76ba195`) | box + p* |
| T4 | ours, one-shot causal | same as B0 (confirm NFO-UNet imports there) | box |
| T4-1 | control: SAM2 image, no integration | same as B0 | box |
| T4-2 | control: warped blob vote, no SAM2 | same as B0 | box |
| T4-3 | control: frame-t blob, no SAM2, no integration | same as B0 | box |

The controls form a 2×2 with T4. All four share one tracker run (same track, same tracker
point), so they differ only in the two factors:

| | integration on | integration off |
|---|---|---|
| SAM2 image model | T4 | T4-1 |
| no SAM2 (blobs) | T4-2 | T4-3 |

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
- Reuse `build_tracker_winner` (`tracking/eval/gt_sam_gate.py`, no GT), restricted to the
  causal buffer instead of the whole segment.
- Segmentation: SAM2 image model, **switched from `sam2.1-hiera-small` (`gt_sam_gate.py:50-53`) to
  base_plus**, on the integrated reference, then the per-frame visibility step. **Prompt
  (rev. 4):** at t₀ it is the shared GT box + p\* (§3). After t₀ the same rule is applied to the
  method's own evidence, with no GT: the distance-transform maximum of the tracker's person blobs
  at frame t, plus their merged box (`prompt_in_crop`, `benchmark/nfo_vos/run_t4.py`). Run 1
  prompted at the tracker's merged-box centre instead. That point was inside the GT box on 94% of
  frames but on the person on only 58%, because a fragmented person's box centre falls in gaps and
  on occluders, so SAM2 segmented the occluder. The output is the modal mask for frame t.
- **Init sensitivity by design.** The prompt only selects T4's track. After that, T4 re-detects
  from blobs every frame. A weak T4 response to v(t₀) is expected from the design and is not a
  finding. The init-robustness question (§5) is about B0 and T1, and T4 is the reference that
  depends on the prompt only weakly.

**T4-1 (integration off):** identical to T4, except the reference image is the current frame
alone. The tracker still uses its 7-frame buffer. The residual |frame − reference| is then 0
everywhere, so the visibility step passes S through unchanged. T4-1 therefore reduces to the SAM2
image model on the raw current frame, prompted at the tracker point, per frame, with no memory and
no integration. Expected to do badly. Its job is the "SAM2 without integration" baseline.

**T4-2 (warped blob vote):** for each buffered frame t−k, shift the tracked person's foreground
blob by the tracker displacement (cx_t − cx_{t−k}, cy_t − cy_{t−k}) into frame t. A pixel joins
S_blob if at least m of the N buffered warped blobs cover it. S_blob then goes through **the same
visibility step as T4**, because the vote fills in occluded pixels (near-amodal), and raw S_blob
would be penalised against the modal GT. Membership is left open: evaluate every
m ∈ {1 (union), …, N (intersection)}, N = 7, and report **T4-2\* = the best m by mean headline J&F over
the whole pilot** (one m for all trials, not per trial). The best-of-grid tuning favours the
control, so T4 > T4-2\* is a conservative result.

**T4-3 (frame-t blob):** the tracked person's foreground blob at frame t (`foreground_mask` /
`refine_mask` output), scored directly. No SAM2, no integration, no extra compute. This is the
one-variable control for the question in §1. Without it, a win for T4 over B0 could come from the
tracker, the image-vs-video SAM2 mode, or the visibility step, not from integration.

## 5. Metrics and analysis

All metrics are computed from the cached masks (§7), per frame in [t₀, t₀+50]. None of them needs
a new inference run.

- **Per frame:** J (region IoU) and F (boundary) from the official DAVIS-2017 evaluation code,
  keeping its handling of empty GT and empty predictions. Report how many frames have empty GT.
  Also precision p = |P∩G|/|P|, recall r = |P∩G|/|G|, and |P|. Since 1/J = 1/p + 1/r − 1, J alone
  cannot separate **drift** (p collapses while P is non-empty) from **fragmentation-induced
  under-segmentation** (r collapses while p stays high). p and r can.
- **Headline (DAVIS semi-supervised convention):** mean J, F and J&F over frames t₀+1 … t₀+49.
  The first (prompted) and last frames are dropped, as `davis2017/evaluation.py:85` does
  (`all_gt_masks[:, 1:-1]`). Every frame has equal weight, hard ones included. DRE, NRE, P_norm,
  decay and the fixed-effects fit use the same frame range.
- **Init failure rate:** the number of starts (of 21) whose mask at the prompted frame t₀ has
  J < 0.5, the standard IoU success threshold (OTB success rate, PASCAL overlap). It separates
  "cannot get the person from the prompt" from "loses the person later", which the headline mixes.
  It is reported beside the headline on all starts. There is no v threshold and no subset analysis
  conditioned on init success: each method fails on different starts, so per-method subsets would
  compare means over different trial sets.
- **Trajectory (secondary):** single-frame J@10/25/50, the discrete form of VOT's expected-overlap
  curve Φ(N_s) (Kristan et al., arXiv 1503.01313). Descriptive only, and not windowed (rev. 4
  decision: one hard frame is allowed to count).
- **Drift and loss (VOTS2023 definitions, `data.votchallenge.net/vots2023/measures.pdf`):**
  DRE = fraction of GT-present frames where P is non-empty and P∩G = ∅ (drifted, still claiming the
  target). NRE = fraction of GT-present frames where P is empty (reported absent).
- **Decay:** DAVIS J_D and F_D (Perazzi et al., CVPR 2016), the citable one-number drift summary.
  It inherits the frame-difficulty confound below, and the fixed-effects model is what controls it.
- **Localization against independent GT:** P_norm (LaSOT; Fan et al., arXiv 1809.07845) between
  the centroid of the predicted modal mask and the centre of the `groundtruth.txt` box. An empty P
  counts as a miss. These boxes are human-annotated, amodal, and independent of the SAM2-large
  pseudo-GT, so this metric does not share the label bias of §2. It also links to the original
  paper's centre-distance localization (`eval/`, `max_dist_error`). The offset between a modal
  centroid and an amodal centre affects every method, so read P_norm as a comparison between
  methods. No box IoU or success AUC: AUC equals average overlap (Čehovin et al., arXiv 1502.05803),
  and a box around a modal mask is biased small against an amodal box.
- **Init robustness (confound and fix).** Trials with different t₀ score *different* frames, and
  within a crossing v(t₀) changes with t₀. A raw J@50 ~ v(t₀) regression therefore mixes prompt
  quality with the difficulty of the frames that follow. The fix: frame t is scored by up to 5
  trials (§3) that see identical content. Per method, fit

    J(t | t₀) = a_t + b·v(t₀) + c·(t − t₀) + d·v(t₀)·(t − t₀) + e·n_frag(t₀) + ε

  with one fixed effect a_t per frame, which absorbs that frame's difficulty. Here b is the effect
  of prompt quality on level, c is drift with elapsed time, and d says whether a better prompt slows
  the drift. The hypothesis "a worse prompt → faster drift" is **d > 0**. b > 0 with d ≈ 0 would mean
  "worse throughout" instead. n_frag(t₀) separates edge truncation (n_frag = 1) from foliage
  fragmentation. Caveat: within one frame, v(t₀) and (t − t₀) both vary through t₀, so b, c and d are
  identified only across frames. With 2 segments, all estimates are descriptive, with no confidence
  intervals in the pilot.
- Comparisons that answer §1:
  - Integration effect, tested twice: **T4 − T4-1** (with SAM2) and **T4-2\* − T4-3** (without).
  - Does SAM2 add anything on top of integration: **T4 − T4-2\***.
  - Does integration beat memory-based propagation: **T4 vs B0, T1**.
- Record per method: wall-clock per frame and peak GPU memory. Hardware: RTX PRO 4000 Blackwell
  Laptop, 16 GB.
- Run record: NFO-UNet commit, method repo commit, environment and config in `run.json`.

## 6. Pilot decision rules (set before the run)

1. B0 J@50 ≳ 0.85 from every start: no failure to study, so stop.
2. T4 ≈ T4-1 and T4-2\* ≈ T4-3: integration is not what helps. Revisit the premise before building the feature-fusion
   SAM2 variant.
3. T4 > T4-1 and T4 > B0, T1, especially at low v(t₀): supports §1. Run the full benchmark.
4. T1 ≫ B0: the motion-aware memory already fixes a large part of the failure. SAMURAI becomes the
   baseline the future variant must beat.
5. b, c, d (§5) are reported, not decided on. Two segments cannot settle their signs. The full run
   sets a rule on d.

## 7. Engineering

- Cache each propagation to disk as `.npz` masks per (method, trial). Analysis reruns must not
  re-infer.
- Visual traces: per trial, a 6-frame montage over [t₀, t₀+50] (prediction vs GT), about dpi 80,
  under `images/benchmark/pilot/`.
- Results under `results/benchmark/pilot/`.
- B0 and T1 run in separate environments: each writes masks to the cache, and one scorer reads all
  of them.

## 8. Deferred (keep in mind, not in the pilot)

- **Required baselines for the full run (decided 2026-10-05): SAM-PT and SAM-PD.** These are the
  published per-frame-SAM, no-memory methods, the direct ancestors of T4-1n (tracker prompt → SAM2
  image per frame). SAM-PT (Rajič et al., arXiv 2307.01197, code `SysCV/sam-pt`): a point tracker
  (CoTracker) re-prompts SAM every frame; reported 79.4 J&F zero-shot on DAVIS-2017. SAM-PD
  (arXiv 2403.04194): the previous mask's box is propagated as the next prompt (jittered
  multi-box + point refinement), with no tracking module. Run both with the same shared t₀ prompt
  and the native frames, and swap their SAM for SAM2.1 b+ where the code allows (one variable:
  where the per-frame prompt comes from). This tests our motion tracker's prompts against
  appearance-based prompt propagation, not integration.

- Ablation ladder decomposing the pseudo-GT reference (re-prompting at clear frames → GT-box clip
  → backward pass), to explain why modal memory-based variants fail.
- DAM4SAM (checkpoint already at `../DAM4SAM/checkpoints/`) and SAM2Long.
- Manual audit (labelme or CVAT, from scratch, modal, AI assist off).
- H-mem instrumentation (SAMURAI gating scores against v(t)).
- Full run: 32 segments, every admissible start, v-tertile edges computed on the pooled trials,
  segment-level bootstrap.
- Larger buffer for T4. Zero-shot/unprompted track. Amodal comparison (e.g. amodal VOS methods)
  only if S itself is ever reported. Extension frames (`_extfix` over `_ext`, §2).
- Synthetic SRE (OTB box shifts and scales at high-v starts). This controls prompt quality on
  identical frames, and is worth running only if the natural-v analysis turns out too collinear.
- **Move the whole benchmark to native 800×600 (planned, not optional).** 224 is an artefact of
  repurposing a bounding-box dataset, not a choice made for this benchmark. Native requires
  (i) the native pseudo-GT stored, which the generator does not currently do
  (`gen_nfo_pseudo_masks.py:346` writes only the 224 version), and (ii) T4's 224-calibrated
  parameters rescaled (most are already person-height-relative via `scale_relative_params`).
  Compute argument for staying at 224 is weak: SAM2 encodes every input at 1024×1024
  (`image_size: 1024`), so B0/T1/T4's SAM2 calls cost the same at either size; only T4's
  MOG2/morphology/blob stages scale with pixel count (~9.6× more pixels). Stay at 224 only if a
  measured T4 per-frame time at native is prohibitive. VOTS ADQ, if
  empty-GT frames turn out common.
