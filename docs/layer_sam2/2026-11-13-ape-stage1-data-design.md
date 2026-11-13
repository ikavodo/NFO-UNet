# Layer-SAM on apes, Stage 1: planted real-foliage training data (design, 2026-11-13, branch exp/ape-stage1)

Status: **design, no code.** It must pass the expert pipeline review (CLAUDE.md) before implementation.
Related notes on branch `exp/feature-integral`: `docs/layer_sam2/2026-11-13-feature-integral-design-notes.md`.

## 1. Question and pre-registered gate
**Question.** Layer-SAM is the frozen SAM2.1 plus the 730k-parameter adapter fed by the integral image. If it is trained
only on **clear** PanAf apes with **planted, static, real-foliage occluders**, does it recover visible ape pixels that
SAM2 misses on **real** occluded videos?

**Primary metric.** Visible-part J on `results/panaf/eval_subset.csv` (in PanAf20K; 74 occluded, 74 partial and 74
clear videos), with the PanAf-SBR masks as GT. Mean over 3 seeds, video-bootstrap CI.

**Pass.** All three conditions must hold:
- ΔJ(occluded) against SAM2.1 under the same prompt protocol: CI excludes 0.
- ΔJ(clear) ≥ −0.01.
- Recall(occluded) improves.

Secondary metrics:
- precision and recall (`tools/mask_pr.py`);
- box IoU and recall of the mask box against the amodal PanAf500 boxes.

**Reference measured today (SAM2.1, first-frame GT mask, 121 frames).** J = 0.926 / 0.879 / 0.742; recall on
occluded 0.832, precision 0.860.

## 2. What exists and is reused unchanged
- The `clip_raw` → `build` → `encode` path of `adapter/fusion_data.py`, the model, loss v3.4 and the training loop of
  `adapter/fusion_adapter.py`.
- The new generator returns the **same raw-clip dict** as `clip_raw`:
  - X: frames;
  - bg: B;
  - V: visible mask;
  - S: silhouette;
  - boxes and jittered boxes `bxj`;
  - cid, modes.
- So **only the data source changes** (one variable).
- N = 40 frames per clip, as in `dataset.py`.

## 3. Sources
- **Ape sources.**
  - The 139 PanAf500 training videos labelled clear that are outside the eval subset.
  - The 80 PanAf-SBR training-only videos, once their sbr100 labels exist (clear ones only).
  - Silhouette S_t = the SBR mask of the chosen ape. Boxes = the human PanAf500 / SBR boxes, which are amodal.
  - Checkpoint selection uses planted clips on the 9 clear validation videos only, never the eval subset.
- **Foliage sources.** Exclude every eval-subset video and the 63 FGBG videos that are PanAf-SBR re-encodes.
  - PanAf500 videos of any class: frames with GT ape boxes, dilated 15%, removed from the near layer.
  - FGBG background clips, `disjoint` train split; these contain no apes.
  - Foliage-source split: 90% train / 10% val **by video/camera**, for the memorisation test (§7).
- **Offline occluder pool (computed once).** Depth Anything V2 Small (frozen) on one frame per source video. Measured
  at 1.6 ms/frame; about 6k frames, so minutes. Per source keep:
  - the RGB;
  - a soft near-layer alpha α = sigmoid((D − 0.8)/0.03), with D rank-normalised per frame;
  - an IR flag (mean chroma < 4);
  - fragmentation statistics: components, boundary/area, coverage.

  Stored as PNG + an index CSV.

## 4. Planting one clip (CPU worker, seeded)
1. **Pick the clip.** An ape video and ape id, and a 40-frame window where the ape has a box in ≥ 90% of frames. Do
   not filter on motion: the real motion mix is kept, and motion per clip is recorded for the analysis.
2. **Region.** R = union of the ape's boxes over the window, plus a 10% margin.
3. **Foliage field over R.**
   - Prefer a single pool crop with an aspect within ±30% of R's, resized to R (scale 0.7–1.6× relative to the
     box, so leaves keep a plausible size).
   - Otherwise a mosaic of 2–4 crops with feathered alpha (Gaussian, σ = 3 px).
   - The IR flag must match the target video.
   - Augmentation: flip, rotation ±10°, brightness/contrast jitter ±15%.
4. **Composite.** The occluder is static, so the same field is used in every frame and in B:
   X'_t = α·F + (1 − α)·X_t and B' = α·F + (1 − α)·B.
5. **Labels.**
   - Hidden H_t = S_t ∧ (α ≥ 0.5); visible V_t = S_t ∧ (α < 0.5).
   - The three pixel classes for loss v3.4 follow from V, S and the background.
6. **Accept / reject.** h_t = |H_t| / |S_t|. Reject if h_t lies outside [0.15, 0.75] for more than 6 consecutive
   frames.
   - 10% of clips are left unoccluded (as in `sample_occlusion`), so the adapter learns that the correction R ≈ 0
     where nothing is hidden.
7. **Background B** = per-pixel median of the window's planted frames. This matches the estimator used at test time,
   for train/test parity.
   - Known limit: a static ape stays inside B. That is recorded, not fixed.
   - Pre-computed per video, unplanted, then composited with α: exact where α ∈ {0, 1}.

## 5. Alignment and prompts
- **Training.** The tracker track is the GT boxes plus jitter (σ = 3 px, as for `SIGMA_A`) plus a random scale of
  ±10%. The integral image shift is the x and y Theil–Sen fit to that track.
- **Test.** Two arms:
  - (i) GT boxes, jittered the same way: the ceiling of the integral image alone;
  - (ii) MD-seeded SAM2 boxes: deployable.
  - The gap between (i) and (ii) is the cost of tracking.
- **Note.** The current integral image shifts in x only (`integral_image.py`). Apes climb, so the shift becomes x and
  y. That is a code change to review.

## 6. Arms (only the occluder source changes)
| arm | occluder |
|---|---|
| P | procedural (`synth/occluders.py`), same ape clips |
| R | real foliage (this design) |
| O | the real-foliage α field, but filled with B instead of F ("turned-off pixels") |

Plus the SAM2.1 reference and the `-noC` control (integral image removed at inference) for arm R.

## 7. Checks before the first training run (cheap, no training)
1. **Fragmentation match.** Over 500 planted clips per arm P / R / O, compute the distributions of:
   - visible fragments per box;
   - boundary/area of V;
   - h_t.

   Compare them with the real occluded SBR masks inside the PanAf500 boxes. Use the Wasserstein distance per
   statistic.
   - Expected, but not required: R is closest.
   - If P is closer on all three, report it and reconsider before training.
2. **Visual montage** of 20 planted clips (3 frames each), for the user to judge.
3. **Memorisation test**, at evaluation time: planted validation J on seen vs unseen foliage sources.

## 8. Compute
- Per clip, the CPU decodes 40 frames at 720×404 (sequential read from a seek; the time is to be measured), plus the
  composite.
- The GPU side is unchanged: two encoder passes per frame. Training time per step is about that of NFO.
- The cluster is the default for training (3 seeds × 1 run).

## 9. Open points for the review
- Window length 40 vs longer: B is from the whole window, not the whole video. Is the parity correct?
- The x+y shift change to the integral image, and whether the tracker scale should enter the shift.
- Is the soft alpha threshold at 0.5 the right definition of "hidden"? Report the sensitivity at 0.3 and 0.7.
- Loss v3.4 weights were tuned for NFO pseudo-GT: keep them fixed (one variable) or re-check?

## Expert pipeline review (2026-11-13), proposed changes (adopt after the user's decisions D-a..D-c)
**Correction.** "Only the data source changes" was false. Colour, B, the alignment source and the box semantics
changed too.

**Blockers.**
1. **Gray vs colour.** The pipeline is grayscale (`fusion_data.py:107,112,132`; `inference.py:67`); the SAM2
   reference ran on RGB. **D-a**: gray everywhere (recommended for Stage 1, baselines rerun on gray), or port to RGB
   behind `parity_snapshot`.
2. **Gate control.** The control is the step-0 `trk` arm: the SAM2 image predictor with the same tracker prompts,
   windows and frames. The GT-mask-seeded SAM2-video reference (`sam2_gap`) becomes secondary.
3. **One B estimator** for train and test: the whole-video median. Record the ape-in-B fraction per clip and
   stratify results by it. A static ape gives gate z ≈ 0 and an empty tracker map.
4. **Planted foliage must not be noise-free.** Add per-frame noise and flutter (`synth/noise`), or re-encode with
   x264. Otherwise the shortcut "zero temporal variance = occluder" is available.
5. **The depth α is the top 20% by construction** (rank normalisation), often ground. Gate pool sources on
   fragmentation and vegetation, and run the §7.1 check on the accepted pool.
6. **Camera leakage.** Cluster all videos by background similarity and drop any source that shares a cluster with an
   eval video.

**Should-fix.**
7. Fix step 200 a priori, or select on about 10 real occluded non-eval videos. Synthetic val misranked losses on NFO.
8. Calibrate the training jitter to the measured error of the MD-seeded SAM2 boxes. Put the x+y shift behind a flag
   (NFO stays x-only; `parity_snapshot` IDENTICAL).
9. Train with tight boxes of S, as on NFO (`dataset.py:389`). The amodal human box becomes a test-time variant only.
10. SBR masks are biased GT on occluded videos. **D-b**: the primary GT for that class is SA-FARI verified 6 fps
    masks, or about 30 hand frames (`cvat_io`).
11. Log the fraction of H with loss weight 0 under v3.4 (8 px near band). If it is above 0.7, plan a NEAR_PX
    ablation.
12. P = procedural α with the same F texture pool. R−P then isolates shape; R−O isolates texture.
13. Add arm **R0**: everything identical, but 100% unoccluded. This is the domain-adaptation control.

**Nice-to-have.**
- The gate estimator: pool the seeds, then a paired video bootstrap; clear CI lower bound > −0.02.
- Reject windows where another ape's mask overlaps R. Prefer FGBG empty clips as foliage sources (deleted boxes
  leave unboxed apes).
- Rotate an oversized crop before cutting, so no black corners appear.
- A void band: W = 0 for 0.3 < α < 0.7.
- Measure the seek/decode time per clip. Above 0.5 s, cache gray memmaps (about 23 GB) with B precomputed.

**D-c (user):** accept the scope growth. These fixes make Stage 1 several days of work, not one.

## User decisions on the review (2026-11-13)
- **D-a: gray** for Stage 1 (pseudo-IR). All arms and baselines run on gray.
- **Comparison:** Layer-SAM (per frame, no memory) against **SAM2.1 video** (memory), with the same initial prompt and
  frames, on gray. Claim: memory is not needed under partial/occluded conditions. The adapter is zero-initialised as
  on NFO.
- **Foliage motion:** cut the same patch from consecutive frames of the source video (frames with no ape in the
  region). Real sway and compression noise come with it, so synthetic flutter is not needed.
- **D-b: the SBR masks are the GT, fill-in included.** The adapter is expected to complement what SAM2 misses under
  occlusion, as the FN-weighted Focal Tversky loss already encodes.
- **Leakage:** handled with background-similarity clusters (to be implemented with the generator).
- Near-layer vegetation filter: accepted.
