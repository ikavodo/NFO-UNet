# NFO video object segmentation under fragmented occlusion: findings (status 2026-10-21)

> **Code paths (2026-10-21 cleanup):** the code moved to `benchmark/nfo_vos/{adapter,baselines,eval,synth,tools}/` (map: `CLAUDE.md`). Script paths cited below that no longer exist (`layers/…`, `downstream/…`, `run_t4.py`, `factorised_*.py`, …) are at git tag `archive/pre-cleanup-2026-10-21`.


Compressed, current view. The full chronological log, with every intermediate test, is in
`archive/nfo_vos_pilot_findings_full.md`; dead ends get one line each in `nfo_failure_log.md`. Numbers are measured on
the data below unless marked otherwise. CIs are 95% window-bootstrap (2,000 resamples, seed 0), paired between methods.

## 1. Task and data

- **NFO.** Four sequences (seq1–4), 800×600 grayscale, fixed camera. One person walks behind dense foliage, so
  the body is visible only in fragments (*fragmented occlusion*).
- **Annotations.**
  - Dense hand-drawn GT **boxes**, about 900 per sequence, covering the whole person.
  - Sparse hand **masks** of the *visible* pixels:
    - seq2–4: 23 frames (`*_handAv.png`; 1 = person, 2 = void/unsure, ignored);
    - seq1: 20 brush masks, which fall in only 3 windows.
- **Protocol.**
  - 61 windows of 40 frames: 15 on seq1 (selection) and 46 on seq2–4 (**single test**).
  - Each window starts with the GT box + one point at t0, the standard semi-supervised VOS prompt.
  - Methods are causal from t0.
- **Task convention:** modal (visible-part) segmentation.

## 2. Methods compared

| id | what it is |
|---|---|
| **B0** | SAM2.1-b+ video predictor (Ravi et al. 2024), prompt at t0 only |
| **T1** | SAMURAI (Yang et al. 2024, arXiv 2411.11922): SAM2 + Kalman motion-aware memory selection |
| **trkbox** | our feature tracker's box alone (no segmentation): a fixed-size kernel at the t0 box size, tracked on SAM2 stride-16 features by maximising 1 − cos(F_t, F_empty-scene), Theil–Sen motion prediction (`layers/track_check.py`) |
| **trk** | per-frame SAM2 (no memory), prompted every frame with the tracker box + point |
| **adapter-pf** | **ours.** `trk` + a layer adapter on the frozen SAM2 (`layers/fusion_adapter.py`); details below |
| vid-prompts | SAM2 video (memory) + tracker box/point every frame (prompts enter the decoder only) |
| vid-adapter-pre | vid-prompts + the same adapter, residual added before memory attention |

**Terminology (2026-10-24).** Two different temporal stacks, two names. *Integral image*: our method's input, the
6 past frames t−6…t−1, not frame t itself (`adapter/integral_image.py`; arm suffix `-noC` = without it). *Recognition
stack*: the evaluation's N = 5 masked frames aligned by GT motion, the same construction for every method
(`recognition.py`, `masked_stack`). Both were called "composite" before 2026-10-24; that word survives only in
`archive/`, in names of code removed in the 2026-10-21 cleanup, and as "composited" (pasted onto a background).

**Layer adapter.**
- **Inputs:** SAM2 encoder features at strides 16/8/4 of three images:
  - the frame;
  - an **integral image** (`adapter/integral_image.py`): frames t−6…t−1 (6 past frames, not frame t), aligned to t by the
    tracker's motion, change-gated against the empty scene and recency-weighted, plus an evidence mass a;
  - the **empty scene** (median of person-free frames).
- **Outputs:**
  - zero-initialised residuals (ControlNet-style) added to the frame features;
  - a correction to the point-prompt token.
- **Training:** about 1 M parameters, trained only on synthetic GPJATK people composited behind foliage. Checkpoint
  v3.4, step 200; layer-aware focal + Focal Tversky loss.

## 3. Evaluation: three metrics, each with a stated blind spot

| metric | question | how | blind spot |
|---|---|---|---|
| **hand J / P / R** | are the visible pixels right? | IoU vs the 23 seq2–4 hand masks, void ignored | sparse (23 frames) |
| **box AUC, P@20, P_norm** | does it track? | LaSOT one-pass protocol, ported from SAMURAI's `extract_results.py` (pytracking); mask → tight box as SAMURAI does | rewards extent beyond the visible pixels: the tracker box alone scores 0.709, while a perfect visible mask's box has mean IoU 0.714 vs GT |
| **recognition AP50 / conf** | does the mask keep the evidence a detector needs once integrated over time? | for each frame t (stride 3): frames t−6…t aligned by **GT-box** motion (same for all methods); uniform masked mean, unmasked pixels filled with the plain aligned mean; YOLO11m person AP50 and matched confidence vs the GT box | rewards near-person recall; a method scores above "no mask" only if its masked pixels beat the synthetic-aperture blur |

**Recognition design checks:**
- Unmasked aligned mean (classic synthetic aperture, cf. airborne optical sectioning, Schedl et al.): AP50 0.818.
- Whole GT box as the mask: lower, 0.778, so covering foliage is penalised.
- A single frame (N = 1): AP50 ≈ 0.3 for every method, so temporal integration is what makes recognition possible.
- The first design filled unmasked pixels with the empty scene. It re-inserted sharp foliage, and masks then lost to
  no-mask (0.48 vs 0.80). Replaced.

## 4. Main result

**Headline (since 2026-10-23): seq2–4 on whole crossings** (full-pass protocol, §9; 24 crossings, recognition N = 5).
This is the table in the talk (results slide). AUC, conf and hand scores: `report.py` block "seq2-4" in
`/storage/slurm/akid/logs/nfo_fullpass_add-1953256.out` (job 1953256, one cluster env). AP50: `recognition.ap50` over
the 1,085 seq2–4 stacks of `fullpass/downstream/recog_N5_uniform_meanfill_fp.csv` (cluster, 2026-11-10; no CI). Ours =
3-seed mean.

| method | hand J | P / R | box AUC | recog AP50 | recog conf |
|---|---|---|---|---|---|
| B0 (SAM2.1 video) | 0.747 | 0.91 / 0.80 | 0.611 | 0.906 | 0.689 |
| T1 (SAMURAI) | 0.741 | 0.90 / 0.80 | 0.615 | 0.904 | 0.687 |
| b0trk (SAM2.1 video + tracker prompts every frame) | 0.757 | 0.83 / 0.89 | 0.694 | 0.893 | 0.646 |
| trk (SAM2.1 image + tracker prompts, no memory) | 0.698 | 0.81 / 0.84 | 0.680 | 0.857 | 0.617 |
| ours without integral image (-noC) | 0.753 | 0.89 / 0.82 | 0.691 | 0.899 | 0.675 |
| **ours** | **0.768** | 0.87 / 0.86 | **0.720** | **0.916** | **0.694** |

Ours − T1 per seed: box AUC +0.102 / +0.109 / +0.103 (every CI > 0); hand J +0.029 / +0.030 / +0.022 (every CI > 0);
recog conf +0.001 / +0.008 / +0.011 (every CI spans 0: on par). Integral image (ours − noC) per seed: box AUC +0.018 /
+0.032 / +0.037, recog conf +0.011 / +0.022 / +0.023, all CIs > 0. Seeds: AUC .717 / .724 / .718, conf .689 / .696 /
.698, AP50 .916 / .916 / .917; noC AUC .699 / .692 / .681, conf .677 / .674 / .675, AP50 .910 / .899 / .887.

**Runtime** (laptop RTX PRO 4000, per frame, `benchmark/nfo_vos/tools/timing.py`; independent of the protocol):
- **ours: 61 ms**, with `adapter/inference.fast_masks` (one encode per frame plus one for the integral image; integral image and SAM2
  preprocessing on the GPU);
- **SAM2 video: 53 ms**.

Masks are unchanged against the benchmark code path: box AUC Δ 0.000, hand J Δ +0.001. The benchmark path itself
takes 445 ms, because it re-encodes everything about 7×.

**Claim supported (seq2–4, whole crossings):** Layer-SAM **tracks clearly better than SAMURAI** (box AUC +0.10, every
seed's CI > 0), **segments the visible person slightly better** (hand J +0.02 to +0.03, every CI > 0; the gain is
recall, 0.86 vs 0.80, at some precision, 0.87 vs 0.90), and is **on par in recognition** (conf, every CI spans 0;
highest AP50, no CI). No other method is at or above T1 on all three: B0 and b0trk lose box AUC or recognition conf.

### 40-frame windows (first protocol, 2026-10-21; superseded as headline, kept for the record)

seq2–4, 46 windows; recognition N = 7 on all 552 stacks. Do not mix these numbers with the table above.

| method | hand J | P / R | box AUC | P@20 | recog AP50 | recog conf |
|---|---|---|---|---|---|---|
| B0 | 0.755 | 0.90 / 0.81 | 0.641 | 0.767 | 0.905 | 0.663 |
| T1 | 0.750 | 0.89 / 0.82 | 0.646 | 0.778 | 0.909 | 0.662 |
| trkbox | 0.250 | 0.25 / 0.96 | 0.709 | 0.840 | 0.755 | 0.461 |
| trk | 0.716 | 0.82 / 0.85 | 0.693 | 0.833 | 0.813 | 0.571 |
| **adapter-pf (ours)** | **0.761** | 0.86 / 0.86 | **0.717** | **0.848** | **0.927** | **0.679** |
| **ours, 3 seeds (mean ± sd)** | **0.760 ± 0.003** | 0.87 / 0.85 | **0.718 ± 0.004** | — | **0.922 ± 0.006** | **0.682 ± 0.004** |
| vid-prompts | 0.748 | 0.82 / 0.89 | 0.713 | 0.859 | — | — |
| vid-adapter-pre | 0.718 | 0.76 / 0.93 | 0.735 | 0.871 | — | — |

All rows except vid-* come from ONE cluster run (2026-10-21, `eval/rescore.sbatch`, log
`results/benchmark/rr/logs/nfo_rescore-1938081.out`): box AUC, hand J and recognition for every final arm in one
environment and one YOLO run. Recognition values differ from the earlier laptop runs (T1 AP50 0.909 here vs 0.916)
by environment drift; differences within the run are what count. vid-* rows are from 2026-10-17 (closed direction).

**adapter-pf − T1:**
- box AUC **+0.070 [+0.056, +0.086]**;
- hand J +0.010 [−0.011, +0.034];
- recognition conf +0.017 [−0.012, +0.044].

**Over 3 training seeds** (seed 0 = adapter-pf; seeds 1, 2 = same loss, step 200; seed 2 trained in fp16 on a Turing
node; all masks from the cluster), against T1:
- box AUC **+0.072 ± 0.004**, and every seed's CI excludes 0;
- hand J +0.010 ± 0.003;
- recognition AP50 +0.013 ± 0.006 and confidence +0.020 ± 0.004 (per seed +0.017 / +0.019 / +0.024, each
  CI touching 0): positive for all three seeds, unresolved for any single one.

Segment-cluster bootstrap for seed 0 (24 segments): box AUC +0.070 [+0.054, +0.088], with the adapter ahead in 24/24
segments.

## 5. What each ingredient does (seq2–4, one variable at a time)

| ingredient | comparison | hand J | box AUC | recog conf |
|---|---|---|---|---|
| motion prompt (tracker box + point every frame) | trk − T1 | −0.034 | **+0.046** | −0.091 |
| empty-scene layer cue | adapter-pf without integral image − trk | **+0.043 [+0.002, +0.099]** | +0.007 | — |
| integral image (past pixels, open-loop) | adapter-pf − adapter-pf-noC | +0.001 | **+0.018 [+0.013, +0.022]** | **+0.024 [+0.013, +0.035]** |
| integral image, **3 seeds** (mean ± sd) | adapter − adapter-noC, per seed | **+0.016 ± 0.013** | **+0.024 ± 0.005** | **+0.025 ± 0.003** (AP50 +0.018 ± 0.005) |
| memory (SAM2 video loop) | vid-prompts − trk | +0.032 | +0.020 | — |
| adapter inside memory | vid-adapter-pre − vid-prompts | **−0.031 [−0.053, −0.011]** | +0.022 | −0.001 |

The −0.031 row is over-segmentation: P 0.76, R 0.93.

Related work for the integral image (AOS ancestor) and for slimmer adapter variants (feature-space integral image, learned
gate): `../layer_sam2/related_work_minimal_adapter.md`.

**Reading:**
- **The motion prompt buys tracking and costs precision.**
- **The empty-scene cue restores the precision.**
- **The integral image adds recall that is real person evidence** (recognition up), not just extent.
  Over 3 seeds it also raises hand J (+0.016 ± 0.013): seed 0's +0.001 was the low seed (seeds 1, 2: +0.022, +0.026,
  CIs excluding 0). Every seed's box AUC and recognition-confidence CI excludes 0.
- **Memory propagates extent:** in the video loop, precision falls and over-extension grows.

**Oracle bound on memory:** SAM2 video with a *non-causal* label written into memory for past frames.

| memory content | hand J | P / R | box AUC | recog conf |
|---|---|---|---|---|
| SAM2-ceiling (GT-box-prompted SAM2.1-L, bidirectional) | 0.756 | 0.87 / 0.84 | 0.668 | 0.689 |
| frozen pseudo-GT (extent-leaning) | 0.726 | 0.78 / 0.90 | — | — |
| *for reference: adapter-pf, no memory* | 0.761 | 0.86 / 0.86 | 0.717 | 0.709 |

The recognition figures here are on the 414-stack subset. **Scope of this bound:** it is only as good as the
content written in. SAM2-ceiling matches the hand masks at J 0.716, *below* the per-frame adapter's own masks (0.761).
So the result is: memory filled with SAM2-ceiling-quality masks does not beat the memory-free adapter, and memory
transmits the precision/recall balance of what is written in. Whether better-than-SAM2-ceiling content (e.g. the
adapter's own masks, or a trained write path) makes memory useful is open; see §8.

**Write-path correction, causal, no training (`vid-write-pf`).** SAM2 video + tracker prompts produces the output, but
the memory encoder is fed the per-frame adapter's mask of the same frame. Seq2–4, recognition on all 552 stacks:

| arm | hand J | P / R | box AUC | recog conf |
|---|---|---|---|---|
| video + prompts (SAM2's own mask written) | 0.748 | 0.82 / 0.89 | 0.713 | 0.627 |
| **video + prompts, adapter mask written** | **0.763** | 0.84 / 0.88 | 0.718 | 0.648 |
| per-frame adapter (no memory) | 0.761 | 0.86 / 0.86 | 0.717 | **0.678** |

- **Correcting what enters memory helps the video output:** J **+0.014 [+0.001, +0.028]** over video + prompts, and
  recognition conf +0.021.
- **It ties the per-frame adapter on J and box** (+0.002, +0.001).
- **It stays below it on recognition:** −0.030 [−0.048, −0.012].
- Memory with adapter-quality content no longer hurts segmentation, but adds nothing over the memory-free adapter.

**Principle (working hypothesis):** closed-loop memory (the model's own past predictions) compounds errors under
fragmented occlusion; open-loop evidence (aligned past pixels vs the empty scene) does not. The architectural
consequence is untested:
- put the correction on the **write path**, so the mask fed to SAM2's memory encoder (`memory_encoder.py:174-177`
  takes only frame features and the mask) is the evidence-grounded one;
- the current read-side adapter (residual on the query features before memory attention) cannot stop wrong masks from
  entering the bank.

## 6. Dense labels: which can stand in for the hand masks?

The hand masks are sparse, so dense labels were built to tighten CIs. A dense label is a valid evaluation proxy only if
it **ranks methods like the hand masks**.

| dense label | how | J vs hand (seq2–4) | P / R | ranks 9 methods like hand J? |
|---|---|---|---|---|
| **SAM2-ceiling** | SAM2.1-L, GT box at person-clear frames, propagated both ways | 0.716 | 0.88 / 0.78 | **yes**: Spearman +0.80, 7 of 8 paired Δ signs |
| frozen pseudo-GT | exemplar silhouettes ∪ SAM2-ceiling | 0.624 | 0.67 / 0.88 | **no**: Spearman −0.07, 3 of 8 |
| SAM2-b+ on frame t, GT box + centre (single image) | per frame | **0.730** | 0.83 / 0.87 | not tested yet |
| SAM2-b+ on a GT-motion integral image, t±1, weights 0.7^\|i−t\| | per frame | 0.690 | 0.77 / 0.87 | not tested (worse than frame t) |
| same, t±3 | per frame | 0.662 | 0.75 / 0.84 | — |

**Rules:**
- Use SAM2-ceiling for **ranking and selection only**. Its effect sizes favour SAM2-video outputs: adapter-pf − T1 is
  −0.063 on SAM2-ceiling vs +0.010 on the hand masks.
- The pseudo-GT measures extent, not correctness.
- Temporal integration makes a label more full-body: J falls monotonically with the window (t: 0.730 → t±1: 0.690 →
  t±3: 0.662).
- The single-frame GT-box SAM2 label is the best pixel match so far. Its rank validity is the next test: as an oracle
  that sees a GT box every frame, it may favour per-frame methods.

## 7. What counts as evidence

- **Cite:**
  - everything in §3–§5;
  - the label agreements in §6;
  - tracker vs blob tracker (box-centre error 10.0 vs 29.7 px vs GT boxes);
  - tracker box ≈ GT box as a SAM2 prompt (seq1 hand frames, n = 12);
  - synthetic results, in-domain only.
- **Withdrawn** (scored on the frozen pseudo-GT, which ranks methods wrongly):
  - adapter +0.040 J per-frame and +0.022 J in video;
  - "memory helps +0.082 J";
  - oracle-memory headroom +0.069 (circular: the label was both written into memory and used for scoring);
  - V̂ vs silhouette results;
  - loss-variant comparisons v3.2–v3.4.
- **Intuition only:**
  - SAM2-ceiling effect sizes;
  - seq1 hand frames (3 windows);
  - box metrics as segmentation evidence;
  - synthetic validation numbers (synthetic→real gap: +0.10 vs +0.01 at step 100).
- **Decisions to revisit,** both made partly on the pseudo-GT: the step-200 checkpoint and the recall-heavy loss
  (focal α 0.67).

## 8. Open questions and next steps (cheapest first)

1. **Re-select the adapter checkpoint and loss.** Score checkpoints it100–it600 per frame on seq1 using SAM2-ceiling J
   (dense), box AUC and recognition. Then compare simple vs layer-aware loss on the same footing.
2. **Rank validity of the single-frame GT-box SAM2 label** (§6), using the same 9-method test.
3. **Second annotator** on the hand frames: the ceiling for hand J, and whether the +0.010 tie with T1 is within
   annotator disagreement.
4. **Memory, decided on synthetic data with exact labels** (GPJATK val clips, exact visible-mask J). Arms:
   - (a) per-frame adapter;
   - (b) video + tracker prompts;
   - (c) video + prompts with the exact mask written into memory, the true upper bound;
   - (d) video + prompts with the adapter's own masks written into memory (write-path correction, zero training).

   If (c) ≤ (a), memory is dead. If (c) > (a), (d) shows how much a corrected write path captures, and video training
   targets the rest. NFO arm `vid-write-pf` = (d) on real data.
5. **Not pursued:**
   - Diffusion-VAS (amodal completion, a different task and non-causal).

## 9. Protocol changes of 2026-10-23 (fixed before the new runs)

- **Recognition at N = 5** frames per recognition stack (was N = 7, chosen a priori). N = 5 is where SAM2.1 video and SAMURAI
  score highest in the N sweep (`eval/recog_vs_N.py`), i.e. the baselines' best case. The sweep stays as a supplement.
- **Reporting over all sequences** (seq1–4) as the headline, with seq1 and seq2–4 alongside. Step 200 was selected on
  seq1, so seq1 is not independent; seq2–4 remains the clean held-out check. **Changed 2026-11-10: seq2–4 on whole crossings is the headline (§4, talk); seq1–4 is reported alongside.** Hand-label scores stay per group (seq1
  brush masks have no void class).
- **Integral-image history** (pre-registered rule): candidates 4, 5, 6 past frames (all inside the training range 4–8; no
  retraining; arms `adapter-<seed>-h4` / `-h5`, `eval/dump_arms.py`). Decided on **seq1 only**: take the shortest
  history that is not worse than 6 in both box AUC and recognition confidence (N = 5), each averaged over the 3 seeds;
  otherwise keep 6. All sequences are then reported with the chosen history.
  **Outcome (2026-10-23): keep 6.** seq1 box AUC, 3-seed mean: 6 frames 0.630, 5 frames 0.624 (Δ −0.005
  [−0.009, −0.002]), 4 frames 0.620 (Δ −0.009 [−0.015, −0.004]); every seed lower with a shorter history, so both fail
  the box-AUC half of the rule and recognition does not enter the decision. Not revisited on other data.
- **Full-pass protocol** (second protocol, next to the 40-frame windows): one clip per pass from its earliest benchmark
  start to the end of the pass (32 clips, 44–155 frames, every frame once; `trials --to-pass-end --first-per-pass`,
  NFO_RUN=fullpass). Motivation: following a person through a whole crossing is the realistic use, and memory drift
  (§`eval/mask_fill.py`: masks of the memory methods shrink ≈ 10% per 10 frames, ours ≈ 1%) is hidden by short clips.
  **Outcome (job 1950229, 2026-10-24, one cluster env; ours = 3 seeds, Δ vs SAMURAI with window-bootstrap 95% CI):**

  | seq1–4, 32 clips | ours | SAMURAI | SAM2.1 video | ours − SAMURAI, per seed |
  |---|---|---|---|---|
  | box AUC | 0.674 / 0.681 / 0.678 | 0.579 | 0.571 | +0.096 … +0.102, every CI > 0 |
  | hand J (all 43 hand frames, in 19 clips) | 0.757 / 0.759 / 0.752 | 0.730 | 0.733 | +0.022 … +0.029, every CI > 0 |
  | recognition conf (N = 5) | 0.662 / 0.669 / 0.671 | 0.663 | 0.666 | −0.001 … +0.008, on par |

  seq2–4 alone: box AUC +0.102 … +0.109, hand J +0.022 … +0.030 (23 frames in 18 clips), recognition on par. Against the
  40-frame windows the box lead grows (+0.065 → +0.10) and hand J turns from on par into a small, significant lead; the
  gain is recall (hand R 0.86 vs 0.81) at some precision (0.86 vs 0.88). Recognition stays on par: it scores 5-frame
  stacks aligned with the true motion, so it sees no clip-length effect by construction. Hand J uses all 43 hand
  frames (every one lies inside a crossing), averaged per clip first: n = 19 in the CIs is clips, not frames. seq1's
  20 brush frames fall in one clip, so seq1 weighs 1/19 of the all-sequence hand J. Pooled per frame instead (seq1 at
  20/43): SAMURAI 0.640, SAM2.1 video 0.623, trk 0.590, ours 0.655 / 0.665 / 0.660 (lower overall because seq1's brush
  masks have no void class); seq2–4 per frame matches per clip within 0.003. The per-clip mean is the headline because
  the bootstrap resamples clips.

  **Ablation on the 32 clips (job 1950449, 2026-10-24, same env; `-noC` = integral image removed at inference):**

  | seq1–4, 32 clips | box AUC | recog conf | hand J (43 frames, 19 clips) | hand P / R |
  |---|---|---|---|---|
  | SAMURAI | 0.579 | 0.663 | 0.730 | 0.879 / 0.806 |
  | SAM2.1 video (b0-c2) | 0.571 | 0.666 | 0.733 | 0.893 / 0.796 |
  | trk (tracker prompts) | 0.628 | 0.596 | 0.685 | 0.803 / 0.827 |
  | b0trk-c2 (SAM2.1 video + tracker prompts, memory on) | 0.661 | 0.636 | 0.745 | 0.821 / 0.883 |
  | adapter-noC, 3 seeds | 0.656 / 0.646 / 0.639 | 0.653 / 0.648 / 0.652 | 0.752 / 0.739 / 0.730 | 0.877 / 0.813 (mean) |
  | adapter, 3 seeds | 0.674 / 0.681 / 0.678 | 0.662 / 0.669 / 0.671 | 0.757 / 0.759 / 0.752 | 0.861 / 0.851 (mean) |

  Steps (3-seed means): position hint (trk − SAMURAI) box +0.049 [−0.013, +0.106], recog −0.067 [−0.101, −0.029],
  hand J −0.045 [−0.088, −0.010]; adapter without integral image (noC − trk) box +0.019, recog +0.055, hand J +0.055,
  mostly precision (0.80 → 0.88); integral image (adapter − noC) box +0.019 / +0.035 / +0.038, recog +0.009 / +0.022 /
  +0.019, hand J +0.005 / +0.020 / +0.022 per seed, every CI > 0 except seed 0 hand J [−0.003, +0.013]; the gain is
  recall (0.81 → 0.85). seq2–4 alone: box +0.018 / +0.032 / +0.037, recog +0.011 / +0.022 / +0.023, hand J +0.004 /
  +0.019 / +0.022. Larger than on the 40-frame windows (box +0.024 ± 0.005).

  **Memory + tracker prompts (`b0trk-c2`, job 1953256, 2026-10-25, same env; `baselines/run_sam2_video.py --trk-prompts`,
  commits ac7cff2 / b79a0df):** SAM2.1 video with our tracker's box + point on every frame after t0; whole-crossing
  version of the old vid-prompts row (§5). The other arms' numbers in that run match job 1950449 exactly.
  - What memory adds on top of the tracker (b0trk − trk): hand J **+0.060 [+0.014, +0.113]**, box +0.033
    [−0.004, +0.076], recog **+0.040 [+0.021, +0.061]**. On the crossings memory does help a tracker-prompted SAM2.1
    (windows: J +0.032, box +0.020).
  - What the tracker adds to memory (b0trk − b0): box **+0.089 [+0.062, +0.116]**, hand J +0.012 [−0.011, +0.036],
    recog −0.029 [−0.054, +0.001]: tracking up, precision down (0.89 → 0.82), recall up (0.80 → 0.88).
  - Ours (no memory) vs b0trk, 3-seed means: box 0.678 vs 0.661, hand J 0.756 vs 0.745, recog 0.667 vs 0.636, AP50
    0.878 vs 0.861, P@20 0.773 vs 0.766 (no paired CI computed for this pair). The memory-free adapter stays above the
    best memory arm on every metric; the margin on segmentation and tracking is small.
  - seq2–4 alone: b0trk − trk J +0.059 [+0.012, +0.119], box +0.014, recog +0.029; b0trk − b0 box +0.083
    [+0.059, +0.107], recog −0.043 [−0.066, −0.021].

## 10. One-pass Layer-SAM: a learned integral image from cached encoder features (2026-11-19, in progress)

Branch `exp/layer-sam-feat`.
- Design: `docs/superpowers/specs/2026-11-19-layer-sam-feature-integral-design.md`.
- Plan and gates: `docs/superpowers/plans/2026-11-19-layer-sam-feature-integral.md`.
- Code: `adapter/feat_integral.py`, `fusion_data.build(feat=True)`, `fusion_adapter.train_step` (env `FEAT=1`),
  `inference.masks_from_arrays` (`net.agg`).

**Idea.** Today's adapter encodes every frame twice: the frame, and the pixel integral image C. Instead, build Ĉ from the
encoder features of the last n_h frames, which are already computed:
- shift each past frame's feature grid by the tracker's Theil–Sen x-shift (bilinear);
- take a per-cell softmax over the shifted frames plus a **null key** (value: the empty scene B; a learned "nothing here"
  slot);
- predict the evidence mass a with a **separate** small head.

The module has 35,460 parameters (the Fusion adapter: 730,401). It is trained first alone (stage 0), then jointly with
Fusion, with two auxiliary losses:
- 1 − cos(Ĉ, E(X̃_t)) on hidden person cells that were visible in some aligned past frame. X̃_t is frame t without the
  occluder, with the same noise and sway.
- BCE(a, person visible in some aligned past frame).

Everything else (Fusion, loss v3.4, step selection on seq1, the tracker) is unchanged.

**What was measured** (synthetic val = 30 held-out clips; laptop RTX PRO 4000):

| test | result |
|---|---|
| Shift equivariance of SAM2 features under subject motion (30 clips) | Whole-cell shifts ≈ exact (s16 1 − cos 0.037 vs person/background contrast 0.92). Half-cell shifts not: 0.268 median, 0.516 p90 (bilinear). Missing information (sub-stride aliasing), so positional encoding cannot fix it. |
| cos(·, clean features) on hidden-recoverable person cells, s16 | **learned 0.500**, uniform average 0.446, pixel enc(C) 0.425, frame t 0.319. Learned − pixel **+0.075 [+0.052, +0.097]** |
| Evidence mass: person-vs-rest AUC; mean on / off person | learned 0.864; 0.392 / 0.126. Pixel Σ w G: 0.851; 0.432 / 0.101 |
| Alignment on real NFO (seq1, n = 12) vs synthetic (n = 232): cos(shifted F_{t−k}, F_t) on visible cells, ages 1–6 | NFO 0.765 / 0.515 / 0.455 / 0.378 / 0.348 / 0.324; synthetic 0.726 / 0.581 / 0.488 / 0.429 / 0.391 / 0.364. A modest gap and the same decay: alignment transfers. |
| Runtime (11 NFO windows, batch 1) | **38.4 ms/frame** (26 fps) vs pixel adapter 59.1 ms, `trk` floor 34.1 ms, SAM2 video 53 ms (2026-10-20). The feature integral costs 4.3 ms. |
| Training memory | 12.4 GB at 32 targets (smoke step), ≈ 14 GB at n_h = 8 (derived): needs a ≥ 16 GB GPU |

**Occluder model** (`tools/fragment_stats.py`). NFO hand labels lie between perlin 12+24 and the depth-stacked 'stack'
foliage:

| | visible components | component area | boundary·h / area |
|---|---|---|---|
| NFO | 4 | 0.015 h² | 42.8 |
| perlin 12+24 | 4 | 0.012 h² | 56.7 |
| stack | 2 | 0.074 h² | 34.2 |

- A 50/50 mix of the two brackets NFO; this is the planned occluder-mix arm.
- NFO walkers move ≈ 3× faster per frame, relative to their height, than the GPJATK walkers.

**Why it should work (argued, not shown).**
- Only the adapter reads Ĉ, so averaged features never reach the frozen decoder. That was the failure in the GPJATK
  feature-fusion test of 2026-10-06.
- On synthetic data, the cached-feature route is closer to the clean person than re-encoding the pixel composite.

**Next: G2.**
- One cluster training (seed 0, 1000 steps; launched 2026-11-19).
- Then seq1 whole crossings (`NFO_RUN=fullpass`, clips 0–7), scored on the laptop: `eval/g2_score.py`.
- Reference (measured, same protocol and environment):

  | arm | box AUC | hand J |
  |---|---|---|
  | pixel adapter seed 0 | 0.544 | 0.523 |
  | `trk` | 0.468 | 0.483 |

- PASS: the new arm ≥ pixel − 0.02 on both. If it passes: 3 seeds and the seq2–4 headline, then the occluder mix.
- If misalignment limits it: neighbourhood / deformable attention around the tracker-aligned position (design §4).
- **In-run check at step 200** (laptop, 12 seq1 selection frames).
  - The integral's contribution (adapter − noC) is +0.010; the gain over trk-box is +0.035. The pixel run at step 200
    had +0.020 / +0.046.
  - The learned evidence mass is weak on NFO (mean on the visible person 0.13 vs 0.42 synthetic; stage-0 module).
  - But swapping in the pixel mass changes the adapter by only +0.004 (`MASS=pixel`), so the mass is not the
    bottleneck.

**G2 (2026-11-19, seq1 full passes, 8 crossings / 20 hand frames, laptop env @lap): PASS.** Cluster job (FEAT=1, seed 0,
1000 steps, RTX 5000 fp16) died at the it-800 validation (allocator fragmentation: 8.5 GB allocated, 5.5 GB reserved-unused;
fixed in 61ef448 with expandable_segments + empty_cache). Selected it700 = best seq1 selection J (0.597; noC 0.590).

| arm | box AUC | hand J | ms/frame (dump) |
|---|---|---|---|
| adapter-feat (one encoder pass, it700) | 0.572 | 0.565 | 41–46 |
| adapter-feat-noC (integral removed) | 0.568 | 0.569 | 41–45 |
| adapter-pf (two passes, pixel integral, it200) | 0.544 | 0.523 | |
| trk (SAM2 + tracker prompts) | 0.468 | 0.483 | |

feat − pf: box AUC +0.028 [+0.015, +0.041], hand J +0.042 [+0.016, +0.072]. **But feat − noC: AUC +0.004 [−0.003, +0.012],
J −0.004 [−0.025, +0.014]:** on seq1 the learned feature integral adds nothing measurable; the gain over adapter-pf is
confounded by step (700 vs 200) and is carried by the Fusion residuals on F_t / F_B. Next: the integral's contribution is the
open question — pixel adapter trained to it700 (same step) and/or seq2–4 feat vs noC over 3 seeds before any claim.

**Synthetic val does not reward the integral (2026-11-19, `tools/syn_noC.py`, 30 val clips, adapter − noC J, clip bootstrap):**
feat it700 −0.002 [−0.005, +0.001], feat it200 −0.004 [−0.007, −0.000], pixel it600 −0.005 [−0.009, −0.002], pixel it200
+0.002 [−0.002, +0.006]. Neither integral helps on synthetic, though the pixel one helps on NFO seq1 (+0.018 box AUC): the
synthetic task is solvable from F_t / F_B, so training never pays for reading Ĉ. "Make the integral useful" is first a
training-data problem (frames hidden at t but visible in history), before per-stride weights. Next: adapter − noC in the
densest occluder group, and the frequency of hidden-at-t / visible-in-history frames in the synthetic mix.

**Where the integral acts on synthetic data (2026-11-19, `tools/syn_noC.py` grouped, 30 val clips, frames with history,
964 frames; rows in `results/benchmark/rr/feat_integral/syn_noC_rows_*.json`).** Per frame: vis = |V_t|/|S_t|, rec = share
of the person hidden at t but visible in the aligned history (Ω). Median vis 0.64, rec 0.22; HARD frames (vis < 0.5 and
≥ half the hidden part seen in history) are 12.1% (117 frames): not rare.

| adapter − noC (J) | feat it700 | pixel it600 |
|---|---|---|
| all | −0.002 [−0.006, +0.001] | −0.006 [−0.009, −0.002] |
| HARD | −0.012 [−0.018, −0.006] | −0.020 [−0.038, −0.001] |
| rec ≥ 0.3 | −0.009 [−0.016, −0.003] | −0.015 [−0.022, −0.009] |
| ρ 0.4–0.6 | −0.005 [−0.011, +0.001] | −0.008 [−0.014, −0.001] |
| HARD: ΔP / ΔR | −0.018 / +0.009 | −0.039 / +0.030 |

The adapter does read the integral: it raises recall and lowers precision more, and most where Ω is large. The integral
carries the *amodal* person (hidden parts seen in history; Airborne Optical Sectioning's purpose), the target is the
*visible* mask, and loss v3.4 gives occluded pixels within 8 px of a fragment weight 0, so spill into Ω is free in training
and an FP in J. The stage-0 L_C target (clean features on hidden cells Ω) is amodal as well. So enriching HARD frames would
enrich exactly the frames where the integral hurts J; it is not the fix. Consistent with NFO: pf − noC is +0.018 on box AUC
(GT boxes are amodal) and +0.016 [−0.001, +0.040] on hand J (not significant).

**Downstream surrogate and the stride oracle (2026-11-19).** NFO has no trustworthy person segmentations, so downstream
metrics decide. Synthetic stand-in for box AUC: IoU of the mask's bbox with the amodal person box (bbox of S_t;
`fusion_adapter._box_iou`). adapter − noC box IoU: pixel it600 **+0.018 [+0.006, +0.031]** (ρ 0.4–0.6: +0.026), feat it700
+0.008 [+0.000, +0.018]: the same ordering as seq1 box AUC (+0.018 / +0.004). For the current model the spill that costs J
buys the box, but this is not a necessary trade: a perfect visible-only mask V_t has box IoU 0.944 vs pixel adapter 0.781 /
noC 0.763 (mode-0 frames), so penalising spill into Ω (W = 1 there) is a legitimate candidate, tested by box IoU. Oracle (`ORACLE=1`, feat it700, no retraining), clean-frame features T in place of Ĉ:
- s8 + s4 only (`oracle-fine`) − adapter: J +0.001, box IoU +0.002 [−0.000, +0.005] (HARD −0.003). Below the
  pre-registered +0.01: per-stride mixing weights cannot matter for this adapter. Not pursued.
- all strides (`oracle-all`) − adapter: box IoU **+0.015 [+0.008, +0.024]**, HARD **+0.054 [+0.016, +0.102]**, J +0.000.
The headroom is in the s16 content of Ĉ (cos to clean 0.50 on hidden cells), not in its fine-stride resolution.

**Held-out seq2–4, feature adapter it700 (seed 0), 2026-11-20, cluster env (jobs 2036262 + dump array; `pass_run.sh heldout`;
24 whole crossings, 23 polygon hand frames; copy in `results/benchmark/fullpass_cluster/`).** The scoring reproduces the
headline (adapter-pf/s1/s2 box AUC 0.717 / 0.724 / 0.718).

| arm | box AUC | hand J |
|---|---|---|
| adapter-pf / s1 / s2 (pixel integral, two passes) | 0.717 / 0.724 / 0.718 | 0.767 / 0.769 / 0.762 |
| b0trk (SAM2 video memory + tracker box/point every frame) | 0.694 | 0.752 |
| **adapter-feat it700 (one pass)** | **0.679** | **0.743** |
| trk (SAM2 image + tracker prompts) | 0.680 | 0.698 |
| SAMURAI (t1) / SAM2 video (b0) | 0.615 / 0.611 | 0.739 / 0.743 |

Paired box AUC: feat − pf **−0.038 [−0.047, −0.030]**; feat − trk −0.001 [−0.015, +0.014]; feat − b0trk −0.015
[−0.037, +0.013]; pf − b0trk **+0.023 [+0.003, +0.048]**. The seq1 G2 advantage of feat over pf (+0.028) reverses on the
held-out sequences: on the box, the feature adapter equals its own tracker prompts, i.e. it carries none of the pixel
integral's held-out box gain (+0.018 to +0.037 per seed vs `-noC`). Consistent with feat ≈ noC. The bar for any Ĉ fix is
now explicit: recover ≈ +0.04 box AUC. The pixel adapter beats SAM2 memory + tracker prompts on box AUC.

### 10.x Diagnosis of the one-pass adapter: where the integral acts and why Ĉ is weaker (2026-11-24, laptop)

Question: where does the feature adapter (feat it700) fall short of the pixel adapter, and is its integral useful at all?
All synthetic numbers: 30 val clips, 964 frames with history, paired clip bootstrap (`tools/syn_noC.py`, rows in
`results/benchmark/rr/feat_integral/syn_noC_rows_*.json`). Pixel synthetic reference = token_seed0 it600.

**1. The held-out gap is not reproduced on any selection set.**
- Synthetic, feat − pixel (paired, same clips): box IoU +0.003 [−0.007, +0.014]; J −0.016 [−0.020, −0.013], all of it
  precision (P −0.022, R +0.004). Same −0.020 J with the integral removed from both, and on frames without history:
  the difference is in the trained Fusion, not the integral.
- seq1 (8 crossings, box AUC): feat − pf **+0.028** [+0.015, +0.040]; the cluster (0.574) and the laptop (0.572) agree, so
  environment drift is excluded. Decomposition: feat − feat-noC +0.004 [−0.003, +0.012] vs pf − pf-noC +0.018
  [+0.013, +0.022]; feat-noC − pf-noC +0.042 [+0.024, +0.058]. On real data the feature integral is idle as well; the
  seq1 advantage is the feat Fusion's.
- seq1 is failure-dominated (fail share feat 0.047, pf 0.100, trk 0.116); seq2–4 has fail = 0 for every arm, and feat
  trails pf on 22 of 24 crossings. seq1 measures not losing the person, seq2–4 box extent. On seq1 frames where every arm
  tracks, feat still leads (box IoU +0.019 [+0.000, +0.038], `tools/box_bias.py`), so seq1 does not reproduce the extent
  deficit either. Open confound: training length (feat it700 vs pf it200), only testable on seq2–4.

**2. FP split (where false positives land).** Share of FP pixels inside the hidden person S ∧ ¬V: feat 0.695, feat-noC
0.686, pixel 0.722, pixel-noC 0.710. About 70% of every adapter's FPs are amodal fill (loss v3.4 does not penalise it);
the integral adds ≈ 0.01 of that share. Feat's precision deficit is background spill.

**3. Which channel carries the integral: appearance C, not evidence mass a.** Inference ablation (`ABL=1`), each − noC:

| | feat J / box IoU | pixel J / box IoU |
|---|---|---|
| only C (a := 0) | **+0.004 / +0.008** | **+0.006 / +0.016** |
| only a (C := B) | −0.005 / +0.004 | −0.012 / +0.002 |
| both (as trained) | −0.002 / +0.008 | −0.006 / +0.018 |

`a` raises recall and spills onto the background (FP share in the hidden person drops); C raises J, precision and box
together and adds person-shaped fill. The occupancy-only redesign proposed the same morning is refuted.
Downstream check (`-noA` arm, first crossing s0 of each of seq1–4, laptop; screening rule below): a := 0 costs box AUC
−0.006 (feat, 3 of 4 clips) / −0.002 (pixel) and recognition conf −0.005 to −0.017. Rejected; `a` stays.
Runtime on the same run: **feat 42 ms/frame, pixel 66 ms/frame** (RTX PRO 4000 laptop).

**4. Neighbourhood keys (test 1, NBR = 1; aggregator-only stage 0, 200 steps from it700, Fusion frozen).** s16 cos to
clean on Ω 0.503 → 0.528 (s8 0.503 → 0.519), val L_C + L_a 0.499 → 0.477; adapter − noC box IoU +0.007 vs +0.008
(control), J −0.010 vs −0.002. Fails its bar (≥ +0.015 and > control). Read as a diagnostic: better Ĉ direction alone
does not reach the output. Not tested: a Fusion retrained on neighbourhood Ĉ.

**5. Why Ĉ helps less although closer to clean: shrinkage towards B.** `tools/feat_stats.py`, s16, synthetic val:

| source | ‖C − B‖ / ‖T − B‖ on Ω (visible) | cos(C − B, T − B) on Ω (visible) |
|---|---|---|
| learned Ĉ | **0.74 (0.67)** | **0.72 (0.73)** |
| enc(pixel integral) | 0.92 (0.93) | 0.65 (0.66) |
| F_t | 0.83 | 0.56 |

Ĉ has the best direction of the change signal but about ¾ of its magnitude, below the frame itself; consistent with
the softmax pulling towards the null key ∅ = B where it is uncertain. Mechanism test (inference rescaling
C′ = B + s (C − B), s ∈ {1.35, 1.7}, `SCALE=`, all strides, feat it700, Fusion as trained), − noC:
J −0.002 (s = 1) → −0.004 → −0.007, box IoU +0.008 → +0.005 → +0.005 (HARD box +0.011 → +0.007 → +0.014, CIs span 0).
Amplifying hurts, monotonically in s: no inference-time fix. The trained Fusion is calibrated to the shrunk scale, so
this does not separate "magnitude is irrelevant" from "Fusion adapted to it". Neighbourhood keys (test 1) also raise
the magnitude (Ω 0.76 → 0.81, visible 0.71 → 0.78; direction 0.724 → 0.735; nbr0 = same 200 extra stage-0 steps) without
output gain. Together: neither a better direction nor a stronger Ĉ reaches a frozen Fusion; the open lever is
how the Fusion is trained on Ĉ.

**6. Probe: does Ĉ carry the information, independent of any adapter?** The inference perturbations above (rescaling,
a := 0, swapping in the nbr aggregator under a frozen Fusion) measure the trained Fusion's calibration, not the content of
its inputs; so do the `-noC` controls used throughout (reliance, not usefulness). `tools/feat_probe.py` (linear probe,
Alain & Bengio arXiv 1610.01644): frozen s16 features, a fresh class-balanced logistic regression per input set, trained on
60 synthetic TRAIN clips (355k cells), tested on the 30 val clips (279k cells); cells in the amodal person box + 50%;
person (S share ≥ 0.5) vs background (share 0). Hidden = person cells with visible share < 0.1. Cells cached
(`results/benchmark/rr/feat_integral/probe/cells_n60.pt`, 4.5 GB); MLP reader (1 × 256) on the same cells.

| input (+ F_t, B) | hidden-vs-background AUC, linear (MLP) | gain over no history |
|---|---|---|
| no history | 0.966 (0.964) | |
| learned Ĉ it700 | 0.973 (0.974) | +0.007 [+0.002, +0.012] (+0.010) |
| enc(pixel integral) | 0.974 (0.976) | +0.009 [+0.002, +0.015] (+0.011) |
| oracle T (clean frame) | 1.000 (1.000) | +0.034 (+0.036) |

Decisive comparison, paired per clip, learned Ĉ − pixel on hidden cells: linear **−0.0015 [−0.0040, +0.0008]**, MLP
−0.0014 [−0.0036, +0.0006]. Ĉ carries about the same hidden-person information as the pixel integral (both close ≈ 25–30%
of the gap to the oracle). nbr1 Ĉ = learned Ĉ (+0.007 / +0.010): neighbourhood keys add no information. Conclusion: the
feature adapter's idle integral is a training / usage problem of the Fusion, not missing content. Blind spot of the probe:
AUC is near saturation (the no-history floor is 0.966, s16 context sees the adjacent visible person), so it compresses
differences; it measures content on synthetic data, not transfer to NFO.

**7. Where Ĉ's information stops (zero-training checks for the retraining proposal).**
- Through the Fusion's shared s16 projection P (256 → 64, applied to X, C and B; `fusion_adapter.py:73`),
  `tools/feat_probe_P.py`, same cached cells: hidden-vs-background gain with the feat adapter's own P: Ĉ +0.015 [+0.008,
  +0.023], pixel +0.018 [+0.009, +0.027]; learned − pixel −0.003 [−0.006, +0.001]. The bottleneck keeps Ĉ's information;
  it reaches the Fusion's first block.
- Reliance over training, from the runs' own logs (seq1 brush J, adapter − noC, 20 frames; weak labels, but paired within
  each checkpoint): feat it100…700 +0.000, +0.010, +0.008, −0.011, +0.009, +0.006, +0.007; pixel it100…600 +0.037, +0.020,
  +0.017, +0.023, −0.013, +0.022. Recall gain from C: feat +0.007…+0.03, pixel +0.04…+0.11. The feat Fusion never learned
  to use Ĉ, from the first 100 steps on; no rise-then-decay, so the training-length confound does not explain idleness.
Remaining difference between the two recipes: the aggregator is trained jointly (mask-loss gradients + λ(L_C + mass BCE))
while the Fusion learns, versus a fixed pixel integral.

**Screening rule (fixed 2026-11-24).** *Superseded 2026-11-25: seq2–4 s0 are held-out test crossings; screen on seq1 crossings only.* Inference switches within one checkpoint are screened on the first crossing (s0)
of each of seq1–4 with box AUC + recognition (paired per-clip SD of such switches ≈ 0.007 box AUC, so 4 clips resolve
≈ ±0.01). Promising switches then run on all crossings, reported with and without the 4 screening clips. Comparisons
between different checkpoints are never screened on 4 clips (per-clip SD ≈ 0.03). Hand J is no longer a decision metric.

8. **Synthetic reliance does not separate the two adapters; the deficit is NFO-specific (2026-11-24).** Reference
   for the frozen-aggregator retrain: the *deployed* pixel adapter (token_seed0 it200) on the 30 synthetic val clips,
   adapter − noC: box IoU +0.008 [+0.002, +0.014], J +0.002 [−0.002, +0.007] (964 history frames); HARD frames box IoU
   +0.015 [−0.001, +0.037]. The feat adapter (it700) has the same +0.008. The +0.018 used earlier as "pixel reliance"
   was it600. So on synthetic data both adapters use their history equally; only on NFO seq1 does the pixel adapter use
   it more (box AUC reliance +0.018 vs +0.004). The defect is a synthetic → NFO transfer of Ĉ or of its use, not
   in-domain usage. Consequence: synthetic reliance cannot read out the frozen-aggregator retrain (its pre-set bar
   +0.015 was taken from the wrong checkpoint). Also seen in the training logs: joint training cut the aggregator's
   evidence mass Σa on synthetic from 8614 (stage0c, step 0) to ≈ 1000 by step 100 (pixel integral 6577); the seq1
   mass in the feat log.json (6271) is the pixel mass, a print bug fixed in 157a1b5 (the adapter itself read Ĉ).
   Files: results/benchmark/rr/feat_integral/frozen/run.log.

9. **Frozen-aggregator retrain: freezing does not make the adapter use Ĉ (2026-11-24).** Fusion from scratch, it700
   aggregator frozen (`AGG_FREEZE=1`, LAM = 0), pixel recipe (1000-step cosine, seed 0), stopped at step 200
   (`feat_frozen_seed0/`). adapter − noC: synthetic box IoU +0.005 [+0.001, +0.009], J −0.007 [−0.011, −0.003]
   (feat it700 +0.008 / pixel it200 +0.008 box IoU); seq1 hand J +0.004 at step 200 (pixel it200 +0.020). Joint
   training is not why the feat adapter ignores Ĉ on NFO. One seed.
10. **NFO probe: on real footage Ĉ carries the same person information as the pixel integral (2026-11-24).**
   `tools/feat_probe_nfo.py`: inputs captured from the deployed inference path (`masks_from_arrays(probe=)`) on the 8
   seq1 full-pass crossings, history frames with a GT box (320k s16 cells); label person = GT-box core (central 50% w,
   80% h) vs background outside the box; 'occl' = the third of person cells with the lowest current-frame change
   1 − cos(F_t, B) vs background. Leave-one-crossing-out, crossing bootstrap. Linear reader, AUC gain over [F_t, B]
   (0.989 all / 0.970 occl): pixel +0.006 / +0.015, it700 Ĉ +0.006 / +0.015, stage0c Ĉ +0.006 / +0.015;
   it700 − pixel +0.0001 [−0.003, +0.005] all, +0.0004 [−0.008, +0.012] occl. it700 − stage0c = 0.0000: joint
   training left Ĉ's content unchanged. Ĉ's gain is steadier across crossings (pixel is −0.028 on s1, Ĉ +0.011). MLP
   reader saturates (none 0.997), gains ≤ 0.003, same ordering. Caveat: box-core labels, and each probe is standardised
   on NFO itself, so it is blind to a synthetic → NFO shift in Ĉ's statistics (the remaining candidate). Visual trace:
   `results/benchmark/rr/feat_integral/probe_nfo/labels_*.jpg`.

11. **Transfer probe: Ĉ's information is there on NFO, but a reader trained on synthetic data cannot read it as well as
   the pixel integral's (2026-11-24).** `tools/feat_probe_transfer.py`: linear reader fitted and standardised on the 60
   synthetic train clips (feat_probe cache), scored on the 8 seq1 crossings (feat_probe_nfo cache), the adapter's own
   situation without the adapter. Transfer AUC all / occl: none 0.920 / 0.820, pixel 0.958 / 0.907, Ĉ (it700)
   0.938 / 0.863. Gain over none: pixel +0.039 [+0.025, +0.053], Ĉ +0.018 [+0.013, +0.023]; **Ĉ − pixel −0.020
   [−0.032, −0.009]** all, −0.044 [−0.067, −0.019] occl (6 of 8 crossings negative, 2 ≈ 0). Fitted on NFO itself the
   two are equal (item 10). So the synthetic → NFO shift is larger for Ĉ than for enc(C), consistent with the adapters'
   reliance on NFO (pixel +0.018 vs feat +0.004 box AUC) and their equal reliance on synthetic data (+0.008 each).
   Diagnosis: the feat line's NFO deficit is a domain shift in how Ĉ encodes the person, not missing information and not
   joint training. Caveat: the label definitions differ between domains (mask vs box core), but equally for all sets.

12. **The Ĉ transfer gap is not a mean / scale shift (2026-11-24).** Same transfer probe with every set standardised
   per channel within each clip / crossing (`NORM=clip`, AdaBN-style; Li et al. ICLR-W 2017, arXiv 1603.04779):
   Ĉ − pixel −0.021 [−0.035, −0.010] all, −0.047 [−0.072, −0.021] occl (unnormalised −0.020 / −0.044). Pixel gain
   +0.040, Ĉ +0.018, unchanged. A per-frame normalisation of Ĉ inside the adapter would therefore not fix the NFO
   deficit: the shift is in which feature directions carry the person, not in their first two moments.

13. **Distillation pilot: teaching the aggregator's α to match the pixel integral on unlabelled NFO closes ≈ 25% of the
   transfer gap (2026-11-24).** `tools/feat_distil.py`: the it700 aggregator fine-tuned (400 steps, AdamW 1e-4, batch 8)
   so that its s16 output matches enc(C) (centred cosine) on tracker-centred 40×40-cell crops of seq1 crossings s4–s7
   (386 frames, no GT). Teacher cosine: train 0.535 → 0.578, held-out s0–s3 0.596 → 0.633 (the fit generalises).
   Transfer gain over [F_t, B] on held-out s0–s3 (reader fitted on 30 synthetic clips), all / occl: pixel +0.028 /
   +0.071, it700 Ĉ +0.013 / +0.036, **distilled Ĉ +0.017 / +0.043** (≥ it700 on all 4 crossings: +0.003 +0.001 +0.002
   +0.010 all). Gap closed: 27% (all), 21% (occl). Since only α was trained, Ĉ stays a convex combination of keys and B:
   the remaining gap may be the limit of mixing alone. Files: results/benchmark/rr/feat_integral/distil/.

14. **A fixed (pixel-rule) α in feature space transfers worse than the learned one (2026-11-24).**
   `tools/feat_fixed_transfer.py`, same held-out readout as item 13: transfer gain all / occl fixed +0.004 / +0.013 vs
   learned +0.013 / +0.036, pixel +0.028 / +0.071. Re-weighting the keys cannot close the gap; the limit is that Ĉ is a
   convex mix of per-frame encodings. Mechanism (argued): the pixel route averages BEFORE the nonlinear encoder E, so the
   occluders are thinned and E encodes a de-occluded person the way it encodes any person; the feature route mixes
   E(X_k), each of which already entangles the person fragment with that frame's foliage, and E(Σ w X) ≠ Σ w E(X). The
   entanglement depends on the foliage's appearance, which differs between procedural and real foliage.
15. **Distillation with a refiner closes the transfer gap (2026-11-24).** `tools/feat_refine.py`: Ĉ' = Ĉ + g([Ĉ − μ,
   B − μ]) at s16 (g: 1×1 512→128, 3×3, 1×1 →256, zero-init; history and empty scene only, no F_t), α and g trained
   (1500 steps, lr 1e-4 / 3e-4) to match enc(C)16 in centred cosine on seq1 s4–s7 (unlabelled) alternating with 20
   synthetic clips (seed 11). Held-out teacher cosine 0.596 → 0.785 (still rising at 1500). Transfer gain on held-out
   s0–s3, all / occl: **refined +0.028 / +0.066**, pixel +0.028 / +0.071, it700 Ĉ +0.013 / +0.036: 96% / 85% of the gap
   closed; per crossing refined vs pixel +0.020/+0.027, +0.022/+0.016, +0.031/+0.009, +0.038/+0.061. Caveats: held-out
   crossings are from the same sequence (scene) as the distillation crossings; probe proxy, not the adapter.
   Next: retrain the Fusion on the refined aggregator, screen on NFO.

16. **Adapter retrained on the refined aggregator: half of the NFO gap closes, recognition does not move (2026-11-24).**
   Fusion from scratch on the frozen distilled + refined aggregator (`REFINED=`, `feat_refined_seed0/`, pixel recipe,
   step 200, seed 0); screen on the four s0 crossings, laptop, paired per crossing (seq1 s0 / seq2 / seq3 / seq4).
   Box AUC: featref 0.616 / 0.703 / 0.630 / 0.719 (mean 0.667), old feat 0.646 / 0.677 / 0.607 / 0.693 (0.656),
   pixel 0.607 / 0.722 / 0.644 / 0.738 (0.678). Reliance (adapter − noC, the screening quantity): featref +0.015 /
   +0.004 / +0.001 / +0.004 (mean +0.006, positive on 4/4), old feat −0.004 (1/4 positive), pixel +0.014 (4/4).
   seq2–4 s0 box AUC gap to pixel −0.042 → −0.017. Recognition (N = 5): seq2–4 s0 conf featref 0.646, feat 0.647,
   pixel 0.656; featref reliance −0.001 vs pixel +0.023; seq1 s0 featref conf 0.589 vs feat 0.633. Only s16 is refined
   (s8 / s4 keep the plain mix) and the evidence mass a still differs between domains (on-person a syn 0.56 / seq1 0.33).
   Four crossings, one seed: indicative only. Files: results/benchmark/rr/feat_integral/refined/run.log.

17. **Full-input distillation (all strides + evidence mass) adds nothing measurable over s16-only (2026-11-24).**
   `tools/feat_refine_full.py` (refiners g / g8 / g4, mass head BCE to the pixel mass; same data and budget as item 15).
   Held-out (s0–s3, every 3rd frame) teacher cosine s16 / s8 / s4 0.597 / 0.649 / 0.713 → 0.783 / 0.735 / 0.783 (train
   0.778 / 0.717 / 0.776), mass L1 0.104 → 0.054; the evidence mass is now domain-matched (Σa median syn 8637 / seq1
   8737, on-person 0.44 / 0.39; s16-only: 889 / 406, 0.56 / 0.33). Fusion retrained on it (`feat_refinedfull_seed0/`),
   s0 screen, box AUC seq1 / seq2 / seq3 / seq4: 0.605 / 0.693 / 0.628 / 0.728 (mean 0.663; s16-only 0.667, old feat
   0.656, pixel 0.678); reliance +0.005 / −0.012 / +0.010 / +0.011 (mean +0.004; s16-only +0.006, pixel +0.014).
   Recognition conf reliance: seq1 s0 +0.014, seq2–4 s0 −0.013 (pixel +0.024 / +0.023). Summary of items 15–17:
   distillation lifts the one-pass adapter's box AUC by ≈ +0.01 and makes its use of the history positive, ≈ half of
   the pixel adapter's; recognition does not follow. Residual: the student matches the teacher at cos ≈ 0.95 on
   synthetic (training loss) but ≈ 0.78 on NFO, i.e. the distilled input is itself still domain-shifted.
   Four crossings, one seed per arm: differences of ±0.01 are within noise.

18. **Student-capacity test: is the distillation gap data, student or information?** (`tools/feat_capacity.py`,
   `results/benchmark/rr/feat_integral/capacity.log`). The distilled students of items 15/17 fit the pixel teacher
   enc(C)16 equally on their NFO training crossings and held-out ones (0.779 / 0.785): underfitting, not data-limited.
   Test: a 5.9 M-parameter conv student sees all one-pass inputs unmixed (6 shifted keys K_k, validity, sub-cell phase,
   F_t, B; 2066 ch → 1×1 512 → 3×3 512 → 3×3 512 → 1×1 256), NFO only (seq1 s4–s7, 386 frames), 3000 steps, AdamW 3e-4
   cosine, seed 0; centred cosine to enc(C)16 in 40×40-cell tracker crops. Reference cos (held-out): B 0.474, F_t 0.515,
   plain mean of K 0.609, current refined student 0.785. Result: train 0.918, held-out **0.849** (plateau from step 2000).
   Reading: most of enc(C) is recoverable from per-frame cached features (the "encoder nonlinearity loses it" account is
   wrong at this level); the main limit of the current student is that it sees only the α-mixed map; a 0.07
   train/held-out gap says more unlabelled real footage would add some. Residual: train fit saturates at 0.92, not 1
   (capacity or information, not separated). One seed; teacher cos is a proxy, not a downstream metric. Next: put this
   student (trained on NFO + synthetic) in the feat adapter, retrain Fusion, s0 screen.

19. **Student size vs real-time cost** (`tools/feat_capacity.py big|small c w`, `capacity_small.log`; same data and
   recipe as item 18; timing: full 64×64 grid, batch 1, bf16 autocast, RTX PRO 4000 laptop, 100 reps). Held-out NFO
   teacher cos / ms per frame: big (5.9 M) **0.849 / 0.83 ms**; shared-1×1 c64 w128 (0.8 M) 0.805 / 0.30 ms; c32 w64
   (0.2 M) 0.767 / 0.15 ms (current refined student 0.785). Shrinking costs accuracy roughly in proportion; the big
   student is already < 1 ms (the FLOP estimate overstated its cost), so it fits the real-time budget as is; a tracker
   crop would cut it further. Choice: the big student.

20. **Which teacher? Clean-frame similarity on synthetic** (`tools/teacher_oracle.py`, `teacher_oracle.json`; 30 synthetic
   val clips, centred cos to T16 = SAM2 features of the occluder-free frame; clip bootstrap). Person cells / hidden cells:
   B 0.075 / 0.083, F_t 0.580 / 0.249, feat-α (Ĉ, stage-0c weights) **0.483 / 0.385**, pixel-hand (deployed C) 0.420 / 0.303,
   pixel-α (same weights mixed in pixels) 0.388 / 0.297, pixel-orc (GT-visibility gate) 0.516 / **0.387**.
   Oracle gate − hand gate +0.096 [+0.068, +0.124] / +0.085 [+0.055, +0.117]; pixel-hand − feat-α −0.063 / −0.082;
   pixel-α − feat-α −0.095 / −0.088. Reading: on synthetic, by this metric, feature mixing is *closer* to the clean
   frame than any non-oracle pixel image, and the "E(C) leaves the convex hull" mechanism gives pixels no advantage
   here. Confound: feat-α's weights were trained on this target and domain; the pixel arms were not. Together with item
   21, cos-to-clean-features does not predict NFO reliance, so the oracle headroom is in a currency that does not
   transfer; no learned pixel gate is built on it.

21. **Big student in the feat adapter: s0 screen** (`tools/feat_bigstudent.py` → `distil/bigs16_s4-7.pt`; Fusion
   retrained 200 steps, AGG_FREEZE, seed 0 → `feat_big_seed0`; arm `adapter-featbig`; `bigstudent/run.log`). Teacher
   fit (centred cos to enc(C)16) at 3000 steps: NFO train 0.889, held-out **0.844** (refined student 0.785), synthetic
   (training clips) 0.865. s0 crossings (seq1 s0 + seq2–4 s0, @lap), box AUC / reliance (arm − noC) / recognition conf
   reliance: featbig 0.662 / **+0.000** / −0.011; featref 0.667 / +0.006 / −0.001; feat 0.656 / −0.004 / +0.000;
   pf (pixel) 0.678 / +0.014 / +0.024 (pf reliance positive on all 4 crossings in both metrics). Reading: raising the
   student's fit to the pixel teacher (0.785 → 0.844 held-out) did not raise reliance; teacher cos does not predict
   whether the adapter uses C. One seed, 4 crossings (screen).

22. **Per-stage runtime** (`tools/profile_stages.py`, `profile_stages.log`; 11 seq1 windows, RTX PRO 4000 laptop, syncs
   per stage). feat 38.8 ms/frame: encoder 25.3, tracker (CPU) 4.0, key shift 0.7, aggregator 1.1, adapter 3.0, decoder
   2.5, rest 3.9. pixel 62.3 ms: encoder 46.8 (two passes), tracker 3.8, adapter 3.6, decoder 2.5, rest 6.5. 30 FPS
   needs ≤ 33 ms: encoder compile (`compile_image_encoder: False` in sam2.1_hiera_b+.yaml) and a GPU/grid tracker are the
   untested levers.

23. **Through the frozen pixel reader** (advisor route 1; `tools/feat_through_reader.py`, `feat_integral/bigpf/`).
   Premise: every feature-route Fusion was retrained on synthetic, where neither integral is rewarded, so its reading of C
   is arbitrary; the pixel Fusion f_pf (token_seed0 it200) is the one reader whose use of C transfers. Step 0 (no
   training; arm `adapter-featbigpf` = item-21 big student + aggregator under f_pf; s0 crossings, @lap): box AUC 0.668,
   reliance +0.010 / +0.000 / +0.002 / +0.003 (mean **+0.004**; featbig with its retrained reader +0.000), recognition conf
   reliance mean **+0.011** (featbig −0.011); pf +0.014 / +0.024. The reader alone moves recognition reliance halfway to
   pixel; box reliance stays below the +0.010 bar. Step 1 (arm `adapter-featbigpft`: aggregator trained 1000 steps through f_pf on seq1 s4–s6, 170 frames; loss = decoder-logit KL + per-stride residual MSE + token + 0.2·anchor to enc(C)/a_pix, dataset-mean normalised): held-out seq1 s7 reader mismatch r16 0.030 → 0.013, token 0.76 → 0.001 (the token, driven by the mass a, was the dominant step-0 mismatch). s0 screen: box AUC **0.675** (pf 0.678; featbig 0.662), reliance +0.012 / +0.004 / +0.017 / +0.009 (mean **+0.010**, pf +0.014), recognition reliance +0.008 / +0.008 / +0.022 / +0.006 (mean +0.011, positive on all 4; pf +0.024). The -noC inputs are identical for f_pf-based arms (C := B, a := 0), so this is arm-vs-pf directly. Pre-set bar (≥ +0.010 on ≥ 2 of 3 seq2–4 crossings): mean met, count 1 of 3: borderline. Closest one-pass result so far; one seed, 4 crossings. Caveat: `box_metrics score` rewrites box_gt.csv with only the arms it is given (earlier screen rows must be rescored from the saved masks). **Held-out, all 24 seq2–4 crossings** (`bigpf/heldout.sh`, @lap; noC = pf-noC, identical f_pf inputs; crossing bootstrap, 5000): box AUC featbigpft **0.710**, pf 0.717 (= the seed-0 headline, so the environment matches), noC 0.699. Reliance featbigpft **+0.011 [+0.009, +0.014]** (positive on 23/24) vs pf +0.018 [+0.015, +0.021]; featbigpft − pf **−0.007 [−0.009, −0.005]** (the earlier one-pass adapter was −0.038, commit d2fe44b). Recognition conf (24 crossings, `recog_N5_uniform_meanfill_heldout_bigpft2.csv`): reliance +0.006 [−0.001, +0.012] (14/24) vs pf +0.014 [+0.006, +0.021]; featbigpft − pf −0.008 [−0.016, −0.000]. Reading: the one-pass route now uses the integral on held-out real footage and recovers ≈ 60% of the pixel route's box gain at ≈ 40 vs 62 ms/frame; recognition gain not yet resolved from 0. One seed (CIs over crossings, not seeds). **v2** (`bigpf/v2.sh`: pixel-gate mass a fed to the reader instead of the student's a + seq1 s1–s6 every 3rd frame, 187 frames; same init and recipe): step-0 token mismatch 0.76 → 0.27 from pixel a alone, final s7 mismatch ≈ v1. Held-out 24 crossings: box AUC 0.710 (= v1), v2 − v1 −0.000 [−0.001, +0.001]; reliance +0.011 (= v1); recognition reliance +0.006 [+0.001, +0.011] (v1 +0.006); v2 − pf box −0.007. No effect; pixel a costs the per-frame gate (dump timing ≈ 47 vs ≈ 43 ms/frame, not profiled). Keep v1. **Screen leak, recount (2026-11-25, cold review):** the s0 screen used seq2/3/4 s0, i.e. 3 of the 24 held-out crossings, for design choices. On the 21 untouched crossings: box AUC featbigpft 0.712, pf 0.720, noC 0.701; reliance +0.012 [+0.009, +0.014] vs pf +0.019 [+0.015, +0.023]; featbigpft − pf −0.007 [−0.009, −0.005]; recognition reliance +0.005 [−0.003, +0.012] vs pf +0.012 [+0.003, +0.021]; v2 − v1 box −0.000. Conclusions unchanged. From now on screens use seq1 crossings only.

24. **Training-only amodal aux head: does a loss that needs history make the reader use Ĉ? (2026-11-25, running)**
   Hypothesis (cold review): loss v3.4 targets the visible mask V with hidden pixels weighted 0 / 0.3, so no term needs history and any
   reader trained on synthetic ignores the integral (items 8, 9, 16, 21). Change, one variable vs `feat_big_seed0` (FEAT, AGG_FREEZE,
   BigS16, Fusion from scratch, 200 of 1000 cosine steps, seed 0): `AMODAL=1` adds `AmHead` (3×3 + 1×1 conv on the Fusion trunk
   x4 + ↑x16; it never trains the residual outputs out[l], so amodal fill is not pushed into the modal mask), loss_fn on the amodal mask S
   weighted 1 on Ω (hidden at t, visible in aligned history; `Om4`) and outside S, 0 on V and unrecoverable hidden cells, λ = 1.
   Dropped at inference: one modal mask is scored. Also fixed: Ω was not zeroed on mode-2 frames (`fusion_data`). Pre-registered:
   kill if the synthetic readout AMGAP = L_am(noC) − L_am(Ĉ) on Ω at it 200 is ≤ 0 or positive on < 8/12 clips; else seq1-only
   screen (8 crossings, @lap) vs featbig: pass = box AUC reliance Δ ≥ +0.010 with CI > 0 and recognition reliance not lower; one
   seed, so a pass means "run a second seed". Smoke: AMGAP at init −0.015. Files: `feat_integral/amodal/run.sh`, `run.log`.
   **Result.** Synthetic AMGAP +0.073 (it 100), **+0.146 (it 200), positive on 12/12 clips** (kill rule passed); synthetic modal J
   adapter 0.600. seq1, 8 crossings, @lap, crossing bootstrap: box AUC featbigam 0.539 / its noC 0.439 vs featbig 0.551 / 0.540;
   reliance **+0.101 [+0.069, +0.128]** (8/8) vs +0.010; Δreliance +0.090 [+0.059, +0.119]; but featbigam − featbig box **−0.011
   [−0.034, +0.015]**: the reliance rise is the noC arm collapsing, not the with-C arm improving. Recognition conf featbigam 0.611
   vs featbig 0.588: **+0.023 [+0.000, +0.043]** (6/8); recognition reliance +0.034 [+0.015, +0.053] vs −0.005. Pre-registered
   bar met (one seed). Reading: the aux loss makes the reader depend on Ĉ (first time on real data), recognition improves, box AUC
   does not. Caveats: one seed; BigS16 was distilled on seq1 s4–s7 frames (unlabelled, both arms).

25. **Speed levers for the one-pass path** (`tools/speed_check.py`, `feat_integral/speed_check.log`, idle GPU). Tracker on GPU
   (same arithmetic, cumsum integral image, vectorised search, numpy Theil–Sen): **0.49 vs 4.62 ms/frame, identical tracks (0.00 px)**
   on 4 seq1 windows. Encoder: full 1024² eager 25.2 ms, `torch.compile` 19.7 ms; 512² crop eager 6.8 ms but the crop changes the
   features: median 1 − cos crop-vs-full s16 **0.194** / s8 0.034 / s4 0.004 vs frame-to-frame (2 apart) 0.003 (Hiera's global
   attention sees the whole image); compiling the crop failed (Inductor error). Projection (derived, not run end-to-end): 38.8 −
   4.1 (tracker) − 5.5 (compile) ≈ 29 ms/frame ≈ 34 FPS.
   Ported into `adapter/tracker.kernel_track` (2026-11-25; device of the input map, numpy input runs on CPU torch): identical
   tracks on all 32 full-pass windows (3027 frames, 0.000 px), 0.35 vs 4.55 ms/frame.
   **Pipeline wall-time pass (2026-11-26, laptop, measured).**
   - **Training step, FEAT/TIE/AMODAL, 24 targets: 4.13 → 2.40 s.** SAM2's CPU preprocessing (to-tensor, 1024² resize,
     normalise) was 46% of a step, because the 6 data workers occupy the CPU. `fusion_data.sam_feats` now runs it on the GPU,
     as inference already did. Not bit-exact: 1 − cos median 4.5e-5 vs 1.9e-3 for frames two apart (bf16 noise). It aligns
     training with inference, so new training runs are not bit-comparable to old ones.
   - **Recognition: 52 → 31 s** for 2 arms × 8 seq1 crossings, CSV byte-identical. Decoded frames are cached per crossing, and the
     float64 masked mean is computed on one channel of NFO's gray frames.
   - **dump_arms:** one process takes several arms and windows (`dump_arms A,B i j ...`); masks identical, 36 → 26 s for 4 dumps.
   - **Open, not bit-exact:** a GPU masked mean in recognition (≈ 5 vs ≈ 30 ms per stack) and `torch.compile` of the encoder (−20%).
   - **dump_arms, shared encoder pass:** the arms of one window share its encoder outputs (`inference.window_feats` →
     `fast_masks(feats=)`); masks identical to the old per-process path, **53 → 23 s** for 3 arms × 2 seq1 windows.

26. **Light, tied, teacher-free integral, from scratch (2026-11-25, pre-registered).** Question: can the 6-key `FeatIntegral`
   (≈ 40k parameters, no `Refined` / `BigS16`, no pixel teacher) with its evidence mass tied to the softmax, a = 1 − α_∅
   (`TIE=1`, no mass head, no mass_loss), replace item 24's frozen distilled BigS16 integral? This changes integral,
   teacher, tie and initialisation together. It is a replacement test, not an ablation.
   Recipe: `AMODAL=1 FEAT=1 TIE=1 AGG=none LAM=1`, Fusion and aggregator from scratch, 200 of 1000 cosine steps, seed 0
   (`feat_integral/tie/run.sh`). Losses: v3.4 + amodal on Ω (λ 1) + feat_loss on Ω (λ 1; synthetic clean-frame features, not
   a teacher). `train_step` now weights feat_loss (`LAM`) and mass_loss (`LAM_MASS`) separately.
   **Why the earlier tie failed, and why this one may not (derived).** Write ℓ_k for the key logits, ℓ_∅ for the null logit,
   Z = Σ e^ℓ. Then α = e^ℓ / Z, Ĉ = Σ α_k K_k + α_∅ B and a = σ(LSE_k ℓ_k − ℓ_∅). The gradients are
   ∂L/∂ℓ_k = α_k (⟨g_C, K_k − Ĉ⟩ + α_∅ g_a) and ∂L/∂ℓ_∅ = α_∅ (⟨g_C, B − Ĉ⟩ − a g_a).
   In background cells K_k ≈ B, so Ĉ ≈ B and only g_a moves α there: Ĉ does not identify α_∅ in background.
   The failed tie (fea385f) is consistent with this. The mass BCE covers the whole 256² map (Ω is ≈ 1–3% of it), and a
   at s4 is a bilinear upsample of α_∅ at s16. So at Ω edges the BCE pulled α_∅ up while feat_loss on Ω pulled it down,
   and Ĉ shrank toward B (s16 cos 0.493 → 0.412).
   Without mass_loss no term targets a directly, so that fight is gone by construction. Two risks remain:
   (R1) a is not anchored in background and may carry no person evidence. The review expects this: the reader's g_a ≈ 0 early,
   since the Fusion residuals are zero-initialised.
   (R2) The modal loss (hidden cells target 0, weight 0.3) could push a down on Ω through the reader.
   Pre-registered readouts on 12 synthetic val clips at it 200 (lines `TIE`, `AMGAP`, `AMGAPC`):
   - **kill** if AMGAP ≤ 0 or positive on < 8/12 clips (item 24's rule);
   - **tie failed for Ĉ (R2)** if cos(Ĉ16 − μ, T16 − μ) on Ω is no higher than the untied stage-0c light integral's value on the
     same clips (`tie/smoke.log`);
   - **tie gives no evidence mass (R1)** if mean a on Ω − mean a on background < 0.05. This is not a kill: the run then tests Ĉ only,
     and "a = 1 − α_∅ is evidence mass" cannot be claimed;
   - `AMGAPC` (Ω-only loss, C := B with a kept) separates use of Ĉ's appearance from a acting as a global flag, which AMGAP
     (C := B and a := 0, background-dominated weights) cannot.
   If synthetic passes, screen on seq1's 8 crossings @lap against featbigam, each arm with its -noC control, all scored in one run.
   **Primary: reliance** (with-C − noC). Box reliance > 0 with CI > 0 counts as using the integral.
   Secondary: the arm-vs-arm comparison. featbigam's BigS16 was distilled on unlabelled seq1 s4–s7 frames, so it has seen
   the screen sequence and the arm-vs-arm contrast is biased against this arm. "Viable" = box AUC and recognition each within
   −0.01 of featbigam; "better" = both > 0. One seed: a pass means "run seeds", not a claim.
   Smoke (`tie/smoke.log`, same 12 clips; one train step ran clean). Tied from scratch at init: cos16 on Ω **0.451**,
   a on Ω 0.852 vs background 0.807, AMGAP −0.091. Untied stage-0c reference: cos16 **0.508**, a 0.440 vs 0.125.
   So the R2 bar is cos16 > 0.508 at it 200.
   **Result (cluster job, all 4 arms dumped and scored there, tag @cl; `tie/train_cl.log`, `fullpass/downstream/box_gt_tie_cl.csv`,
   `recog_N5_uniform_meanfill_seq1_tie_cl.csv`; crossing bootstrap, 8 seq1 crossings).**
   Synthetic, it 200:
   - kill rule passed: AMGAP **+0.050**, positive on 11/12 clips;
   - R1 passed: a on Ω **0.934** vs background **0.821** (Δ 0.113 ≥ 0.05; init 0.852 / 0.807). The review had predicted a flat a;
     it was wrong;
   - R2 bar **missed**: cos16 **0.499** vs 0.508. But cos16 rises monotonically (0.452 → 0.468 → 0.499 at it 0 / 100 / 200),
     the opposite of the failed tie's drop to 0.412. This is consistent with an under-trained Ĉ (200 steps from scratch), not with
     the mass fight;
   - AMGAPC +8.5 (12/12): sign only. C := B with a kept is an input combination the reader never saw, so its size measures
     calibration, not information.
   seq1, feattie vs featbigam (the cluster's featbigam reproduces the laptop's: box 0.538 / noC 0.442 vs 0.539 / 0.439):
   - box AUC feattie **0.535** (noC 0.512) vs featbigam 0.538 (noC 0.442); feattie − featbigam **−0.003 [−0.009, +0.003]**;
   - box reliance (with-C − noC) **+0.023 [+0.011, +0.038]** (8/8) vs +0.096. The reader uses the integral, but its no-integral
     arm stays strong where featbigam's collapses;
   - recognition conf (per-crossing mean) feattie **0.608** (noC 0.594) vs featbigam 0.603 (noC 0.573); feattie − featbigam
     **+0.006 [−0.008, +0.018]**; recognition reliance +0.014 [−0.008, +0.040] (not shown > 0).
   Verdict against the pre-registered bars: **viable, not better.** Both metrics are within −0.01 of featbigam; box is not > 0.
   The light integral has 21,986 parameters (measured: `FeatIntegral(tie=True)`), against 5.9 M for BigS16 +
   the refiner. It uses no pixel teacher and has never seen seq1, and it matches featbigam on seq1 box AUC and recognition (one seed).
   Next (decided by the pre-registration: pass → seeds): freeze this recipe as the final configuration, train seeds 0–2, and
   score held-out seq2–4 against B0, T1 and the pixel adapter in one environment.

27. **Held-out seq2–4, 3 seeds of the tied light integral (2026-11-26, pre-registered, cluster).** The recipe of item 26
   (`AMODAL=1 FEAT=1 TIE=1 AGG=none LAM=1`, step 200) is frozen. Seed 0 is the item-26 checkpoint; seeds 1 and 2 are new
   (`adapter/feat_tie_seeds.sh`: training array → 6 mask arms × 24 crossings as array tasks → one scoring job).
   All arms are scored in one cluster run: feattie s0–s2 and their -noC controls, B0 (`b0-c2`), T1 (`t1-c2`), and the pixel
   adapter `adapter-pf/s1/s2` with its -noC controls. Paired bootstrap over crossings.
   Goal bar: **beats both B0 and T1 on box AUC AND recognition**, i.e. the 3-seed mean − baseline > 0 with CI > 0, and every
   seed > baseline.
   Context: the pixel adapter beats T1 on box AUC (+0.10) and is on par on recognition (+0.002 to +0.011), so recognition is the
   binding metric. Also reported: the gap to the pixel adapter (two encoder passes) and box reliance per seed.
   The speed claim is not part of this job; it needs one end-to-end timing run.
   **Result (job 2047755, all arms scored in one cluster run; seq2–4, 24 crossings, crossing bootstrap 4000;
   `fullpass/downstream/box_gt_heldout_tie_cl.csv`, `recog_N5_uniform_meanfill_heldout_tie_cl.csv`).**
   Box AUC: feattie s0/s1/s2 0.656 / 0.653 / 0.676 (noC 0.640 / 0.650 / 0.658); B0 0.611, T1 0.615; pixel adapter
   0.717 / 0.724 / 0.718 (noC 0.699 / 0.692 / 0.681). 3-seed mean − B0 **+0.051 [+0.023, +0.090]**, − T1 **+0.047 [+0.016, +0.087]**,
   every seed above both. − pixel adapter mean **−0.058 [−0.068, −0.049]**; − pixel noC mean −0.029 [−0.037, −0.022].
   Box reliance (C − noC) per seed +0.016 [+0.011, +0.021] / +0.003 [−0.001, +0.008] / +0.018 [+0.010, +0.027] (pixel +0.018 / +0.032 / +0.037).
   Recognition conf: feattie 0.697 / 0.695 / 0.694 (noC 0.691 / 0.692 / 0.692); B0 0.689, T1 0.687; pixel 0.689 / 0.696 / 0.698
   (noC 0.677 / 0.674 / 0.675). Mean − B0 +0.006 [−0.009, +0.025], − T1 +0.008 [−0.006, +0.024], every seed above both;
   − pixel adapter +0.001 [−0.011, +0.014]. Recognition reliance +0.006 / +0.003 / +0.002, all CIs span 0 (pixel +0.011 / +0.022 / +0.023).
   **Verdict: bar not met.** Box passes; recognition fails the CI clause (the pixel adapter fails the same clause: + 0.007
   [−0.012, +0.027] vs T1). As a replacement for the pixel adapter it loses 0.058 box AUC at equal recognition, and it is
   0.048 below the through-reader one-pass arm featbigpft (item 23: 0.710, @lap, one seed; environments differ, pf = 0.717 in both).
   Its integral contributes little on real footage (box reliance ≈ ½ the pixel route's, recognition reliance ≈ 0).
   Reading: the tie itself is not the problem (item 26 synthetic checks pass); the deficit is consistent with the from-scratch,
   teacher-free reader: even with the integral removed it sits 0.029 below the pixel noC arms (caveat: noC removes different
   inputs in the two models, so this is suggestive only). Not tested: the light tied integral read by the frozen pixel reader.


- `benchmark/nfo_vos/downstream/`:
  - `box_metrics.py`: OPE metrics, hand-vs-GT box ceiling;
  - `recognition.py`: masked recognition stacks + YOLO;
  - `dump_arms.py` + `.sbatch`: masks per method;
  - `report.py`: paired table;
  - `composite_label.py`: §6 label test.
- Masks: `results/benchmark/rr/masks/<method>/<window>.npz`. Scores: `results/benchmark/rr/downstream/*.csv`.
- B0/T1 masks were rerun on the cluster as `b0-c2` / `t1-c2`. Cluster nodes node2 and node22 lack CUDA kernels for the
  env and are excluded.
