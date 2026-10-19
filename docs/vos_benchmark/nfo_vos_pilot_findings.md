# NFO video object segmentation under fragmented occlusion: findings (status 2026-10-19)

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

**Layer adapter.**
- **Inputs:** SAM2 encoder features at strides 16/8/4 of three images:
  - the frame;
  - a **composite**: frames t−6…t−1, aligned to t by the tracker's motion and change-gated against the empty scene,
    plus an evidence mass a;
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

## 4. Main result (seq2–4, 46 windows; recognition on all 552 composites)

| method | hand J | P / R | box AUC | P@20 | recog AP50 | recog conf |
|---|---|---|---|---|---|---|
| B0 | 0.755 | 0.90 / 0.81 | 0.641 | 0.767 | — | — |
| T1 | 0.750 | 0.89 / 0.82 | 0.646 | 0.778 | 0.916 | 0.664 |
| trkbox | 0.250 | 0.25 / 0.96 | 0.709 | 0.840 | — | — |
| trk | 0.716 | 0.82 / 0.85 | 0.693 | 0.833 | — | — |
| **adapter-pf (ours)** | **0.761** | 0.86 / 0.86 | **0.717** | **0.848** | **0.926** | **0.678** |
| vid-prompts | 0.748 | 0.82 / 0.89 | 0.713 | 0.859 | — | — |
| vid-adapter-pre | 0.718 | 0.76 / 0.93 | 0.735 | 0.871 | — | — |

The empty recognition cells were scored on the 414-composite subset only: see §5 and
`results/benchmark/rr/logs/recog_N7_meanfill_all.log`.

**adapter-pf − T1:**
- box AUC **+0.070 [+0.056, +0.086]**;
- hand J +0.010 [−0.011, +0.034];
- recognition conf +0.015 [−0.014, +0.042].

**Claim supported:** the per-frame layer adapter **tracks clearly better than SAMURAI and ties it on visible-pixel
segmentation and recognition.** No other method is at or above T1 on all three.

## 5. What each ingredient does (seq2–4, one variable at a time)

| ingredient | comparison | hand J | box AUC | recog conf |
|---|---|---|---|---|
| motion prompt (tracker box + point every frame) | trk − T1 | −0.034 | **+0.046** | −0.080 |
| empty-scene layer cue | adapter-pf without composite − trk | **+0.043 [+0.002, +0.099]** | +0.007 | — |
| composite (past pixels, open-loop) | adapter-pf − adapter-pf-noC | +0.001 | **+0.018 [+0.013, +0.022]** | **+0.022 [+0.010, +0.033]** |
| memory (SAM2 video loop) | vid-prompts − trk | +0.032 | +0.020 | — |
| adapter inside memory | vid-adapter-pre − vid-prompts | **−0.031 [−0.053, −0.011]** | +0.022 | −0.001 |

The −0.031 row is over-segmentation: P 0.76, R 0.93.

**Reading:**
- **The motion prompt buys tracking and costs precision.**
- **The empty-scene cue restores the precision.**
- **The composite adds recall that is real person evidence** (recognition up), not just extent.
- **Memory propagates extent:** in the video loop, precision falls and over-extension grows.

**Oracle bound on memory:** SAM2 video with a *non-causal* label written into memory for past frames.

| memory content | hand J | P / R | box AUC | recog conf |
|---|---|---|---|---|
| SAM2-ceiling (GT-box-prompted SAM2.1-L, bidirectional) | 0.756 | 0.87 / 0.84 | 0.668 | 0.689 |
| frozen pseudo-GT (extent-leaning) | 0.726 | 0.78 / 0.90 | — | — |
| *for reference: adapter-pf, no memory* | 0.761 | 0.86 / 0.86 | 0.717 | 0.709 |

The recognition figures here are on the 414-composite subset. **Scope of this bound:** it is only as good as the
content written in. SAM2-ceiling matches the hand masks at J 0.716, *below* the per-frame adapter's own masks (0.761).
So the result is: memory filled with SAM2-ceiling-quality masks does not beat the memory-free adapter, and memory
transmits the precision/recall balance of what is written in. Whether better-than-SAM2-ceiling content (e.g. the
adapter's own masks, or a trained write path) makes memory useful is open; see §8.

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
| SAM2-b+ on a GT-motion composite, t±1, weights 0.7^\|i−t\| | per frame | 0.690 | 0.77 / 0.87 | not tested (worse than frame t) |
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

## Code map

- `benchmark/nfo_vos/downstream/`:
  - `box_metrics.py`: OPE metrics, hand-vs-GT box ceiling;
  - `recognition.py`: masked composites + YOLO;
  - `dump_arms.py` + `.sbatch`: masks per method;
  - `report.py`: paired table;
  - `composite_label.py`: §6 label test.
- Masks: `results/benchmark/rr/masks/<method>/<window>.npz`. Scores: `results/benchmark/rr/downstream/*.csv`.
- B0/T1 masks were rerun on the cluster as `b0-c2` / `t1-c2`. Cluster nodes node2 and node22 lack CUDA kernels for the
  env and are excluded.
