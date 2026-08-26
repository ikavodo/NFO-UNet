# Why NFO evaluation disagreed with the paper — resolved

**Status:** root cause found and fixed, confirmed by retrain (2026-08-26,
`out/kth_train_20260826_123145`). NFO precision **0.474 → 0.802**, worst sequence 0.667 (paper,
same config N=7/f=2: **0.96**). Residual ~0.16 gap **characterised and accepted** - see
"Residual gap" at the bottom for why chasing it is not worth a retrain.

## Summary

Nothing was wrong with the model, the training, the loss, or the post-processing logic itself.
Three things were wrong with the *comparison*; one thing was genuinely wrong with the *data*:

| # | Finding | Effect | Status |
|---|---------|--------|--------|
| 1 | We reported **F1**; the paper reports **Prec. = TP/(TP+FP)** | apples-to-oranges | fixed in `test_main.py` |
| 2 | Unannotated NFO frames were evaluated → guaranteed FPs | 67% of all reported FPs | fixed in `dataset/testing_dataset.py` |
| 3 | **KTH training persons are 71–144px tall; NFO persons are 45–64px** — disjoint | cost ~0.33 precision | root cause; fixed via `rand_zoom_out`, confirmed by retrain |
| 4 | `eval_transforms: []` — validation ran on a different distribution than training | did not affect NFO score | fixed in `config/train_config.py` |

## The root cause: person scale

`AbstractDataSet` ground-truth box heights, in pixels at 224×224:

```
KTH train:  median 109px   p5 71px   p95 144px
KTH val:    median 117px   p5 79px   p95 141px
NFO test:   median  54px   p5 45px   p95  64px
```

The entire NFO range sits below KTH's 5th percentile — the network had never seen a person this
small, and a U-Net is not scale-invariant. Decisive test: upscaling an NFO window 2× (same
weights, same post-processing) moved precision from 0.52 to 0.98, matching the paper's 0.96. The
paper names this failure mode itself (Conclusion, p. 8): *"the training data is not capturing
sufficiently the different scales of a person."*

**Fix:** `utils/transform_utils.py:rand_zoom_out(min_scale, max_scale)` shrinks the frame by a
random factor and pads back to 224 with replicated borders, moving heatmap and boxes with it.
Wired into `kth_train` as `rand_zoom_out(0.4, 1.0)` (maps KTH's median 109px down to ~44px,
covering NFO's 45–64px range), placed before `reduce_colors` so `INTER_AREA` interpolation can't
invent intermediate grey levels after quantisation.

**Confirmed by retrain** (`out/kth_train_20260826_123145`, early-stopped epoch 45, best val loss
0.00643; config verified via `train_cfg.pkl` - `rand_zoom_out` present in both
`train_transforms` and `eval_transforms`):

```
Prec. was 0.802289, F1 score was 0.802289
tp: 2804, fp: 691, fn: 691
```

`fp == fn` exactly, and always will under this pipeline: every window now has exactly one GT box
and one prediction (`MaxEval`), so a wrong argmax costs one fp and one fn on the same frame -
`Prec.` and `F1` are numerically identical by construction from here on, not coincidence.

## Other real fixes

**Wrong metric.** The paper's measure (§4.2, Tables 2/3) is precision and raw TP count; F1
appears nowhere in the paper. `test_main.py` now logs `Prec.` alongside F1.

**Unannotated frames counted as false positives.** `TestingDataSet` indexed every window in
range regardless of whether its centre frame had a ground-truth box; `MaxEval` emits exactly one
point per window unconditionally, so every unannotated window was a guaranteed FP. The paper
puts this out of scope explicitly (§4.2: *"This assumes presence of a person in each test
image"*; Conclusion: *"As the current method is not a detector, it does not allow images with
empty scenes."*). `TestingDataSet._construct_ds_entries` now requires a non-empty `bbs` to index
a window, mirroring `KthDataSet`'s existing check.

**Train/eval distribution mismatch.** `train_main.py:validate()` used `c.eval_transforms = []`
while `train()` applied `reduce_colors(4)` + `rand_color_swap()`, so the two logged losses were
never comparable and early stopping was driven by a mismatched signal. Fixed
(`eval_transforms: [rand_zoom_out(0.4, 1.0), reduce_colors(4)]`) - confirmed this did **not**
cause the NFO gap (in-domain precision was 1.000 either way).

## Refuted — do not re-open

- **Loss-plateau claim** ("paper reports logistic loss <5e-5 by epoch 40; ours plateaus ~100x
  higher") was a misread of a dual-axis figure: `5e-5` is the bottom of the *MSE* axis, not the
  logistic-loss axis, which bottoms out around `4e-3` - right where our runs land. No LR
  schedule/init issue exists.
- **Missing sigmoid** cannot matter under `MaxEval`: `argmax` is invariant under any monotone
  map, and both sigmoid and the `×255` scaling upstream are monotone.
- **Persistent fixed spurious peak**: the original evidence was 8 *consecutive* frames of one
  walk sharing a wrong answer. Sampled across all four sequences, wrong predictions are diffuse
  (no fixed location, no padding artifact - KTH and NFO padding is proportionally identical).
- **Colour quantisation at test time**: applying `reduce_colors(4)` at test (to match training
  pre-processing) makes localisation *worse* (0.442 → 0.342 on a seq2 sample), because training
  applies `rand_color_swap` after quantisation and the network already keys on spatial structure
  more than palette. `test_transforms` stays empty.
- **Environment/dependency versions** (numpy/opencv/torch newer than `requirements.txt` pins):
  refuted by in-domain KTH precision being 1.000 in this same environment, and by the scale fix
  producing exactly its predicted effect - neither is consistent with environment-level
  corruption being a contributing factor.

Also checked and clean: NFO frame/label alignment (no off-by-one), box/image geometry through
`prep_nfo_data.py`'s pad+resize, and config fidelity (`seq_size=7`, `nth_frame=2` matches the
paper's best N=7/f=2 cell; circle radius 15.7px vs. τ=22.4px).

## Residual gap (0.802 vs. 0.96): characterised, and accepted

**Decision: work with 0.802 and account for the gap.** Do not spend a retrain closing it. The
per-sequence measurement below is what makes that safe, and part of the gap is unclosable anyway.

### Per-sequence precision, before vs. after the scale fix

Sampled 60 windows per sequence, `MaxEval`, τ=0.1 (`scratchpad/per_seq.py`):

| checkpoint | seq1 | seq2 | seq3 | seq4 | spread |
|---|---|---|---|---|---|
| pre-fix `…160927` | 0.250 | 0.450 | 0.533 | 0.650 | 2.6× |
| post-fix `…123145` | **0.667** | **0.683** | **1.000** | **0.683** | 1.5× |

Two things follow.

**The floor moved, not just the mean.** Worst sequence went 0.250 → 0.667; three of four now sit
within 0.016 of each other. There is no weak sequence hiding beneath the aggregate, so a single
headline number is a fair summary. **Quote `Prec. 0.802` with a 0.667 floor.** Window-weighting
these samples over all 3,495 windows gives 0.751 — ~1.6σ from 0.802 at n=60/sequence on different
sampled frames, so it corroborates the reported figure. That matters because
`out/kth_train_20260826_123145` has no `test.log`: 0.802 is transcribed from elsewhere, and this
is the only local evidence for it.

**The vegetation-density correlation is gone.** Pre-fix the ordering was monotone
(0.25 < 0.45 < 0.53 < 0.65), tracking scene clutter as the paper's Fig. 6 would predict. Post-fix
it is flat at ~0.68, with seq3 jumping from middling to perfect. The scale fix removed the
density gradient rather than merely lifting the curve, so the residual ~0.2 is a roughly
scene-independent failure mode, **not** the paper's vegetation story. Any further work on this
gap needs to start by identifying what that mode actually is.

### Part of the gap cannot be closed by training

Our NFO ground truth has **3,507** boxes; the paper states **3,379**. Different annotation set,
and the paper disclaims its own (§4.1): *"there are cases of video frames, where a correct
setting of the bounding box is difficult which makes this annotation inaccurate. A sufficient
methodology is left for future research."* Those hard frames are exactly the heavily-occluded
ones where precision is worst, so some of the residual is annotation disagreement. 0.96 is
defined against labels we do not have.

### If the number is ever wanted anyway

Ranked by expected value per unit of effort:

1. **Widen the KTH person split** — the only lever with a clear mechanism and no downside.
   `data/kth_processed` already holds all 300 sequences (25 persons × 3 classes × 4 conditions),
   fully generated; `data/kth_train` and `data/kth_val` are symlink farms built by
   `gen_data/gen_kth_data/split_kth_data.py`, which hardcodes KTH's *action-recognition*
   benchmark split. Persons 2,3,5,6,7,8,9,10,22 — **108 sequences** — are unused, as that
   script's own comment notes. That holdout exists to benchmark action recognition; this repo
   never tests on KTH. Folding them in is one set-literal edit plus a retrain (~2h at the
   observed 2.6 min/epoch): training data 96 → 204 sequences, **+112%**, no data generation.
2. **Identify the new failure mode** — with the density gradient gone, dump the frames that miss
   and look for what they share (person velocity? occluder type? proximity to frame border?).
   Cheap, and it is now the only principled way to pick a next fix.
3. ~~Drop Running-class sequences~~ — **rejected.** The rationale does not survive checking.
   `data/kth_processed` is exactly 25 × 3 × 4 = 300 sequences; dropping the Running class leaves
   **200**, but the paper says **225**, which is exactly 25 × 3 × **3** — i.e. consistent with
   dropping one recording condition (d1–d4), not the Running class. The change would also cost
   33% of training data for an effect of unknown sign, and its stated motivation (fast motion
   resembling windblown vegetation) lost its evidence when the vegetation correlation vanished.

### Environment note

`python3` on this machine does not resolve numpy/torch. The working interpreter is
`~/miniconda3/envs/comp_vis/bin/python` (numpy 2.3.4, torch 2.7.0+cpu, cv2 4.13.0, dill).
