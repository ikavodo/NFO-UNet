# Why NFO evaluation disagreed with the paper — resolved

**Status:** root cause found and fixed, confirmed by retrain (2026-08-26,
`out/kth_train_20260826_123145`). NFO precision **0.474 → 0.802** (paper, same config N=7/f=2:
**0.96**). Residual ~0.16 gap: two untried divergences remain, see "Next steps" at the bottom.

## Summary

Nothing was wrong with the model, the training, the loss, or the post-processing logic itself.
Three things were wrong with the *comparison*; one thing was genuinely wrong with the *data*:

| # | Finding | Effect | Status |
|---|---------|--------|--------|
| 1 | We reported **F1**; the paper reports **Prec. = TP/(TP+FP)** | apples-to-oranges | fixed in `test_main.py` |
| 2 | Unannotated NFO frames were evaluated → guaranteed FPs (67% of all reported FPs) | fixed in `dataset/testing_dataset.py` |
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

## Next steps (residual 0.802 → 0.96 gap)

1. Drop Running-class sequences from `data/kth_train`/`data/kth_val` - the paper excludes them
   ("reduced the dataset to 225 sequences useful for our needs"). Training on fast motion may
   teach the net that large inter-frame displacement alone signals "person", which also
   describes windblown vegetation.
2. Widen the KTH person split beyond the current 8/25 persons (paper labelled all 25).
3. If neither closes it, break down `Prec.` per NFO sequence (seq1-4) to check whether the
   residual gap is still vegetation-density-correlated (expected, matches the paper's Fig. 6) or
   has shifted to a new failure mode now that scale is fixed.
