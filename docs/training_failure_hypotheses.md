# Why NFO evaluation disagreed with the paper — resolved

**Status:** root cause found and confirmed by retrain. All four earlier hypotheses are closed
(two were real but minor, two were refuted). Investigated 2026-08-26 against
`out/kth_train_20260825_160927`; scale fix confirmed via retrain `out/kth_train_20260826_123145`
(Prec. 0.474 → 0.802, see bottom of this file). Small residual gap (0.802 vs. paper's 0.96)
remains, see "Confirmed: retrained with `rand_zoom_out`" section below for next steps.

## TL;DR

Nothing is wrong with the model, the training, the loss, or the post-processing. Three separate
things were wrong with the *comparison*, and one thing is genuinely wrong with the *data*:

| # | Finding | Effect | Status |
|---|---------|--------|--------|
| 1 | We reported **F1**; the paper reports **Prec. = TP/(TP+FP)** | apples-to-oranges | fixed in `test_main.py` |
| 2 | Unannotated NFO frames were evaluated → **3,781 guaranteed FPs** | 67% of all reported FPs | fixed in `dataset/testing_dataset.py` |
| 3 | **KTH training persons are 71–144px tall; NFO persons are 45–64px** — disjoint | costs **0.46 precision** | root cause; `rand_zoom_out` added, needs retrain |
| 4 | `eval_transforms: []` — validation ran on a different distribution than training | cosmetic here | fixed in `config/train_config.py` |

Corrected score for the **existing** checkpoint, no retraining: **Prec. 0.474** (was reported as
"F1 0.308"). Paper, same config (N=7, f=2): **0.96**.

## The root cause: person scale

`AbstractDataSet` ground-truth box heights, in pixels at 224×224 (sentinel `-1,-1,1,1` rows
excluded — those are the authors' "not annotated" marker, not real boxes):

```
KTH train:  median 109px   p5 71px   p95 144px
KTH val:    median 117px   p5 79px   p95 141px
NFO test:   median  54px   p5 45px   p95  64px
```

**The entire NFO range sits below KTH's 5th percentile.** The network has never seen a person
this small, and a U-Net is not scale-invariant.

Decisive test (`scratchpad/scale_test.py`): upscale each NFO window 2× and crop 224 back out,
with the ground truth placed at a *random* offset from the crop centre so an "always guess
centre" strategy cannot win:

```
n=100  NFO as-is (54px person):          prec = 0.520
       NFO 2x upscaled (108px person):   prec = 0.980
       "always guess centre" baseline:   prec = 0.060
```

Same weights, same post-processing, same τ. Only the apparent person size changed, and
precision went from 0.52 to 0.98 — i.e. to the paper's 0.96. The scale gap *is* the KTH→NFO gap.

The paper says this itself, in the Conclusion (p. 8): *"The experiments show further that motion
of vegetation causes failure cases and that **the training data is not capturing sufficiently the
different scales of a person**."* We hit exactly the failure they flagged.

Supporting evidence — in-domain performance is perfect, so nothing upstream of the domain shift
is broken:

```
KTH val (same distribution as training): prec = 1.000 (100/100), loss 0.0070
NFO per sequence: seq1 0.250, seq2 0.450, seq3 0.533, seq4 0.650
```

Per-sequence spread tracks vegetation density (the paper's Figure 6 ordering), and wrong
predictions are **scattered** (std 0.13–0.37 in both axes), not clustered. This is
content-dependent confusion under occlusion, not a systematic artifact.

### What was done about it

`utils/transform_utils.py:rand_zoom_out(min_scale, max_scale)` — shrinks the frame by a random
factor and pads back to 224 with replicated borders, moving heatmap and boxes with it. Wired
into `kth_train` as `rand_zoom_out(0.4, 1.0)`, which maps the KTH median (109px) down to ~44px
and so covers NFO's 45–64px range. Has a 200-case self-check (run the `__main__`-style snippet
in the commit message, or re-derive: box centre and heatmap centroid must track the content).

**Not yet validated — this needs a retrain to confirm.** It is placed before `reduce_colors` so
the INTER_AREA interpolation cannot invent intermediate grey levels after quantisation.

## Finding 1: wrong metric

The paper's measure (§4.2, and Tables 2/3) is **precision** and the raw **TP count**:

> The method's precision or sensitivity (Prec.) is then derived from the TP/FP values by […]
> Threshold τ is set in the experiments to 10% of the image width/height which is
> 0.1 × 224 = 22.4 pixel.

Table 3, N=7 / f=2 (our config): **Prec. 0.96, TP 3234** out of 3,379 annotated test images.
F1 appears nowhere in the paper. `test_main.py` now logs `Prec.` alongside F1.

## Finding 2: unannotated frames were counted as false positives

`TestingDataSet` indexed every window whose sequence neighbourhood was in range — 7,276 of them.
Only **3,495** of those centre frames carry a ground-truth box. `MaxEval` emits exactly one point
per window unconditionally, so each of the remaining **3,781** windows was a guaranteed false
positive.

The arithmetic from the old run confirms this exactly:

```
tp + fp = 1657 + 5619 = 7276   <- exactly the window count
tp + fn = 1657 + 1838 = 3495   <- exactly the annotated-window count
fp - 3781 = 1838 = fn          <- one prediction per frame: matched -> tp, else fp AND fn
```

The paper puts this explicitly out of scope (§4.2):

> This assumes presence of a person in each test image. The absence of persons introduces
> true/false negatives and the method's ability to reject a localisation hypothesis which is
> out of the scope of this paper.

and again in the Conclusion: *"As the current method is not a detector, it does not allow images
with empty scenes."*

`TestingDataSet._construct_ds_entries` now requires a non-empty `bbs` to index a window,
mirroring `KthDataSet`'s existing `and gauss_file is not None`. Verified: 3,495 windows, and
every indexed window has a box.

## Finding 4: validation ran on a different distribution than training

`train_main.py:validate()` uses `c.eval_transforms`, which was `[]`, while `train()` used
`reduce_colors(4)` + `rand_color_swap()`. So the two logged losses were never comparable, and
early stopping (patience 15, fired epoch 54) was driven by a mismatched signal. Measured:
`loss(kth_val, raw) = 0.0070` vs `loss(kth_val, quantised) = 0.0114`.

Real bug, now fixed (`eval_transforms: [rand_zoom_out(0.4, 1.0), reduce_colors(4)]`), but it did
**not** cause the F1 gap — in-domain precision is 1.000 either way.

## Refuted hypotheses — do not re-open these

### Hypothesis 3 (loss plateau ~100× above the paper's floor): premise was a misread plot

The claim was "paper reports logistic loss <5e-5 by epoch 40; ours plateaus at 0.0066, ~100×
higher." Figure 7 (p. 8) is a **dual-axis** chart, rendered here directly from the PDF:

- **Left axis: `L_mse`**, ticks 1.00E-03 … 5.00E-05.
- **Right axis: `L_log`**, ticks 2.00E-02, 1.00E-02, 8.00E-03, 6.00E-03, **4.00E-03**.

The `5e-5` figure is the bottom of the **MSE** axis. The two `L_log` curves (n5,2 and n7,2) end
near the bottom of the **right** axis, i.e. around **4e-3**. Our best val loss of **0.0066** and
train loss of **0.0036** at epoch 54 sit right on the paper's actual logistic-loss floor. There
was never a 100× gap, and in-domain precision of 1.000 independently shows the model is not
undertrained. No LR schedule, weight-decay, or init investigation is needed.

### Hypothesis 1 (missing sigmoid): mathematically cannot matter under `MaxEval`

`MaxEval.extract_centers` is `np.argmax(hm)`, and everything upstream of it in
`AbstractEval._preprocess` is `hm * 255` with `init_thresh=None`. `argmax` is invariant under any
monotonically increasing map, and `sigmoid` and `×255` are both monotone. Applying a sigmoid
therefore cannot change which pixel wins — not "probably doesn't matter", cannot. (It *did*
matter under the old `ThresholdEval`, which is why the hypothesis looked plausible; that
extractor is gone.)

### Hypothesis 2 (persistent fixed spurious secondary peak): refuted by measurement

The original evidence was seq2 indices 23–30 landing repeatedly near (0.34, 0.88). Those are
eight *consecutive* frames of one walk — consecutive frames naturally share a wrong answer.
Sampled across all four sequences, wrong predictions are diffuse:

```
seq1 wrong_pred mean=(0.42,0.65) std=(0.27,0.34)
seq2 wrong_pred mean=(0.25,0.45) std=(0.31,0.37)
seq3 wrong_pred mean=(0.36,0.44) std=(0.13,0.39)
seq4 wrong_pred mean=(0.23,0.49) std=(0.30,0.33)
```

No fixed mode. No padding artifact either — `pad_img_and_bb_to_square` uses `BORDER_REPLICATE`
for KTH and NFO alike (`prep_nfo_data.py` calls the same `scale_and_pad_img_to_square`), and both
are 4:3 sources so the padding is proportionally identical.

### Colour quantisation at test time: refuted experimentally

`nfo_test` has `test_transforms: []` while training used `reduce_colors(4)`, and §3.2 does
present `f_rad` as method *pre-processing*, so this looked like a real train/test mismatch.
Applying it at test makes localisation **worse**:

```
seq2, n=120:  raw grayscale        prec = 0.442
              reduce_colors(4)     prec = 0.342
```

Plausible reason: training applies `rand_color_swap` *after* quantisation, so the network is
already invariant to the palette and keys on spatial structure — which raw grayscale supplies
more of. Leave `test_transforms` empty.

## Also checked and clean (so these aren't re-suspected)

- **NFO frame/label alignment.** Filenames are 0-indexed contiguous (`00000_or.jpg` …); gt keys
  are 0-based with max = frame count − 1. No off-by-one.
- **Geometry.** `prep_nfo_data.py` passes the box through the same
  `scale_and_pad_img_to_square` as the image, so pad+resize keeps them consistent. Visually
  confirmed by overlaying gt boxes on processed frames.
- **Config fidelity.** `seq_size=7`, `nth_frame=2` = the paper's best N=7, f=2 cell.
  Circle radius 0.07 × 224 = 15.7px against τ = 22.4px.

## Remaining divergences from the paper (lower priority than the scale fix)

Both would need a retrain to evaluate, and neither is likely to matter as much as scale:

1. **Running sequences are in the training set** (32 train + 32 val). The paper dropped them:
   *"class Running comprises sequences of fast motion which reduced the dataset to 225 sequences
   useful for our needs."* Training on fast motion may teach the net that large inter-frame
   displacement means "person", which is also what moving vegetation looks like.
2. **Only 8 of 25 KTH persons are used for training.** `kth_labels.zip` contains 301 labelled
   sequences across all 25 persons; `data/kth_train` uses 96 of them (~13k samples), with a
   50/50 person split to `data/kth_val`. The paper labelled all sequences.

## Confirmed: retrained with `rand_zoom_out`, 2026-08-26

```
Prec. was 0.802289, F1 score was 0.802289
tp: 2804, fp: 691, fn: 691
```

(`out/kth_train_20260826_123145`, early-stopped epoch 45, best val loss 0.00643 - config
verified via `train_cfg.pkl`: `rand_zoom_out` present in both `train_transforms` and
`eval_transforms`, `seq_size=7`, `nth_frame=2`.) Precision jumped **0.474 → 0.802**, most of the
way to the paper's 0.96. `fp == fn` exactly and always will under this fixed pipeline: every
window now has exactly one GT box and one prediction, so a wrong argmax costs one fp and one fn
on the same frame - `Prec.` and `F1` are now numerically identical by construction, not
coincidence.

**Remaining gap (0.802 vs. 0.96):** the two lower-priority divergences flagged above are the
next things to try, in order:
1. Drop Running-class sequences from `data/kth_train`/`data/kth_val` (paper explicitly excludes
   fast motion: "reduced the dataset to 225 sequences useful for our needs") - training on large
   inter-frame displacement may be teaching the net that motion magnitude alone signals "person",
   which also describes windblown vegetation.
2. Widen the KTH person split beyond the current 8/25 persons (paper labelled all 25).
3. If neither closes it, break down `Prec.` per NFO sequence (as before: seq1-4) to check whether
   the residual gap is still vegetation-density-correlated (expected, matches paper's Fig. 6) or
   has shifted to a new failure mode now that scale is fixed.
