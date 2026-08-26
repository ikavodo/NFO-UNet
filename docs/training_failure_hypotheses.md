# Why NFO evaluation is still failing (F1≈0.04) despite a paper-matched training setup

**Status:** unresolved, handing off to a fresh session for deeper debugging.
**Audience:** a future Claude session (higher effort / stronger model), after re-reading
`docs/Person_Localisation_under_Fragmented_Occlusion.pdf`.

## Summary

Training itself looks healthy: `out/kth_train_20260825_160927/train.log` shows smooth,
monotonically-decreasing train/val loss, early-stopping at epoch 54 (paper's own logistic-loss
runs stop at 38-44/100, so this is in a plausible range), best val loss 0.00663. The model is
not garbage - spot-checking raw predictions against ground truth shows it genuinely finds the
correct location on many individual frames. But the aggregate NFO eval result is:

```
F1 score was 0.03895
tp: 2809, fp: 137934, fn: 686
```

`fp` is enormous - **137,934**, against roughly 7,300 total evaluation windows in the NFO
dataset. That's ~19 false positives per window on average, which is a different kind of problem
than "the model is bad at localizing" - it smells like a post-processing/evaluation pipeline
issue layered on top of a model that's at least partially working.

Everything below is a hypothesis, not a confirmed root cause. Confirm/refute before acting on it.

## What's already fixed and shouldn't need revisiting

(See git log for full detail; summarized so this doesn't get re-litigated.)

- Resolution mismatch (NFO was being fed at native 800x600 vs. the 224x224 the network trains
  at) - fixed, `data/nfo_processed` is now correctly padded+resized to 224x224 matching KTH.
- Loss/heatmap pairing - `LogisticLoss` + `HeatMap.CIRCLE` (not MSE + Gaussian), per the paper's
  own head-to-head comparison (logistic wins on final NFO F1 in their Table 4.3: 0.906 vs 0.739
  for the `n5,2` config). Confirmed correct, not a candidate for the current bug.
- `lr=1e-3`, `batch_size=16`, `seq_size=7`, `nth_frame=2`, color discretization (`cbest=2`) +
  swapping, geometric augmentation - all matched to the paper's stated values.
- `num_workers`/SLURM `--cpus-per-task` mismatch, `conda activate` failing in non-interactive
  SLURM shells - both fixed, unrelated to this issue but mentioned so they're not re-suspected.

## Correction to a prior "fix": `eval_method` should be `MaxEval`, not `ThresholdEval`

Earlier in this investigation `eval_method` was switched from `MaxEval` to `ThresholdEval`,
believing the paper's phrase "threshold evaluation metric" named this extraction algorithm. On
rereading `docs/Person_*.md` §3.2 ("Post-Processing"), that's wrong:

> As the method assumes **unimodality** of H by construction, the simplest approach... is to
> report the image (heatmap) coordinates of the pixel with the largest value. Nuisances might
> interfere the assumption of unimodality during training and inference. In this case, H has
> many local maxima that needs careful outlier detection as further post-processing **which is
> out of the scope of this paper**.

The paper's actual described method is a single global argmax per frame - exactly
`eval/max_eval.py:MaxEval.extract_centers` (`np.argmax(hm)`, one `BoundingBox`, done). The
"threshold" in "threshold evaluation metric" (§4.2, τ=0.1) refers to the **distance-tolerance
matching criterion** used to call a prediction TP/FP (shared by both `MaxEval` and
`ThresholdEval` via `AbstractEval.max_dist_error`), not to `ThresholdEval`'s Otsu+multi-contour
*extraction* algorithm - that class name is a coincidental collision with our codebase, not a
faithful reproduction of the paper's method.

This directly and structurally explains the fp explosion better than Hypothesis 1 below:
`ThresholdEval` emits 16-25 detections per frame (one per Otsu-surviving contour, no cap), while
`MaxEval` can never emit more than one detection per frame no matter how noisy/unbounded the
underlying output is. Recommended first move for the next session: **revert `eval_method` back
to `MaxEval` in `nfo_test`/`kth_val_test`, rerun, and compare `tp`/`fp`/`fn` before touching
anything else.** This is a one-line config change and should be tried before Hypothesis 1's
sigmoid fix - it's cheaper, and the paper itself explicitly disclaims handling multi-modal
outputs, so reproducing multi-modal *extraction* was never faithful to begin with.

Note this doesn't necessarily explain Hypothesis 2's spurious fixed secondary peak (with
`MaxEval`, that peak would still occasionally win the argmax and produce one wrong TP/FN pair
per affected frame) - but it should collapse the fp count from noise-contour multiplication,
which was independent of and much larger than that effect.

**Confirmed - rerun after the fix (2026-08-26, remote GPU, same checkpoint
`out/kth_train_20260825_160927`):**
```
F1 score was 0.307678
tp: 1657, fp: 5619, fn: 1838
```
vs. the original `ThresholdEval` run: `F1 0.03895, tp: 2809, fp: 137934, fn: 686`. fp dropped
24x (137934 -> 5619) and F1 rose 8x (0.039 -> 0.308) from this one-line change alone - strong
confirmation that `ThresholdEval`'s multi-contour extraction, not the model itself, was the
dominant source of the fp explosion.

Still far from the paper's reported ~0.9 F1, and now the residual errors look like genuine
localization mistakes rather than postprocessing artifacts: `fp` (5619) still exceeds `tp`
(1657) by ~3.4x, and `fn` roughly tripled (686 -> 1838, since a wrong single argmax now costs
both a miss and a false alarm on the same frame, instead of being drowned in noise-contours that
happened to sometimes include a correct hit). This reopens both remaining hypotheses as the next
real targets, now uncontaminated by extraction noise:
- Hypothesis 1 (sigmoid/output-scaling) may still affect *which* pixel wins the argmax under an
  unbounded, non-monotonic-under-noise score field.
- Hypothesis 2 (persistent spurious secondary peak) is now the more likely dominant explanation
  for the remaining fp/fn, since with a single-point extractor a competing fixed mode directly
  costs one wrong prediction per frame it wins on - exactly consistent with a ~3.4:1 fp:tp
  ratio if that mode wins a large minority of frames.

## Hypothesis 1 (secondary): the eval pipeline assumes a bounded [0,1] output; logistic-loss output is unbounded and never gets sigmoided

Still worth checking (it affects normalization/argmax stability even under `MaxEval`), but no
longer the primary suspect for the fp count - see correction above.

**The paper's own math** (Section 3, method description) defines the logistic-loss branch's
output explicitly as an *unbounded real-valued utility*: predicted heatmap pixels
`V(i,j) ∈ [-∞, ∞]`, with the *ground truth* pixels being the bounded classification labels
`Y(i,j) ≡ y ∈ {-1, 1}`. This is standard logistic-regression framing: `V` is a raw score, and the
associated probability is `sigmoid(V)`, not `V` itself.

**Grep confirms there is no `sigmoid` anywhere in this codebase:**
```
$ grep -rn "sigmoid" --include=*.py .
(no results)
```
`network/unet.py`'s `forward()` returns the raw final-conv output directly - no activation
function at all (matches `conv_9.3` in the paper's own architecture table, a plain `1x1` conv
with no listed activation). This is *correct* for training (both `MSELoss` and `LogisticLoss`
operate on raw/target-transformed values, not post-sigmoid probabilities), but it means the
value that reaches evaluation is the raw, unbounded `V(i,j)`.

**The eval pipeline was written assuming a bounded, probability-like heatmap:**
- `eval/abstract_eval.py:_preprocess`: `hm = batched_hms[i, 0, ...] * 255` - multiplying by 255
  only makes sense if the input is already roughly in `[0, 1]`.
- `eval/threshold_eval.py:normalize`: does a per-frame min-max stretch to fill the full
  `[0, 255]` range, `regardless of the input's absolute scale or confidence`. Feed it noise with
  a tiny true dynamic range, and it will still stretch that noise to full contrast.
- `AbstractEval.init_thresh` (an absolute-confidence gate, applied *before* normalization) is
  unset (`None`) in `nfo_test`/`kth_val_test` - and even if set, a threshold expressed "as a
  fraction of 255" is meaningless against a raw range that spans roughly `[-34, +15]` (measured
  directly, see below), not `[0, 1]`.

**Measured raw output range, spot-checking the actual trained model on real NFO windows:**
```
idx=23 out_min=-33.615 out_max=10.128 n_contours=24
idx=27 out_min=-32.189 out_max=11.626 n_contours=25
idx=29 out_min=-31.325 out_max=15.346 n_contours=23
```
16-25 separate contours per single frame. Each contour becomes one predicted center in
`ThresholdEval.extract_centers` (one `argmax` per surviving Otsu-thresholded contour, no cap on
count, no minimum-area filter). With ~1 ground-truth box per frame, most of those 16-25
predictions are automatically false positives - this arithmetic alone plausibly accounts for
the fp explosion (`137934 / ~7300 windows ≈ 19/window`, in the same range as the measured
per-frame contour counts).

**Cross-check against the paper's own reported numbers:** Table 4.3 (MSE vs. logistic loss,
NFO test set) reports **total** fp counts of 271 and 198 for their two logistic-trained
networks - across their *entire* test set, not per frame. Our per-frame contour count alone
(16-25) dwarfs their whole-dataset fp count. Whatever their actual postprocessing did, it did
not produce anything like this many spurious detections per frame - which argues fairly
strongly that something in *our* postprocessing pipeline, not the trained model itself, is the
dominant problem.

### How to verify this hypothesis

1. Apply `sigmoid` to the raw output before it reaches `AbstractEval`/`ThresholdEval` (either in
   `test_main.py`'s `evaluate()` before calling `retrieve_centers`, or as a wrapper), so values
   are bounded to `(0, 1)` before the `* 255` / min-max-normalize / Otsu-threshold chain. Rerun
   the same NFO eval and check whether `fp` drops to a sane order of magnitude.
2. Independently of the sigmoid question, check contour *size*: are the 16-25 contours per frame
   mostly tiny (1-5px) noise specks, or comparably-sized blobs? If tiny, a minimum-area filter in
   `ThresholdEval.extract_centers` (reject contours below some pixel-area threshold before the
   `argmax`-per-contour step) is an independent, complementary fix worth having regardless of
   the sigmoid question.
3. Re-read the paper specifically for any postprocessing detail beyond what's already extracted
   in `docs/Person_Localisation_under_Fragmented_Occlusion.md` - in particular whether they
   describe applying a sigmoid/normalization step for the logistic-loss branch specifically, or
   any contour/area filtering in their threshold-evaluation description (Section 4.2). This
   report's authors (i.e., the current session) may have missed something on the first pass.

## Hypothesis 2 (secondary, likely independent): a persistent, fixed spurious secondary peak

Spot-checking consecutive frames of the same NFO walk (`seq2`, indices 23-30) shows the model
alternating between the *correct* location and a *fixed, wrong* location:

```
idx=23 gt=(0.88,0.46) pred_argmax=(0.35,0.89)   <- wrong, fixed location
idx=24 gt=(0.87,0.46) pred_argmax=(0.90,0.47)   <- correct
idx=25 gt=(0.86,0.46) pred_argmax=(0.85,0.45)   <- correct
idx=26 gt=(0.86,0.46) pred_argmax=(0.84,0.45)   <- correct
idx=27 gt=(0.85,0.46) pred_argmax=(0.33,0.89)   <- wrong, same fixed location as idx=23
idx=28 gt=(0.84,0.46) pred_argmax=(0.34,0.88)   <- wrong, same fixed location
idx=29 gt=(0.84,0.46) pred_argmax=(0.33,0.88)   <- wrong, same fixed location
idx=30 gt=(0.83,0.46) pred_argmax=(0.36,0.88)   <- wrong, same fixed location
```

This is a *different* symptom than an earlier (5-epoch, MSE, sanity-config) model, which was
stuck at one wrong location on essentially every frame. This model has clearly learned the real
signal (idx 24-26 are excellent) but retains a second, competing, spatially-fixed mode around
`(x≈0.34, y≈0.88)` that sometimes outscores the correct one at `argmax`. Notably this candidate
location is bottom-left-ish - worth checking whether it coincides with something structural
(e.g. a consistent artifact from the pad-to-square + border-replicate step used to make both
KTH and NFO frames square before the 224x224 resize - KTH's padding is proportionally much
smaller than NFO's, given KTH's native 160x120 vs. NFO's 800x600, so a padding-replication
artifact that's negligible in KTH could be much more prominent in NFO and something the network
never learned to ignore).

### How to verify this hypothesis

1. Check whether `(≈0.34, ≈0.88)` (or a similar fixed point) recurs across *different* NFO
   sequences (seq1/3/4), not just seq2 - if it's the exact same absolute location regardless of
   scene content, that's strong evidence of a learned artifact bias rather than a real
   content-dependent confusion (e.g. vegetation, as we found for the classical tracker baseline
   in `tracking/`).
2. Visualize the padded/replicated border region of a few NFO frames directly and compare its
   proportional size/appearance to KTH's - if NFO's border-replication artifact is visually much
   more prominent, that supports the padding-artifact hypothesis over, say, a generic
   overfitting explanation.
3. This is independent of Hypothesis 1 - fixing the sigmoid/normalization issue won't
   necessarily remove this secondary mode, it'll just stop *drowning it in tens of thousands of
   unrelated noise-contour false positives*. Expect to still need to address this even after
   Hypothesis 1's fix.

## Suggested order of attack

1. ~~Revert `eval_method` to `MaxEval`~~ - **done, confirmed** (see rerun result above). fp
   24x lower, F1 0.039 -> 0.308.
2. Dig into Hypothesis 2 next using the spot-check approach already demonstrated in this
   session's history (load the model, iterate NFO windows, compare `argmax` position to ground
   truth, look for a recurring fixed wrong location across multiple sequences) - now the more
   likely dominant explanation for the remaining fp:tp≈3.4:1 gap.
3. If that doesn't fully explain the residual gap, revisit Hypothesis 1 (sigmoid/output-scaling)
   - it's no longer expected to matter for fp *count*, but could still affect argmax stability.
4. Whatever's left after both, compare against the paper's own reported ~0.9 F1 to judge whether
   the remaining gap is a training/data issue (e.g. NFO domain shift from KTH) rather than a
   pipeline bug.
