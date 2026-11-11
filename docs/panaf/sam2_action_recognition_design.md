# Behaviour recognition from SAM2 masks and features (plain SAM2 vs. SAM2 + NFO adapter) on PanAf500

Status: design, not built. 2026-11-11. Not yet reviewed (expert review is due before any model code is written).

## 0. Claim being tested

Two claims, kept apart:

- **(a) Validity.** Recognition built on SAM2 masks/features is a reasonable recogniser on PanAf500. It lands in the
  range of the paper's Table 3, measured on the same clips and metrics.
- **(b) Contribution.** Swapping plain SAM2 for SAM2 + adapter (the adapter was trained on NFO, the foliage-occlusion
  dataset, and is not retrained here) improves recognition, mostly on occluded apes. The gain should trace back to
  the better masks (§5).

(b) carries the paper and (a) makes it credible. (b) assumes the adapter first improves tracking on PanAf (§6,
the kill test).

## 1. What the adapter is (as built in `../NFO-UNet`, read-only)

- The SAM2.1 b+ **image** predictor, frozen, run **once per frame with no memory**.
  The memory variants are closed in NFO-UNet: the oracle bound there was +0.012 J.
- Per frame t, a kernel tracker (fixed-size box matched on SAM2 stride-16 features) produces a box and point prompt.
  An integral image C (frames t−6..t−1 shifted to align with the tracker's motion, change-gated against the empty scene
  B) is the evidence input.
- `Fusion` adds zero-initialised residuals R^ℓ to the encoder features at strides 16, 8 and 4:
  F̂^ℓ = F^ℓ + R^ℓ. F̂^16 is the decoder's image embedding and F̂^8, F̂^4 are its high-res skips. It also adds a
  learned correction δ to the tracker's point token.
- `decode()` (`fusion_adapter.py:136`) keeps only the mask logits. The mask decoder also returns `iou_pred`,
  `sam_tokens_out` (the output mask token, 256-d) and `object_score_logits`. Section 3 needs all of them, so call the
  decoder through a PanAf-side wrapper and leave NFO-UNet unmodified.

### Transfer risks to settle before trusting any number
1. **No empty scene B.** NFO's B is a median over warm-up frames with no person in them. Camera traps fire *because*
   an ape moved, so the ape is often there from the first frames (in `1W3EJbBJOf` it appears at frame 6). A median
   over the whole clip only works if the ape moves. Sitting is the most frequent behaviour (76k boxes), so for those
   clips the median *contains the ape*. Required check: render B for ~10 clips and look at it.
2. **Motion model.** The integral image's alignment is an x-only Theil–Sen fit, which suits a person crossing the
   frame. Apes walk toward the camera, climb vertically, and sit still.
3. **Night infrared and auto-exposure.** These may break the change gate, which was calibrated on NFO noise.
4. **The NFO recognition metric does not transfer.** It is YOLO confidence on registered crops, and COCO has no ape
   class. The PanAf behaviour labels replace it.

The first two risks are the same kind of failure the NFO docs record as "train/test parity".
They can make the adapter fail on PanAf for reasons that have nothing to do with foliage.

## 2. Shared stage: compute masks once, cache to disk

For each video, each `ape_id` is prompted **once** with its GT box on the first frame where it appears. After that
the pipeline runs on its own (tracker prompts each frame, as in NFO). This is computed once per arm and cached.
Clip building and recognition only read the cache.

Cached per (video, ape_id, frame):
| field | shape | source |
|---|---|---|
| mask logits | 256² (decoder low-res) → stored as RLE at native resolution | decoder |
| `obj_score` | scalar logit | `object_score_logits` |
| `iou_pred` | scalar | decoder |
| `tok` | 256 | `sam_tokens_out` (the output mask token) |
| `pool16` | 256 | masked average pool of F̂^16 (§3) |
| `ctx16` | 256 | average of F̂^16 over a ring around the mask (§3) |
| `geom` | 6 | box from the mask (cx, cy, w, h, normalised by frame size), mask area, centroid displacement since t−1 |

Also cache `pool16`/`ctx16` computed on the *plain* encoder features F^16 for the adapter arm. This makes the 2×2
in §5 free.

Storage: 180k frames × ~1.5 apes × ~800 floats ≈ 0.9 GB per arm in fp16. Compute: about 2–3 GPU-hours per arm for
all 500 videos at NFO's measured 61 ms/frame.

## 3. Per-frame object vector: three options

Notation: F̂^16 ∈ R^{256×64×64} (a 1024² input at stride 16), and M ∈ [0,1]^{64×64} is the mask probability
(sigmoid of the logits) area-resampled to 64×64.

**(i) Masked average pooling (MAP)** (named technique: Zhang et al., SG-One, 2018/2020, few-shot segmentation)

  x_t = Σ_ij M_ij F̂^16_{:,ij} / (Σ_ij M_ij + ε)

This is the average appearance of the ape's visible pixels. For the adapter arm, F̂ = F + R, so x_t includes the
adapter's residual. MAP alone throws away the context (tree trunk vs. ground), and context separates climbing,
hanging and sitting. So also compute ctx_t, the same average over a ring (the mask dilated by ~2 cells, minus the
mask itself).

**(ii) The decoder's output mask token** `tok` (256-d). This is what SAM2's video mode turns into the object pointer
(`obj_ptr_proj`, `sam2_base.py:522`). It is SAM2's own compressed summary of "this object in this frame", produced by
two-way attention between the prompt and F̂^16. It costs nothing extra, but it was trained to *produce a mask*,
not to describe pose, so it may carry little behaviour information. (i) vs. (ii) is an ablation, not a prior.

**(iii) Pixels:** the masked crop fed to a video network (§4B).

Why not the memory encoder: it is not in the adapter pipeline. Its output is a 64-channel spatial map built for
memory attention, not a per-object descriptor. Running it on adapter masks would bring the closed memory arm back.

**Motion has to come from `geom`.** Pooled appearance hardly changes between standing and walking, so most of the
signal separating walking/running/standing/climbing up vs. down is the change in box position and size over time.
Always feed `geom` alongside the appearance vector.

Per frame, the input vector is u_t = [x_t ; ctx_t ; geom_t ; v_t], where v_t ∈ {0,1} is a "mask present" bit,
1 if obj_score > 0 and mask area > 0. Its dimension is D ≈ 256+256+6+1.

## 4. Recognition models

### 4A. Small temporal head on frozen features (main experiment, cheap)
- **Clips:** the paper's protocol (Sakib & Burghardt 2020). Per ape, take maximal runs of one behaviour lasting at
  least t frames, and sample 16-frame windows from them. Test t = 16. Train t ∈ {16, 32}. The splits are the
  dataset's own (train 400 / val 25 / test 75 videos).
- The clip's GT label and frame range come from the annotations. The *inputs* come from the cache, i.e. from what the
  arm's tracker actually produced on those frames, so tracking failures propagate into recognition.
- **Model:**
  ```
  u_1..u_16 (D) → Linear(D,128) + learned temporal position embedding (16×128)
             → prepend [CLS] → 2× TransformerEncoderLayer(d=128, heads=4, ff=256, dropout 0.1)
             → LayerNorm([CLS]) → Linear(128, 9)
  ```
  The model has about 0.3M parameters. Frames with v_t = 0 stay in the sequence, with their appearance zeroed and the
  bit set. They are *not* masked out of attention: "the tracker lost the ape" is part of what each arm is judged on.
  A GRU is a valid alternative. With 16 steps, whichever is simpler to tune wins.
- **Training:** cross-entropy (as in the paper), AdamW 1e-3, cosine schedule, about 100 epochs, model selection on
  val C-Avg. Optionally, logit adjustment (Menon et al. 2020) for the tail, applied to every arm alike.
- **Metrics:** top-1 accuracy and C-Avg (mean per-class accuracy), as in Table 3. Also report top-1 and C-Avg on the
  occlusion subset (§6). Run 3 seeds.

### 4B. Masked-crop video network (bridge to the paper, more expensive, optional)
For each clip, crop around the union of the 16 mask-derived boxes, multiply by a dilated mask (or blur the
background), and resize to the network's input size. Fine-tune Kinetics-400 X3D with the paper's recipe. This is the
paper's own model with masks added at the input. The paper's comparable number is an X3D with GT boxes on every frame.
Here only the first frame gets a GT box, which makes this the harder setting. Do 4B only after 4A shows a gap between
arms.

### Integrating into the paper's models
Feeding SAM2 tokens *into* TimeSformer/X3D, e.g. as extra tokens, creates a new architecture whose benefit cannot be
separated from the change of backbone. Two clean levels instead:
- input level (4B), comparable to the paper;
- representation level (4A), the cheap test of claims (a) and (b).

## 5. Arms, and the 2×2 that attributes the gain

Main comparison: plain SAM2 vs. SAM2 + adapter. The SAM2.1 b+ checkpoint, prompts, clips and recognition head are
identical in both; only the arm changes.

The adapter changes *both* the mask M and the features F̂. To show that "better masks give better recognition",
pool all four combinations (free with the §2 cache):

| | plain features F | adapter features F̂ |
|---|---|---|
| **plain-SAM2 mask** | baseline | features only |
| **adapter mask** | masks only | full |

If most of the gain sits in the "masks only" cell, the mask-quality story holds. If it sits in "features only", the
adapter helps recognition through representation, which is a different claim from the one planned.

**The control for claim (a)** is a box crop with the same features and the same head: plain-SAM2 F^16 average-pooled
over the box from the mask (or the GT box), instead of over the mask. This is the one-variable test of "masks are a
better region than boxes". Full frame is not that control.

**External baseline:** Mueller et al. 2025 (V-JEPA with domain-adaptive pretraining, arXiv 2509.12193) is the current
state of the art on PanAf behaviour, ahead of the IJCV Table 3. See
`docs/lit_pass_2026-11-11_sam2_occlusion_behaviour.md`. Which PanAf subset they report on is unverified.

Later arms: DAM4SAM, SAMURAI, SAM2Long (training-free, occlusion- or distractor-oriented), and SAM3 with a concept
prompt as a prompt-free reference.

Fine-tuning the adapter on PanAf is not possible with its current loss, because PanAf has no masks. A box-supervised
mask loss such as BoxInst's projection loss (Tian et al., CVPR 2021) would make it possible. Deprioritised, since it
weakens the claim that the adapter generalises.

## 6. Tracking evaluation and the kill test (comes first)

- **Per-frame scoring:** the mask-derived box counts as a hit (TP) if IoU ≥ 0.5 with the GT box of the same
  `ape_id`. A prediction in a frame where that ape has no GT box is a false positive (FP). A GT box with no prediction
  or IoU < 0.5 is a miss (FN).
- **AP needs a ranking score, not calibration.** AP integrates precision over recall as predictions are admitted in
  order of their score, so only the *order* of the scores matters. With no score you get one precision/recall point.
  Candidate scores: `obj_score` (SAM2's presence head), `iou_pred`, and mean mask probability.
  - **Caveat:** the adapter was trained on a mask loss only. Its residual feeds the object-score head, which never
    received a training signal under the adapter, so the adapter arm's `obj_score` may rank badly even when its masks
    are good. Report AP under each score, and also recall at a fixed operating point.
  - Whether `obj_score` drops under occlusion is itself measurable: compare its distribution on occluded vs. clear
    frames of the occlusion subset. That could later serve as a signal for switching the adapter on or off.
- **Occlusion subset:** 30 videos from the **test split** (75 videos; the recognition head trains on the train
  split, so this keeps it unseen), split into about 15 with foliage occlusion and 15 clear controls (the controls show
  the adapter does no harm). Per track, label the occluded frame intervals (coarse start/end, levels none/partial/heavy),
  and record whether boxes cover only the visible part. This turns "the adapter helps under occlusion" into a
  stratified measurement.
- **Kill test:** plain SAM2 vs. SAM2 + adapter, tracking only, on these 30 videos. The pre-registered outcome:
  per-frame recall at IoU 0.5 on occluded frames, with the difference and a bootstrap 95% CI over videos.
  If the CI includes 0, stop and diagnose §1 risks 1–2 before going further. Recognition (4A) runs only if this passes.
  Before that, look at the empty-scene renders (§1, risk 1), which is a check that needs no GPU.

## 7. Open decisions
1. Which feature to pool, i.e. (i) vs. (ii): settled by the 4A ablation.
2. How to estimate B on camera-trap clips (median over the full clip vs. frames without an ape vs. a per-site median
   when several clips come from the same camera). This needs looking at the B renders.
3. Whether the head is trained once per arm (default) or once on plain features and then applied to both. Once per
   arm is fair; train-once isolates the input distribution shift.
