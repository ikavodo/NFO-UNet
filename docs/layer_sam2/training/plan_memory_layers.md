# Plan: layer adapter + SAM2 memory (video mode), 2026-10-16. Draft for expert review

## 0. Evidence that motivates it (measured 2026-10-16, seq1, frozen silhouette pseudo-GT, 15 windows)

| arm | J | P | R |
|---|---|---|---|
| T1 (SAMURAI: memory, GT box at t0 only) | 0.488 | 0.824 | 0.515 |
| per-frame SAM2, **no memory**, kernel-f16 tracker box + point every frame | 0.507 | 0.772 | 0.571 |
| + layer adapter v3.4 (residuals from frame / composite / empty-scene features, layer-aware loss), step 200 | **0.546** | 0.804 | 0.596 |

**Reading (user, 2026-10-16).** Under fragmented occlusion, *where* the object is matters more than memory, because any
prompt is partial. So motion is the key input.

**Hypothesis.** Memory should still help once we have both the tracker prompt and the composite. Pixels seen in earlier
frames can be re-introduced through memory, so recall is retained over time without losing precision.

## 1. Existing pieces (code)

- **Per-frame pipeline** (`layers/fusion_adapter.py`, `fusion_data.py`):
  - kernel-f16 tracker, causal from the t0 box;
  - pixel composite C_t of t−n_h..t−1, aligned by the track, with evidence mass a_t (rate-adaptive change gate);
  - SAM2.1 b+ encoder features F_t, F_C, F_B at strides 16/8/4;
  - adapter residuals R^ℓ with zero-conv init, added to F_t^ℓ;
  - frozen decoder with sparse prompt = tracker box + tracker point (the point token corrected by the composite);
  - loss: layer-aware weights + focal α 0.67 + Focal Tversky.
- **SAM2 video internals** (`sam2_base.py`):
  - `track_step`;
  - memory attention on stride-16 tokens (`_prepare_memory_conditioned_features`, :630/:933);
  - decoder (:951);
  - memory encoder (:973 → `memory_encoder.py:174-177`: x = out_proj(fuser(pix_feat_proj(F) + mask_downsampler(σ(mask))))),
    num_maskmem 7, object pointers.
- **SAMURAI:** the same with Kalman-based memory selection.

## 2. Options

**A. Adapter inside a custom video loop (memory + per-frame tracker prompts).** Per frame t:
1. memory attention of F_t^16 against our bank → F_t^{16,mem};
2. F̂^16 = F_t^{16,mem} + R^16 (R computed from raw F_t, F_C, F_B as trained), F̂^{8,4} = F^{8,4} + R^{8,4};
3. decoder with the tracker box + point token;
4. memory encoder on (F_t, predicted mask) → bank (FIFO, num_maskmem), plus the object pointer.

Effect: improved masks enter memory, so memory improves gradually.
- Zero-shot first, using the v3.4 weights.
- Then train in video mode: unroll K frames, backpropagate through frozen memory attention and memory encoder into the
  adapter, truncated BPTT.

**B. Composite as an extra memory entry per frame** (user's test plan option A, "extra memory tokens + logit bias").
- Encode (F_C, mask_C) with the memory encoder; mask_C = a_t (soft), or the adapter's mask on C.
- Append it as one more memory entry for frame t, with its own temporal position (a learned "composite" tpos, or the
  t−1 slot).
- Composite tokens are aligned to frame t, so they share positions with the current tokens. A fragment visible at t but
  hidden at t−1 matches the composite's key at the same position. This is how memory recovers pixels that SAM2's own
  bank (built from its predicted masks) never contained.
- **Sharper variant:** attention bias +λ·log(a_t + ε) on composite keys, λ learned and zero-initialised, so the
  composite is trusted only where it has evidence.
  - Implementation: an additive attn_mask in the cross-attention of the memory-attention layers (RoPEAttention →
    scaled_dot_product_attention).

**C. Both** (A + B). The final target.

## 3. Sanity checks, pre-registered, cheapest first

- **SC1, plumbing.** The custom loop with the adapter off, composite off and prompt only at t0 must reproduce SAM2 video
  predictor masks (mean IoU ≥ 0.99 between the two). Guards against implementation bugs.
- **SC2, zero-shot value** (seq1 pseudo-GT, the same 15 windows, every 4th frame). Arms:
  - (i) per-frame adapter (0.546);
  - (ii) loop + memory + tracker prompts, no adapter;
  - (iii) = (ii) + adapter residual (A, zero-shot);
  - (iv) = (iii) + composite memory entry (B, no bias).

  Memory is worth training only if:
  - (ii) ≥ per-frame SAM2 (0.507) − 0.01, i.e. memory doesn't hurt the tracker-prompted SAM2; and
  - (iii) or (iv) ≥ (i) − 0.01, with R not lower.
- **SC3, recall over time.** R as a function of frame index within the window, per arm. Memory's claimed benefit is that
  recall decays less over time. Report the slope.
- **SC4, B's mechanism, on synthetic (exact layers).** The share of attention mass that newly-visible person tokens
  (visible at t, occluded at t−1) place on composite memory tokens, vs on SAM2's own memory tokens. If ≈ 0, B cannot
  work as claimed.

## 4. Training in video mode (only after SC2)

- Clips of 40 frames.
- Unroll K = 8 consecutive frames, starting at t ≥ n_h with the bank warmed on the preceding frames without gradient.
- Same loss, on all K frames.
- Frozen SAM2: gradients flow through memory attention and the memory encoder into the adapter (A), plus λ and the
  composite memory projection (B).
- Cost: about K× per-frame, so fewer steps or fewer targets.

## 5. Open questions for review

- Is the memory bank our own FIFO (num_maskmem 7) or SAMURAI's selection?
- Should per-frame tracker prompts make every frame a conditioning frame (SAM2 semantics), or should prompts enter only
  the decoder?
- How should the composite memory's temporal position be encoded?
- Does adding R^16 after memory attention, when trained without memory, transfer zero-shot?
- How to prevent error accumulation through memory, i.e. leaks feeding back?

## 6. Next week: poster deadline (user, 2026-10-16). Organised results with the adapter on video

**Must have:**
1. **Second annotator** on a subset of the hand frames: inter-annotator agreement, the reference ceiling for J/R.
2. **Larger benchmark** on the same windows and labels: B0, T1, the memory-free tracker-prompted SAM2, V̂, Diffusion-VAS,
   adapter per-frame, adapter-in-video. seq1 for selection, seq2–4 handAv for the single test.
3. **Video-mode training result**, worth a try even if worse. Stage 1 first: detached bank, gradient only into frame t.
   Same data path as v3.4.
4. **Understanding of our own method.** One-variable ablations on the video arm:
   - −composite (adapter-noC);
   - −tracker prompts;
   - residual before vs after memory attention;
   - simple loss vs layer-aware loss.

**Loss question:** test whether a simpler loss (global focal α + Focal Tversky, without layer weights) matches the
layer-aware one *in video mode*. Prefer the simpler one if it is within the CI.

**Not now:** composite / background from buffered features.

## 7. Closing the memory headroom: memory-conditioned read adapter over a corrected bank (design, 2026-10-19)

**Measured motivation** (`synth_memory.py`, `downstream/`):
- **Synthetic, exact visible masks V** (2-clip smoke test; 30-clip run pending):
  - per-frame adapter: J 0.628;
  - video, adapter masks written into memory (`write-pf`): 0.623;
  - **video, exact V written into memory: 0.714**;
  - video + read-side zero-shot adapter (residual before memory attention): 0.576.
- **NFO seq2–4 hand labels:** write-pf ties the per-frame adapter (0.763 vs 0.761) and beats video + prompts by
  +0.014 [+0.001, +0.028].

**Diagnosis:**
1. **Memory amplifies the quality of what is written.** Perfect content gives +0.09 J over the per-frame adapter; the
   adapter's own content gives ±0.
2. **The memory-blind adapter double-counts.** Its residual assumes nothing else supplies the object prior, but in video
   the decoder input F_t^mem already carries memory's extent: over-segmentation, R 0.93, P 0.76.

**Design, two adapter roles:**
- **A_w, the write adapter.** The frozen per-frame adapter (v3.4 it200, or the re-selected checkpoint). Its mask m_t^w
  is what SAM2's memory encoder receives: MemEnc(F_t, m_t^w), `memory_encoder.py:174-177`. It is open-loop: grounded in
  frame / composite / empty-scene evidence, never in past predictions. This equals the tested `write-pf` arm.
- **A_r, the read adapter (trainable).** Same architecture, initialised from A_w, with one extra stride-16 input group:
  the memory-conditioned feature F_t^mem and F_t^mem − F_t (zero-initialised weights, so step 0 = A_w).
  - Its residual R¹⁶ is added to **F_t^mem, the decoder input after memory attention**, not before it.
  - R⁸ and R⁴ go to the high-res skips as before; prompt = tracker box + point + δ.
  - It learns to add only what memory lacks (fragments new at t, removal of propagated FP).
- **Output:** Decoder(F_t^mem + R¹⁶_r, hi-res + R⁸⁴_r, token_r). The bank is always written by A_w, so the train-time
  bank and the test-time bank have the same provenance (no exposure bias; no scheduled sampling needed).

**Training: computationally ≈ the per-frame pipeline.**
- **Per clip (unchanged):** `fusion_data.build` gives features, composites, tracker, targets.
- **Bank (no grad):** A_w masks for the targets (one batched forward, already part of eval). Then the frozen SAM2 video
  loop over the clip's 40 frames with tracker prompts, A_w masks written into memory (the `oracle-mem` hook with
  labels = A_w masks). A capture hook stores F_t^mem (the `_forward_sam_heads` input) for the targets.
  - Extra cost per clip: SAM2 re-encodes 40 frames + 40 memory-attention / decoder / memory-encoder steps, all
    no-grad. About 1–2 s per clip (measured: synth_memory ≈ 4 video runs per clip).
- **Gradient only into A_r:** one decoder pass per target, as in per-frame training. No backprop through memory
  attention (R is added after it) and none across frames, so no BPTT.
- **Targets:** same max 32 per clip; 2 clips per step, as now.

**Loss.** Memory supplies recall, so the recall-heavy weighting is dropped:
- focal α 0.5 + Focal Tversky FN 0.5 / FP 0.5 (≈ Dice);
- the layer-aware weights W (ignore occluded person near visible fragments) kept, because that is a label convention,
  not a recall price.

Ablation: the same without W (the "simple loss").

**Init and LR:** A_r from A_w weights, memory-input conv zero-initialised. AdamW 1e-4 (lower than the original 3e-4,
since this is fine-tuning). Cosine schedule, 1,000 steps.

**Validation and selection (every 100 steps):**
- 12 synthetic val clips, exact-V J of the full video arm (write A_w, read A_r);
- seq1 15 windows, J vs SAM2-ceiling (dense ranking proxy, valid within the SAM2-video family);
- best checkpoint by synthetic J, with seq1 as a tie-breaker.

**Single test at the end** (`dump_arms` arm `vid-write-read`): seq2–4 hand J, box AUC, recognition, against T1,
adapter-pf, write-pf.

**Pre-registered success bar:** write-read ≥ adapter-pf + 0.02 J on synthetic (CI > 0), and ≥ adapter-pf on seq2–4
hand J and recognition (not worse beyond CI).

**Cluster:** same sbatch pattern.
- Exclude node2 and node22.
- Request an Ampere-or-newer GPU (bf16).
- Keep 2 clips per step and use the extra memory to run 2 seeds and the loss ablation in parallel as an array, rather
  than a larger batch: a larger batch changes the optimisation; parallel runs answer the open questions.
