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
