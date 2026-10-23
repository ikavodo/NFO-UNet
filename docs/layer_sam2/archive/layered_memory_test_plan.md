# Layered Memory for SAM2: Test Plan

Oct 12, 2026 · @Ido Akov

## Setup and hypothesis

Separating SAM2's memory into a motion-aligned dynamic layer and an unaligned static layer should let the mask decoder ignore the static occluder and keep segmenting the object through persistent occlusion.

Scene assumptions: static camera, static foreground occluder (vegetation), moving object behind it with roughly horizontal 1D translation. In pixel space, aligning frames by the object's motion and aggregating gives an integrated object image. This plan tests the same idea on SAM2 features.

Where things sit in SAM2:

1. Hiera backbone + FPN gives current-frame features f\_t at stride 16 (64×64, 256-d for 1024 input).
2. The memory encoder fuses past features with predicted masks into memory features (64-d), stored in the memory bank with object pointers.
3. Memory attention (4 blocks, 2D RoPE) cross-attends f\_t to the memory bank.
4. The mask decoder takes the conditioned features plus prompt tokens; the prompt encoder also accepts a dense mask prompt.

The layers are built once per frame from the memory bank. Options differ only in where they enter: the mask prompt (C), memory attention logits (A), or the memory attention output (B).

## Building the layers

Shift each memory entry by the object's displacement since it was stored, then aggregate: the object reinforces, the occluder smears out. Aggregating without the shift does the reverse.

For memory entry i with offset Δt\_i = t − t\_i and object velocity v (feature cells per frame at stride 16):

```latex
D(x) = \operatorname{agg}_i \, M_i(x - v\,\Delta t_i), \qquad S(x) = \operatorname{agg}_i \, M_i(x)
```

```python
def build_layers(mem, dt, v, agg="mean"):     # mem: [N,C,H,W], dt: [N] (float)
    N, C, H, W = mem.shape
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=mem.device),
                            torch.linspace(-1, 1, W, device=mem.device), indexing="ij")
    gx = xs[None] - (2 * v * dt / (W - 1))[:, None, None]   # horizontal shift
    grid = torch.stack([gx, ys.expand_as(gx)], -1)
    warped = F.grid_sample(mem, grid, align_corners=True)
    valid = F.grid_sample(torch.ones_like(mem[:, :1]), grid, align_corners=True)
    if agg == "median":
        w = warped.masked_fill(valid < 0.5, float("nan"))
        D = torch.nanmedian(w, dim=0).values.nan_to_num(0.0)
    else:
        D = (warped * valid).sum(0) / valid.sum(0).clamp(min=1e-6)
    S = mem.mean(0)
    return D, S                                   # both in current-frame coordinates
```

Notes:

- **Which features.** Memory features (64-d, mask-conditioned) for options A and B; cached backbone features f (256-d) for option C, so the score compares like with like. SAM2 does not keep past backbone features by default, so cache them per frame.
- **Positions.** D and S are in current-frame coordinates, so they get the current grid's RoPE positions. Shifting RoPE positions alone is not a substitute: nothing gets averaged, so no layer separation happens.
- **Velocity.** Estimate v from the centroids of past predicted masks (least-squares fit over the bank); fall back to v = 0 when the fit is unreliable.
- **Sub-cell motion.** At stride 16, small motions are sub-cell; bilinear grid\_sample handles this.

## Baseline 0: warp-aggregate past masks

Run this first: if motion-compensated past masks already fill the occlusion holes, the feature-space options must beat it to justify themselves.

Apply `build_layers` to the past predicted mask logits (low-res, \[N,1,256,256\]) instead of features, with v rescaled to that resolution. Use the dynamic output D as the mask prompt for the current frame, or union it with SAM2's prediction.

- **Cost:** a few lines, no model change, no training.
- **Expected failure:** drift when v is wrong; no appearance evidence, so it cannot recover from a bad past mask.
- **Report:** J&F on occluded frames only, alongside vanilla SAM2.

## Option C: dense mask prompt

Score each location by whether the current feature looks more like the dynamic or the static layer, and pass that score through SAM2's existing mask-prompt input. No architecture change, no training, and the score map can be inspected directly.

```latex
s(x) = \tau \left[ \cos\big(f_t(x), D_f(x)\big) - \cos\big(f_t(x), S_f(x)\big) \right] + b
```

D\_f and S\_f are built from cached backbone features (256-d), not memory features.

```python
D_f, S_f = build_layers(feat_bank, dt, v)              # [C,H,W] each
s = tau * (F.cosine_similarity(f_t, D_f, dim=0)
           - F.cosine_similarity(f_t, S_f, dim=0)) + b  # [H,W]
mask_input = F.interpolate(s[None, None], size=(256, 256), mode="bilinear")
# pass as mask_input to the predictor / prompt encoder for frame t
```

- **Knobs:** τ ∈ {5, 10, 20}, b ∈ {−2, 0}, aggregation mean vs median.
- **Check first:** visualize s on occluded frames. Object should be positive, vegetation negative. If not, layer separation fails at stride 16 and A/B will not help either.
- **Risk:** the mask prompt can override memory evidence; compare with and without SAM2's own memory conditioning.

## Option A: extra memory tokens with logit bias

Append D and S as extra memory tokens in every memory cross-attention layer, and add +β to logits on D and −β on S. This is the main candidate: training-free and stays in-distribution for the frozen decoder.

```latex
\mathrm{Attn}(q) = \mathrm{softmax}\!\left( \frac{q K^\top}{\sqrt{d}} + [\,0_{mem};\; +\beta\,1_D;\; -\beta\,1_S\,] \right) V
```

```python
# inside each memory cross-attention block, after projecting D and S to k/v
k = torch.cat([k_mem, k_D, k_S], dim=-2)
v = torch.cat([v_mem, v_D, v_S], dim=-2)
bias = torch.cat([q.new_zeros(n_mem),
                  q.new_full((n_D,), beta),
                  q.new_full((n_S,), -beta)])
out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)  # float mask is added to logits
```

- **RoPE:** apply the current-frame grid positions to k\_D and k\_S. Leave object-pointer tokens unrotated, as in SAM2.
- **Knobs:** β ∈ {0, 1, 2, 4}; layers {all 4, last 2}; static tokens on vs off (β\_S = −∞).
- **Ablation:** D only, S only, both. β\_S = −∞ isolates whether adding D helps or removing S helps.
- **Limitation:** −β only lowers S's share of attention. It never actively suppresses the static component.

## Option B: differential attention

Subtract the static readout from the memory readout, the feature-space analogue of background subtraction. Use this only if A leaves the occluder leaking into the mask.

```latex
\mathrm{out}(q) = \mathrm{Attn}\big(q, [K_{mem}; K_D], [V_{mem}; V_D]\big) - \lambda\, \mathrm{Attn}(q, K_S, V_S)
```

```python
out = (F.scaled_dot_product_attention(q, torch.cat([k_mem, k_D], -2), torch.cat([v_mem, v_D], -2))
       - lam * F.scaled_dot_product_attention(q, k_S, v_S))
```

Related: Differential Transformer (Ye et al., 2024).

- **Knobs:** λ ∈ {0.1, 0.25, 0.5}; last layer only vs all.
- **Risk:** subtraction pushes features off the frozen decoder's distribution. Keep λ small, or fine-tune only the memory-attention layers with λ learnable per head.
- **Watch for:** holes in the mask where the object's own appearance resembles the static layer (e.g. object parked long enough to enter S).

## Test plan

Run in order; each step gates the next.

| # | Test | Training | Go on if |
| --- | --- | --- | --- |
| 0 | Baseline 0: warp past masks | None | Always (sets the bar) |
| 1 | Option C: inspect score map s | None | Object positive, vegetation negative on occluded frames |
| 2 | Option C: J&F | None | Beats vanilla SAM2 on occluded frames |
| 3 | Option A: β sweep + D/S ablation | None | Beats Baseline 0 |
| 4 | Option B: λ sweep | None, then memory-attention fine-tune | Occluder still leaks after A |

Evaluation:

- **Metric:** J&F split into fully visible, partially occluded, and fully occluded frames. Gains should concentrate in the occluded splits.
- **Data:** a few clips matching the assumptions first (static camera, horizontal motion behind vegetation), then a held-out set to check nothing breaks on ordinary videos.
- **Velocity:** report results with estimated v and with ground-truth v, to separate motion-estimation error from the method.
- **Diagnostics:** save D, S, and s as images for every clip; most failures will be visible there before they show in metrics.

* [ ] Cache backbone features per frame in the SAM2 video predictor
* [ ] Implement `build_layers` + velocity fit from mask centroids
* [ ] Baseline 0
* [ ] Option C
* [ ] Option A
* [ ] Option B (if needed)
