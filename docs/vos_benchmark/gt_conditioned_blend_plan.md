# Implementation Plan: GT-Conditioned Exemplar Blending

Staged plan; each stage is evaluable on its own, so stop when gains flatten.

## Stage 0: Data and precomputation

For each frame k, the inputs are the RGB frame, disparity, empty-scene image, P_k, and G_k where it exists. For frames without G_k, use P_k in its place in everything below. N_k is computed as now.

Precompute once per exemplar Sᵢ: the binary mask and the vertical extent of each of the 4 bands. Don't precompute SDFs here; they are recomputed after alignment (Stage 2).

Signed distance: `edt(~M) - edt(M)` with `scipy.ndimage.distance_transform_edt` (negative inside, pixel units). On GPU, `cupy`/`cucim` provide the same transform.

## Stage 1: Boundary split and constraint sets

Cheap, deterministic, and everything downstream depends on it.

1. ∂G = G \ erode(G, 1).
2. ∂G_true = ∂G ∩ dilate(N, r), r ≈ 3 px — the G-edges facing see-through background.
3. ∂G_occ = ∂G \ ∂G_true — no constraint; keep for debugging.
4. Sample constraint pixels so the LP stays small (~2–5k rows):
   - **G_in**: all pixels within ~5 px inside ∂G, plus a sparse random sample of the deep interior.
   - **N_out**: N pixels within ~30 px of the union of shortlisted exemplars. N pixels far from every candidate can't be violated, so drop them.
   - **Edge**: all of ∂G_true.

Before continuing, render ∂G_true (green) and ∂G_occ (red) on 20–30 frames. If the split looks wrong, N is usually too sparse: loosen r or the background threshold (8) before touching anything downstream.

## Stage 2: Alignment by energy (replaces the current score)

For each exemplar, and per band independently (the energy decomposes over rows), grid-search the horizontal shift dx ∈ [−R, R]. Score with:

```
E = Σ_{G_in} relu(φ + m) + λ Σ_{N_out} relu(m − φ) + μ Σ_{edge} |φ|
```

During the search, evaluate E on the per-band SDF of the shifted mask (cheap). After choosing shifts, assemble the full shifted mask and recompute a single SDF — this removes seams at band boundaries.

Keep the top **K = 8** exemplars by E (not 3).

Defaults: m = 2 px, λ = 1, μ = 0.5, R = current shift range.

## Stage 3: LP-weighted SDF blend

Variables: weights w ∈ ℝᴷ plus one slack per constraint row. Solve with `scipy.optimize.linprog(method="highs")`:

```python
# Phi_*: (n_rows, K) SDF values of the K aligned exemplars at sampled pixels
# minimize  1ᵀs_in + λ·1ᵀs_out + μ·1ᵀ(e⁺ + e⁻)
# s.t.  Phi_in  w + m <= s_in        (G inside)
#      -Phi_out w + m <= s_out       (N outside)
#       Phi_edge w = e⁺ - e⁻         (contour on true edges)
#       w >= 0, 1ᵀw = 1, slacks >= 0
```

Final shape: φ_blend = Σ wᵢ φᵢ over the full-resolution SDFs.

Log the optimal objective (total violation in pixels) per frame — this is the per-frame quality signal.

## Stage 4: Residual refinement (gated)

Run only when the Stage 3 violation per constraint pixel exceeds a threshold (e.g. 0.5 px).

Set φ = φ_blend + δ and optimize δ with torch Adam (~200 steps, GPU). Loss: the same E (smooth hinge) + α‖δ‖² + β·TV(δ). Start with α large so the shape bends only where constraints force it.

Threshold φ at zero for the final mask; no redistancing needed.

## Stage 5: Label assembly

```
L = {φ < 0} ∪ G ∪ P   (optionally \ N)
```

Keep "minus N" as an ablation flag. With hard N constraints in the fit it should rarely matter; if it changes many pixels, suspect Stage 1.

## Stage 6: Evaluation

1. **Primary metric.** Against the hand-polygon subset: IoU, recall, precision, boundary F-score. Compare: current pipeline; Stage 2 + uniform top-3 blend; Stage 3 LP weights.
2. **Self-supervised recall check.** On frames with G, delete a horizontal band of G (torso, legs, …) to simulate occlusion, run the pipeline, and measure how much of the deleted region is recovered. Repeat with random bands.
3. **Ablations:**
   - μ = 0 vs μ > 0 (contribution of the boundary split)
   - K = 3 vs 8
   - Stage 4 on vs off
   - sweep over m
4. **Failure triage.** Sort frames by LP violation and inspect the top 5%: either poses missing from the bank or errors in N.

## Build order

1. Stage 1, then Stage 2's energy as a drop-in replacement for the current score → evaluate.
2. Add Stage 3 → evaluate.
3. Stage 4 only if triage shows pose mismatches.

Stages 1–3 are roughly 200 lines of numpy/scipy.
