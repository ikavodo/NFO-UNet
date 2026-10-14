# Learned layered refinement for SAM2 under fragmented occlusion — architecture draft (2026-10-14)

Status: design only, nothing built. Motivation: the training-free V̂ = (M_C ∧ change) ∪ T1 beats T1 on seq1 (+0.042 J vs
polygons, GT-free) but every hand-set parameter failed to transfer: a fixed threshold adds nothing on seq2–4 (0/46
windows), Otsu leaks (4.8× T1 area) and loses to T1 vs brush. Goal: learn every such parameter from layered
synthetic data (GPJATK person / occluder / background layers), keep SAM2 frozen.

## 1. What is learned (replaces a hand-set value)

| Hand-set today | Where | Learned replacement |
|---|---|---|
| change threshold τ, robust-z normalisation | V̂ rule | background-comparison branch + per-frame normalisation (module B) |
| recency weights exp(−age/3.5), cleaning σ(z − 3) | composite C | temporal attention weights over history (module A) |
| constant-velocity Theil–Sen alignment | composite C | coarse shift from the blob tracker + learned residual offsets (module A) |
| "trust background vs composite" | — | per-pixel layer gate (module C), reliability learned |
| envelope ∧ change ∪ base, add-only | output | zero-init residual on SAM2's stride-4 path and output logit (module D) |
| SAM2 prompt on the composite | S1 | predicted envelope as a dense prompt (module E, phase 3) |

## 2. Training data: layered compositing

Per synthetic clip, with known layers: background B_t (static scene, optionally jittered), person P_t with matte
m_t (GPJATK walker), occluder O_t with matte a_t (foliage):

$$I_t = a_t\,O_t + (1 - a_t)\big(m_t P_t + (1 - m_t) B_t\big)$$

Exact targets, all pixel-level:

$$V_t = m_t (1 - a_t)\ \text{(visible person)}, \qquad S_t = m_t\ \text{(amodal extent)}, \qquad a_t\ \text{(occluder)}$$

Randomisations, each mirroring an observed failure:
- the empty-scene estimate given to the model is the median of K person-free frames, K ∈ [10, 200] (seq2 had 23,
  seq1 168: the transfer failure of 2026-10-14);
- illumination drift between that estimate and the frames (seq1 segment 0: empty scene far in time);
- camera jitter (common affine on B and O, independent of the person) — the background layer becomes unreliable;
- occluder sway (smooth elastic warp of O, a over time) — static-occluder assumption broken;
- occluder scale/density so that visible fragments span sub-cell to multi-cell (H2: recall 0.00 below half a cell);
- grayscale, sensor noise, compression.

## 3. Architecture (SAM2 frozen; trainable ≈ 1–3 M parameters)

Notation: F^s(X) = frozen SAM2 image-encoder features of image X at stride s ∈ {4, 8, 16} (feat_s0, feat_s1,
image_embed). R_t = frame t; B̂ = empty-scene estimate; H = {t−N, …, t−1}.

**A. Person layer (learned composite).** Coarse shift δ_i from the blob tracker; residual offsets from a small
correlation + deformable alignment block at stride 8; temporal attention fusion:

$$\tilde F_i^s = \mathrm{DeformAlign}\big(F^s(R_i);\ \delta_i + \Delta\delta_i\big), \qquad
w_i(x) = \operatorname{softmax}_{i \in H}\ g_\phi\big([\tilde F_i^s(x),\ F^s(R_t)(x),\ F^s(\hat B)(x)]\big)$$

$$C^s(x) = \sum_{i \in H} w_i(x)\,\tilde F_i^s(x)$$

Ancestor: EDVR's PCD alignment + TSA fusion (Wang et al., CVPRW 2019, arXiv 1905.02716; from memory). The composite
lives in feature space, so no second SAM2 pass on a composite image is needed.

**B. Background layer comparison.** Learned comparison instead of cosine / pixel SSD, plus a full-resolution pixel
branch (pixel change was the strongest cue: AUC 0.94 on sub-cell pieces):

$$D_B^s(x) = h_\psi^s\big([F^s(R_t),\ F^s(\hat B),\ F^s(R_t) - F^s(\hat B),\ F^s(R_t) \odot F^s(\hat B)]\big), \qquad
D_B^{pix} = \mathrm{CNN}_{small}\big([R_t, \hat B, |R_t - \hat B|]\big)$$

Per-frame instance normalisation of D_B is the learned counterpart of the robust z. Ancestor: BSUV-Net (Tezcan,
Ishwar & Konrad, WACV 2020, arXiv 1907.11371; from memory): background subtraction for unseen videos from the
current frame plus empty and recent background frames.

**C. Layer gate (reliability).** Per pixel, the current frame attends over its two layer explanations, the
attention weights being the learned reliability:

$$\alpha(x) = \operatorname{softmax}\Big(\big\langle q(F^s(R_t)(x)),\ k(\cdot)\big\rangle\Big)\ \text{over}\ \{C^s(x),\ F^s(\hat B)(x)\} \oplus r(\text{frame stats})$$

r(·) = small MLP on per-frame statistics (residual of B̂ outside the person, spread of the aligned history) — the
reliability term; under jitter it should move weight from B to C (testable by construction).

**D. Injection into SAM2, zero-initialised (identity at start).**

$$\text{feat}_{s1}' = \text{feat}_{s1} + Z_8\big(u_8\big), \qquad \text{feat}_{s0}' = \text{feat}_{s0} + Z_4\big(u_4\big), \qquad
\ell'(x) = \ell(x) + Z_{out}\big(u_{pix}(x)\big)$$

u_s = fused features [α-gated C^s, D_B^s, F^s(R_t)]; Z = 1×1 conv initialised to 0 (ControlNet, arXiv 2302.05543).
feat_s0/s1 are added in SAM2's output upscaling only (mask_decoder.py:224-225), the one place sub-cell detail
enters. Output head precedent: SAM-HQ (Ke et al., NeurIPS 2023, arXiv 2306.01567). No add-only constraint by
default; monitor removals (2026-10-13: removing SAM2 pixels by a background cue cost −0.040 J).

**E. Phase 3 only — memory and prompt.** C^16 encoded with SAM2's memory encoder (mask-conditioned, so non-person
cells are marked) as extra memory tokens with a learned per-head bias β_C·r_C; predicted envelope Ŝ as a dense
prompt every N/2 frames (S1).

## 4. Losses (multi-task, possible only with layered data)

$$\mathcal L = \mathcal L_{BCE+Dice}(\hat V, V) + \lambda_S\,\mathcal L(\hat S, S) + \lambda_O\,\mathcal L(\hat a, a)
+ \lambda_{small}\,\mathcal L_{V}\big|_{\text{fragments} < 1\ \text{cell}}$$

Auxiliary heads: Ŝ from module A (trains extent), â from module B (trains occluder). The last term up-weights
visible components smaller than a stride-16 cell, the H2 failure mode.

## 5. Protocol

- Train on synthetic GPJATK composites only. Dev: NFO seq1 (hand polygons/brush). Test: NFO seq2–4 human labels
  (the user's and a colleague's), never used for any choice.
- Baselines: B0, T1, V̂ (hand rule, the non-learned ancestor), SAM2 fine-tuned on the same synthetic data (does the
  layer structure matter, or just more data?).
- Ablations, one module at a time replaced by its hand-set counterpart (table in §1); B̂-only; C-only; no gate.
- Report params, FLOPs, memory, wall-clock; 3 seeds.

## 6. Phases and gates

1. D(out) + B only: learned replacement of the threshold. Gate: on seq1 ≥ V̂ (raw τ), and non-zero, non-leaking
   additions on seq2–4 (added area within the range seen on seq1).
2. + A, C, D(s0/s1): sub-cell pieces and reliability. Gate: jitter-variant synthetic test shows the gate moving to C.
3. + E (memory / prompt). Gate: recall gain over phase 2 on human labels.

## 7. Risks

- Synthetic-to-real gap: matte edges of GPJATK cut-outs, foliage statistics. Mitigate by compositing on real NFO
  empty scenes where possible.
- B̂ quality at test time (short warm-ups): covered by the K randomisation, but real drift may exceed it.
- Novelty: layered / background-aware VOS and BSUV-style subtraction exist; the claim must be the combination
  (frozen SAM2 + learned layer comparison at stride 4 + layered synthetic supervision) evaluated on fragmented
  occlusion. A literature pass is required before building.
