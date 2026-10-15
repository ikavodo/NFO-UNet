# Learned layered refinement for SAM2 under fragmented occlusion — architecture draft (2026-10-14)

## v3 (2026-10-15): layers in feature space — the general strategy (user's direction)

**Data.** GPJATK walkers with occluder ≡ background, as in MovingMNIST-OcclusionBench. Settings:
- perlin mask, clump 24 px, density ≈ 0.45;
- background = random crops of the foliage photo (scale 960–2667 px, h-flip);
- `gpjatk_foliage.make_layered_clip(mode='perlin')`.

This is the closest arm in the layered experiment. It fails only R (0.911 vs ≤ 0.89) and the void share. More
foliage photos are deferred (candidate source: PercepTreeV1, norlab-ulaval) until training shows overfitting to one
photo.

**One operation, two layers.** Background and composite are the same thing: an aligned, weighted temporal
aggregation of SAM2 encoder features, under different motions.

  F_L(x) = Σ_i w_{L,i}(x) · F_i(x + Δ_{L,i}),   Σ_i w_{L,i}(x) = 1,   L ∈ {bg, obj}

- F_i: SAM2 image-encoder features of past frame i (`backbone_fpn`, strides 4 / 8 / 16; `sam2_base.py:606-620`).
- Δ_bg = 0, since the camera is static.
- Δ_obj,i = object displacement from i to t.
  - Training: GT, from the silhouette bbox or the MoCap pelvis (master_thesis `src/registration.py`), with
    jitter added.
  - Test: the tracker's displacement.
- w: predicted by one small shared network from (F_i, F_t, layer embedding). It learns *which past frames to trust
  at each location*. This is the learned replacement for the hand-made gate G and the age weights in
  `factorised_gtfree.composite`.
- Ancestor: Flow-Guided Feature Aggregation (Zhu et al., ICCV 2017, arXiv 1703.10025): neighbour features warped
  by flow, adaptively weighted, trained end to end. Our differences:
  - warps come from the tracker's displacement, not optical flow (flow is unreliable through foliage);
  - two layers instead of one;
  - the output is used for a visibility comparison, not for detection.
- **One shared network with a layer embedding**, not two networks. The two tasks differ only in Δ and in which
  frames to trust. A second network is the ablation.
- Pyramid: one aggregation per FPN level, shared weights across levels plus a level embedding.
  - Start at stride 16, where H2 bites and memory attention operates.
  - Add strides 8 and 4 only if the oracle test below shows that stride 16 cannot resolve sub-cell visibility.

**Targets** (loss only where noted):
- **Background.** Input: N noisy frames (sensor noise, flicker, JPEG, small sway; the person may pass). Target:
  Enc(B_clean). Loss over the whole frame.
  - Note: with a static camera, the pixel median of noisy empty frames is already ≈ clean. A *learned* background
    only pays off with illumination change, sway, or a person present during the estimate.
  - So the first baseline is Enc(pixel median), no learning.
- **Composite.** Target: Enc(clean frame t). The clean frame is I^g_t, the person on the background with no
  occluder. Loss only on dilate(S_t) ∧ O_t, where O_t = pixels visible in at least one aligned input frame (or in
  frame t).
  - Restricting to O_t matters: a body part hidden in every input frame has an unobservable current pose. Asking
    for it teaches hallucination.
  - The current frame t is an input, so visible parts are copied, not predicted.
  - Rejected alternative: "clean features blurred by GT motion". It would make the target the blurred average of
    the hand composite, which bakes in gait blur. The sharp current-pose target is better as long as the
    observability mask is used.

**Use inside SAM2** (after the features exist; order of trial):
1. Visibility score s(x) = cos(F_t, F_obj) − cos(F_t, F_bg).
   - Low s means the frame matches the background: penalise mask logits there (bad initialisations, leaks).
   - High s means the frame matches the object better: add evidence.
   - First as a fixed logit bias, then as a learned dense prompt (mask_decoder.py:205).
2. F_obj and F_bg as extra memory tokens in memory attention, or as an attention bias −λ·(1 − p_vis).

**Rhyme with documented failures.**
- Amodal completion alone lost on modal labels: Diffusion-VAS −0.057 J; post-hoc V̂ ±0.
- Any gain must come from the *comparison* s(x), not from F_obj itself.
- Do not evaluate F_obj by how amodal it is.

**Cheapest kill test (no training).** Oracle features on the synthetic clips:
- F_obj = Enc(clean frame t), F_bg = Enc(B).
- Compute s(x) at stride 16 (bilinearly upsampled) and at strides 8 / 4.
- Measure the AUC of s for visible person (V) vs occluded person (S ∖ V), per fragment-size bin.
- Pre-registered:
  - if the stride-16 oracle AUC on sub-cell fragments is < 0.7, stride 16 cannot carry the fix, so go to the
    high-res levels;
  - if all levels are < 0.7, feature-space layers cannot fix H2 even with perfect inputs, so stop.

**Oracle kill test, result** (measured 2026-10-15, `synth/oracle_visibility.py`):
- Setup: 24 held-out perlin clips, every 3rd frame, true anisotropic SAM2 cells.
- AUC of s for visible (V, binned by fragment size) vs occluded person pixels:

| level | all | [0,.5) | [.5,1) | [1,4) | [4,∞) cells |
|---|---|---|---|---|---|
| stride 16 (256 ch) | 0.959 | **0.612** | 0.716 | 0.854 | 0.965 |
| stride 8 (64 ch) | 0.989 | 0.939 | 0.973 | 0.979 | 0.990 |
| stride 4 (32 ch) | 0.990 | 0.969 | 0.981 | 0.985 | 0.990 |

- At stride 16 alone: the obj term gives 0.941, the bg term 0.896.
- **Verdict (pre-registered):**
  - stride 16 fails on sub-cell fragments (0.612 < 0.7), so it cannot carry the fix;
  - strides 8 / 4 pass, so feature-space layers go to the high-res levels.
  - This matches the v2 injection site on feat_s0 / feat_s1 (mask_decoder.py:224-225).
- **Upper bound only:** B is exact and noise-free here.

**Loss weighting for the composite target.** SAM2 tokens have large receptive fields, so a person token also
encodes its surrounding background. "Background doesn't matter" can only hold approximately. Two measures:
- Weight each token by the person fraction of its cell, inside dilate(S_t) ∧ O_t; weight 0 elsewhere.
- Give the composite head F_bg as an input, so it can copy the background context instead of predicting it.

**Cell-size correction.** SAM2 resizes to 1024×1024 without keeping the aspect ratio
(sam2/utils/transforms.py:32). A 16-px cell is therefore (W/64)×(H/64) native px: 15 × 8.44 on GPJATK, 12.5 × 9.38 on
NFO. Earlier fragment bins used 15² and 12.5².

Re-binned T1 recall with true cells:

| | [0,.5) | [.5,1) | [1,4) | [4,∞) cells |
|---|---|---|---|---|
| NFO seq2–4 handAv | 0.13 (n 122) | 0.44 (n 8) | 0.59 (n 22) | 0.82 (n 42) |
| synthetic perlin held-out | 0.08 | 0.17 | 0.52 | 0.89 |

NFO label area shares with true cells: 0.010 / 0.008 / 0.065 / 0.916. The layered-experiment bands were set
with the old cells. The qualitative pattern stands; the bands are not re-fitted (dataset choice closed by the
user, 2026-10-15).

## v2 (2026-10-14 evening): "Layer-SAM2" — the version to build first on GPJATK

Supersedes §3 below for phase 1. Written after the modal seq2–4 scoring (`score_void.py`):
- post-hoc V̂ = T1 on modal labels;
- T1's remaining headroom is mostly its outline (+0.099 J if missed pixels ≤ 3 px were added), then false
  positives (+0.069), with sub-cell fragments last (+0.017).

### Target: what a human means by "modal"
On GPJATK the layers give, per pixel:
- V = m(1 − a), the visible person;
- S = m, the amodal person.

Humans label something in between: they bridge small gaps between visible pieces and stay inside the body.
Parametrised target:

$$T_r = V \cup \big(\varphi_r(V) \wedge S\big), \qquad \varphi_r = \text{morphological closing, disk radius } r$$

- r = 0 is strict modal (SAM2's behaviour); r → ∞ is amodal S.
- r is measured from real human labels, not chosen.
- First estimate (seq1, n = 5 frames with both labels): closing the brush mask toward the polygon peaks at r ≈ 8 px,
  but only reaches J 0.64. So closing explains part of the annotator convention, not all of it. Re-estimate on
  GPJATK-vs-human once synthetic frames are labelled by a human.

### Inputs per frame t (all GT-free at test time)
- R_t: the frame.
- B̂: the empty scene, median of the warm-up frames.
- C_t: the person composite, past frames t−6..t−1 shifted onto t by the blob tracker's velocity (as now).

All three go through SAM2's frozen image encoder. The B̂ encoding is computed once per window. Each gives features at
strides 16 / 8 / 4 (image_embed, feat_s1, feat_s0).

### Three added pieces, each zero-initialised (at start the model is exactly SAM2 / SAMURAI)

| piece | input | where it enters SAM2 | code location | what it should buy |
|---|---|---|---|---|
| **B: background comparison** | [F(R), F(B̂), F(R) − F(B̂)] at s = 16, 8, 4 → small conv → D^s | D^16: added to the dense prompt embedding (same path as a mask prompt). D^8, D^4: zero-init 1×1 residual on feat_s1, feat_s0 | `mask_decoder.py:205` (`src + dense_prompt_embeddings`), `:224-225` | Precision: drop foliage false positives (+0.069 headroom). Recall where the person is visibly different |
| **A: composite** | F(C_t) at s = 16, 8, 4 | Same two paths as B (separate convs) | same | Extent S: where gaps may be bridged, and where not |
| **H: human-modal output token** | one new learned token, like SAM-HQ's HQ token (Ke et al., NeurIPS 2023, arXiv 2306.01567) | Appended to `mask_tokens`; its mask is the output. SAM2's own tokens stay untouched as the fallback | `mask_decoder.py:58, 185` | Learns the bridging convention T_r. The outline (+0.099 headroom) is trained against human-style targets |

Memory and prompting stay unchanged in phase 1: the S1 test showed injected masks snap back. H's mask can be written
to memory in phase 2.

### Loss (GPJATK)

$$\mathcal L = \mathcal L_{BCE+Dice}(\hat T, T_{r^*}) + \lambda_V\,\mathcal L(\hat V, V) + \lambda_S\,\mathcal L(\hat S, S)$$

V̂ and Ŝ are 1×1 auxiliary heads on the stride-4 features. They force B to carry visibility and A to carry extent,
so each piece's job is inspectable.

### Ablation table (one variable per row, same data, same trainable-parameter budget)

| row | B | A | H target | question it answers |
|---|---|---|---|---|
| 0 | – | – | – | SAM2 / SAMURAI zero-shot |
| 1 | – | – | T_{r*} | "more data + a new token" baseline (what a reviewer demands) |
| 2 | ✓ | – | T_{r*} | does the empty scene add beyond row 1? |
| 3 | – | ✓ | T_{r*} | does the composite add beyond row 1? |
| 4 | ✓ | ✓ | T_{r*} | together |
| 5 | ✓ | ✓ | V (r = 0) | is the gain from the target convention or from the layers? |

Evaluation:
- GPJATK held-out: J vs T_{r*}, and recall binned by fragment size.
- NFO seq1 (brush + polygon) and seq2–4 tier A (void), for development.
- seq2–4 tier B, labelled with void, as the final test.
- Bar: row 4 > row 1 on NFO, CI above 0. The human noise floor (inter-annotator J) is reported beside it.

### Step 1 before any training (no learning, ~half a day)
Build ~200 GPJATK composites:
- GPJATK walker × real NFO foliage layer × real NFO empty scene;
- fragment sizes spanning sub-cell to multi-cell.

Run T1 and B0, and compute the same breakdown as on NFO: recall by fragment size, missed area at the outline vs
inside vs outside, false-positive share, and the dilation test. Synthetic is a valid proxy only if the profiles
match NFO's within reason. Otherwise fix the compositing first (matte edges, foliage statistics, blur, noise).


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
