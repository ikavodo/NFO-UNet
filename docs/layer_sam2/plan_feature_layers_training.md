# Training plan: learned static / dynamic feature layers for SAM2 (written 2026-10-15 for a fresh session)

Read first:
- `docs/layer_sam2/layered_sam2_architecture.md`, section **v3**, including the oracle result and the cell-size
  correction;
- `docs/layer_sam2/synthetic_nfo_design.md`, last two sections: the data generator and why the perlin arm is used;
- `docs/vos_benchmark/nfo_failure_log.md`.

## 0. Context (one paragraph per fact; all measured unless marked)

**Goal.** Beat SAM2.1 (B0) and SAMURAI (T1) on modal person segmentation under fragmented foliage (NFO) by
recovering missed visible pixels.

**Failure mechanism (H2).** T1 finds 13% of visible fragments smaller than half a SAM2 cell, against 82% of
fragments ≥ 4 cells (NFO seq2–4 hand labels, true cells).

**SAM2 cell.** SAM2 resizes every frame to 1024×1024, so a 16-px cell is (W/64)×(H/64) native px: 15 × 8.44 on
GPJATK, 12.5 × 9.38 on NFO.

**Idea.** Two feature layers per frame t:
- a static *background* layer F_bg (what the empty scene looks like in SAM2 feature space);
- a dynamic *composite* layer F_obj (what the unoccluded person looks like *now*, accumulated from past frames
  aligned by the tracker's motion).

A visibility score s(x) = cos(F_t, F_obj) − cos(F_t, F_bg) then says where the current frame shows the person.
Using s inside SAM2 is the *next* plan, not this one.

**Oracle result** (perfect F_obj = Enc(clean frame), F_bg = Enc(empty scene), synthetic held-out clips). AUC of s,
visible vs occluded person pixels, sub-cell fragments:
- stride 16: 0.612;
- stride 8: 0.939;
- stride 4: 0.969.

So the layers must live at **strides 8 and 4** (`high_res_feats`, 64 and 32 channels).

**This plan.** Train the network that produces F_obj, and check how much of the oracle AUC survives with learned
features, noisy inputs and imperfect alignment.

## 1. Architecture: tracker-guided deformable aggregation (the modern form of the idea)

The idea is unchanged from FGFA (Zhu et al., ICCV 2017, arXiv 1703.10025): warp past-frame features to the current
frame and fuse them with learned weights. Three later developments replace its parts:

| part | FGFA 2017 | here | descends from |
|---|---|---|---|
| motion | optical flow (fails through foliage) | **tracker / box motion as the coarse warp** | flow-guided deformable alignment, BasicVSR++ (Chan et al., CVPR 2022, arXiv 2104.13371) |
| fine alignment | none | **learned residual offsets** around the coarse warp (non-rigid: gait) | deformable alignment, EDVR (Wang et al., CVPRW 2019, arXiv 1905.02716) and TDAN (Tian et al., CVPR 2020; arXiv id not verified) |
| fusion | cosine-similarity weights | **deformable attention**: K sampled points per past frame, softmax weights over (frame, point) | Deformable DETR (Zhu et al., ICLR 2021, arXiv 2010.04159) |
| layers | one | **two: static (zero motion) and dynamic (object motion)** | layered video decomposition, Omnimatte (Lu et al., CVPR 2021, arXiv 2105.06993) and Layered Neural Atlases (Kasten et al., SIGGRAPH Asia 2021, arXiv 2109.11418). Those are per-video optimisations; ours is feed-forward |

**Operator, per level ℓ ∈ {s8, s4}, per layer L ∈ {bg, obj}:**

  F̂_L(x) = Σ_{i ∈ history} Σ_{k=1..K} a_{L,i,k}(x) · F_i(p_{L,i}(x) + δ_{L,i,k}(x))

- F_i: frozen SAM2.1 b+ features of frame i at level ℓ (the image predictor's
  `_features['high_res_feats']`, `sam2_image_predictor.py:127`). The current frame t is included in the history.
- p_{L,i}(x): coarse reference point.
  - L = bg: p = x.
  - L = obj: p = A_{i→t}⁻¹(x), with A the similarity transform (translation + scale) mapping the person's box in
    frame i onto frame t.
  - Training: GT silhouette boxes, or the GPJATK MoCap pelvis (`../master_thesis/src/registration.py`), plus jitter
    (§2).
  - Test: the tracker's boxes.
- δ_{L,i,k}(x): learned offsets.
- a_{L,i,k}(x): learned weights, softmax over (i, k).
- Both are predicted by a small conv head from [F_t(x), F_i(p(x)), layer embedding, level embedding,
  age (t − i)].
- Grid-sample with bilinear interpolation.
- **Shared weights for both layers and both levels** (layer and level embeddings). Separate networks are an
  ablation (§5).
- **Coarse-to-fine:** offsets at s8 are upsampled ×2 and added as initial offsets at s4, as in EDVR's PCD pyramid.
- Sizes: K = 4 points, history N = 7 frames (t − 6 … t), hidden width 64.
  - Expected: ≈ 0.5–1 M parameters (assumed; count and report it).
  - Encoder frozen.
- **Composite head input also gets F_bg**, so it can copy background context at the person's boundary instead of
  predicting it.

**Background, first version: not learned** (user's decision, 2026-10-15).
- F_bg = Enc(pixel median of the warm-up frames), the same protocol as NFO.
- The learned bg layer (same operator, p = x) is phase C: only once inputs contain sway or illumination change, or a
  person passes during the estimate.

**Recurrent variant (ablation only).** Replace the window with a per-layer ConvGRU state, updated each frame.
- This is the O(1)-per-frame form derived in `nfo_vos_pilot_findings.md` (keyframe section).
- Ancestor: BasicVSR++ propagation.

## 2. Data

**Generator.** `benchmark/nfo_vos/synth/gpjatk_foliage.make_layered_clip(mode='perlin')`.
- Occluder ≡ background (occluded person pixels show B).
- Clump 24 px; ρ ∈ U[0.35, 0.55] per clip.
- B = random crop of the foliage photo: scale 960–2667 px, h-flip.
- Person planted with a σ = 1 feathered silhouette.
- Clip length 40 frames.

**Splits by subject** (GPJATK has p1–p32):
- train p1–p20;
- val p21–p26;
- test p27–p32 (`HOLDOUT` in the generator, untouched until the end).
- All sequences and all four cameras, ≈ 4 clips per sequence with different crops and seeds.
- Generate on the fly (0.5 s per clip, measured).

**Input nuisance**, applied to the frames only, never to the targets. Calibrated to NFO warm-up residuals
(measured 2026-10-15; robust std of frame − median over warm-up frames):

| seq | std | per-frame global brightness spread |
|---|---|---|
| seq1 | 2.97 | 0.19 |
| seq2 | 11.86 | 0.48 |
| seq3 | 5.93 | 0.50 |
| seq4 | 4.45 | 0.16 |

Sample per clip:
- additive Gaussian noise σ ∈ U[3, 12];
- per-frame gain ×(1 + U[−0.01, 0.01]);
- foliage sway: B and its mask translated by ≤ 1 px per frame, sinusoidally;
- JPEG quality U[70, 95].

NFO's seq2 residual is probably mostly sway, not sensor noise. Check this before trusting σ = 12 as noise.

**Alignment jitter** (training): per frame, box centre ± N(0, σ_a) with σ_a ∈ U[0, 6] px, and scale × (1 + N(0, 0.03)).

**Encoding.** Run the frozen encoder on the fly: (N + 1) noisy frames + 1 clean frame per sample, at about 0.074 s
per encode (measured).
- If training is encoder-bound, cache s4/s8 feature **crops** around the person's trajectory box instead of full
  maps. Full maps take ≈ 6 MB per frame at fp16, too much for disk.

## 3. Losses

**Composite.**

  L_obj = Σ_x w(x) · (1 − cos(F̂_obj(x), Enc(clean_t)(x))) / Σ_x w(x),   per level, summed over s8 and s4

- Target: SAM2 features of the clean current frame (person planted on B, no occluder, no noise).
- Weight w(x):
  - w = person fraction of cell x, inside dilate(S_t, 1 cell) ∧ O_t;
  - w = 0 elsewhere.
- O_t = pixels of S_t visible in at least one *aligned* history frame, using GT alignment. A body part hidden in
  every input frame has an unobservable current pose; asking for it teaches hallucination.
- Rejected target: motion-blurred clean features. They bake in gait blur.

**Auxiliary visibility head** (tiny, 1×1 conv on [F_t, F̂_obj, F_bg, s] at s4). BCE against V_t inside
dilate(S_t). It is trained jointly but **reported separately**: the gates (§4) use the parameter-free score s, so
that a strong head cannot hide weak features.

**Optimiser.** AdamW, lr 3e-4, cosine schedule, 20 k iterations, batch 4 clips × 1 target frame. Seed fixed and
recorded. 3 seeds for the final numbers.

## 4. Evaluation and pre-registered gates (val subjects; test only at the end)

All AUCs are visible (V) vs occluded person pixels (S ∖ V), by fragment-size bin in **true SAM2 cells**, as in
`synth/oracle_visibility.py`. Inputs are noisy and alignment comes from jittered boxes (σ_a = 3 px) unless stated.

**Baselines** (same frames, same score s):
- **B-none:** F_obj := F_t, the do-nothing control. Then s = 1 − cos(F_t, F_bg): background change only.
- **B-hand:** F_obj := Enc(hand composite, `factorised_gtfree.composite`) with the same box alignment. This is
  today's method in feature space.
- **B-warp:** FGFA-style fixed age-weighted average of box-warped features, no learned offsets or weights. Isolates
  the learned part.
- **Oracle:** F_obj = Enc(clean_t). Upper bound: 0.939 (s8) and 0.969 (s4) on sub-cell fragments, noise-free.
  Re-measure it with noisy inputs.

**Gates:**
- **G1, the decisive one.** Learned s at s4, sub-cell AUC:
  - ≥ 0.80;
  - and > B-hand and > B-warp, with paired clip-bootstrap 95% CIs excluding 0.
  - Kill: < 0.70, or not above B-none.
- **G2, robustness.** G1's AUC at σ_a ∈ {0, 3, 6, 10} px and with boxes from a real tracker (SAMURAI's own box on
  the synthetic clips). Report the slope. Pass: the AUC at the tracker's boxes is ≥ 0.75.
- **G3, feature fidelity** (diagnostic, not a gate). Mean weighted cos(F̂_obj, Enc(clean_t)) for learned vs B-hand
  vs B-warp.
- **G4, NFO transfer** (report only; NFO has no amodal truth).
  - Frames: NFO seq2–4 tier-A frames with handAv labels, tracker alignment, pixel-median F_bg.
  - AUC of s for labelled person vs non-person pixels inside the dilated label box, void excluded.
  - Compare with B-none and B-hand.
  - Tier B stays untouched.

## 5. Ablations (one variable per row, same data, same budget, 3 seeds)

| row | change vs the full model |
|---|---|
| 1 | no coarse warp (p = x for obj too) |
| 2 | coarse warp, no learned offsets (δ = 0) |
| 3 | coarse warp + offsets, fixed age weights instead of learned a |
| 4 | s8 only / s4 only |
| 5 | separate networks for bg and obj (only once the bg layer is learned) |
| 6 | recurrent ConvGRU instead of the window |
| 7 | history N ∈ {4, 7, 12} |

Report per row: parameters, FLOPs per frame, ms per frame, peak GPU memory, and the G1 AUC.

## 6. Steps for the implementing session (each gated on the previous)

1. `benchmark/nfo_vos/synth/noise.py`: the nuisance model of §2. Montage of clean vs noisy frames into
   `images/benchmark/gpjatk_foliage/noise.png`; the user judges it.
2. `benchmark/nfo_vos/layers/dataset.py`: on-the-fly clips, splits, jittered boxes, targets (clean frame, S, V, O_t).
   Unit check: O_t ⊇ V_t and O_t ⊆ S_t on 10 clips.
3. Baselines B-none, B-hand, B-warp and the noisy oracle, on val: the G1 table *before any training*. If even the
   noisy oracle is < 0.80 on sub-cell fragments, revisit the bar with the user before training.
4. `benchmark/nfo_vos/layers/model.py` + `train.py`: §1 and §3. Overfit test first: 1 clip, loss → ~0, s AUC →
   oracle.
5. Full training, 1 seed, then G1. On a pass: 3 seeds, then G2, G3, G4, then the ablations.
6. Write results into `docs/vos_benchmark/nfo_vos_pilot_findings.md` (constructive) and
   `docs/vos_benchmark/nfo_failure_log.md` (one line per dead end).

**Not in this plan:** injecting s or the layers into SAM2 (dense prompt, high-res residual, attention bias, extra
tokens). That is the next plan, after G1 passes.

## 7. Risks

- **Synthetic-only training may learn this one photo's texture.** Watch G4. If transfer fails, add more foliage
  photos. Candidate source: PercepTreeV1 (norlab-ulaval); check the licence first.
- **Offsets may learn to "find the person" anywhere,** bypassing the tracker. Mitigations: bound |δ| ≤ 2 cells, and
  ablation row 1 shows whether the warp matters.
- **The clean-frame target encodes the background context** (large receptive fields). This is handled by F_bg as an
  input and by the w weighting.
- **Novelty pass:** deferred by the user (2026-10-15); do not run it unprompted.

## 8. Working rules (the user's standing preferences)

- Run with `../samurai/.venv/bin/python` (sam2); the depth model / transformers live in `~/miniconda3/envs/dvas`.
- Commit only code and docs, with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Check `git status` before
  every commit.
- Ask before dispatching subagents. Minimise turns.
- Images go into `images/...`. The user judges visual checks; never claim one "looks right".
- Calibrated language (measured / derived / assumed). One variable per comparison. Paired bootstrap CIs.
- On first mention in a message to the user, re-explain each label (B0, T1, F_bg, F_obj, s, G1, …).
- No weekend work. Nag after 19:00.

## 9. Steps 1–3 results (2026-10-15)

**Step 1, nuisance model** (`synth/noise.py`; montage `images/benchmark/gpjatk_foliage/noise.png`, passed by the user).
- Calibration on NFO warm-ups (`noise.calibrate`, measured): flat-region σ 1.5 / 4.8 / 3.2 / 0.3 (seq1–4); lag-1
  temporal correlation of the residual 0.83–0.91. The residual is mostly slow, correlated scene change, not white
  noise; the linear rigid-sway model (Var r = σ_n² + σ_u²|∇B|²) does not fit (intercepts 2.8–10.6 ≫ flat σ).
- **Change vs §2 (accepted by the user):** white noise σ ∈ U[1, 5] instead of U[3, 12]. Rigid sway ≤ 1 px, gain ±1 %,
  JPEG 70–95 as planned. Sway moves B and the matte; V follows; the clean target uses the unswayed B.

**Step 2, dataset** (`layers/dataset.py`). Val = 30 sequences × 4 cameras = 120 clips. Unit check on 10 clips:
O_t ⊇ V_t and O_t ⊆ S_t hold on all 120 targets; mean |V|/|S| 0.59, |O|/|S| 0.82.

**Step 3, G1 baselines before training** (`layers/g1_baselines.py`, val, σ_a = 3 px, measured). s4 sub-cell AUC:
none 0.872 [0.860, 0.884] · hand 0.724 · warp 0.821 · warp without t 0.743 · noisy oracle 0.962 [0.956, 0.966] ·
oracle without photometric noise 0.966. Every non-learned F_obj is *below* the do-nothing control (hand −0.148,
warp −0.051, paired CIs exclude 0). Full table: `results/benchmark/rr/layers_g1/g1_table.txt`.
- Mechanism (derived, small-deviation expansion of the cosines): with F_t = F_bg + Δ_t, F_obj = F_bg + Δ_p,
  s ≈ (⟨Δ_t, Δ_p⟩ − ½‖Δ_p‖²)/‖F‖². B-none is an energy detector (s = ½‖Δ_t‖²); a template is a matched filter that
  rejects non-person change ε on occluded pixels but costs ½‖e‖² (template error: gait blur, jitter) on every visible
  pixel. It helps only when ‖ε‖² > ‖e‖². The synthetic clips have small ε, so B-none wins. Camouflaged pixels
  (Δ_p ≈ 0) are out of reach for every arm.
- **Proposed amendment to G1 (pending the user):** drop the absolute ≥ 0.80 clause, which B-none already passes; the
  bar is learned > B-none (0.872) with a paired CI excluding 0; headroom to the noisy oracle ≈ 0.09.

**Nuisance arms, pre-registered before running** (one variable each vs the step-3 'base', same 120 clip specs):
- `flutter`: non-rigid displacement field on B and matte (`noise.flutter_fields`, Gaussian σ 12 px, AR(1) ρ 0.97, RMS
  2.1 px). Calibrated to seq2's residual std (synthetic 11.5 vs 11.96); lag-1 reaches only 0.76 vs 0.87 (saturates
  ≈ 0.79 for any ρ at matching std: the textured photo responds nonlinearly to 1–2 px displacements).
  Prediction (derived): B-none drops; the noisy oracle drops less; the oracle − none gap widens. If B-none does not
  drop, the composite has no job even under NFO-like nuisance: reconsider the learned-F_obj route before training.
- `chunks`: occluder = perlin clump 12 ∪ 24 (each at density 1 − √(1 − ρ)). Prediction: all arms drop together
  (smaller fragments shrink ‖Δ_t‖² for every arm); the oracle − none gap roughly unchanged.
- Smoke test (1 clip, not evidence): base none/oracle 0.81/0.93; flutter 0.68/0.78; chunks 0.75/0.84. The clean
  target uses the unswayed B, so under flutter the oracle's context is ≈ 2 px off as well. Check whether the full
  run's oracle drop is mostly that.
- Run: `sbatch benchmark/nfo_vos/layers/g1_baselines.sbatch` (array 0 = flutter, 1 = chunks).
