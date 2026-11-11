# Next direction: Layer-SAM on moving, foliage-occluded apes (PanAf500), MSc-sized (6 months)

Status 2026-11-11. This doc replaces the separate PanAf session. Earlier PanAf docs from that session are in this folder:
`sam2_action_recognition_design.md` (the recognition design) and `lit_pass_2026-11-11_sam2_occlusion_behaviour.md`
(related work). Tools are in `benchmark/panaf/`. The data stays read-only in `../PanAf20K/PanAf500` (`PANAF_ROOT`).

## Goal

Does the NFO method generalise from people crossing behind foliage to apes moving behind vegetation, and do its masks help
downstream behaviour recognition? Both are judged on the subset of videos where the method can apply.

- **Method.** Layer-SAM: frozen SAM2.1 + integral image + adapter.
- **Applicable subset.** The ape moves while fragmented by vegetation.
- **Not the goal.** All 500 videos. On static or unoccluded apes the method is expected to add nothing, and must not hurt.

## Established (numbers, all measured)

### NFO, seq2–4 held-out, whole crossings, Layer-SAM vs SAMURAI (findings §4)

| method | box AUC | hand J | recognition conf | AP50 |
|---|---|---|---|---|
| Layer-SAM (3 seeds) | 0.717 / 0.724 / 0.718 | 0.770 / 0.771 / 0.763 | 0.689 / 0.696 / 0.698 | 0.916 / 0.916 / 0.917 |
| SAMURAI | 0.615 | 0.741 | 0.687 | 0.904 |

### Gait from masks (`docs/vos_benchmark/2026-11-11-gait-from-masks-design.md`)

Classifier trained on KTH, applied to NFO seq2–4, k = 16 frames:

| input | accuracy |
|---|---|
| masks (ours) | 0.774 |
| fair box control | 0.612 |
| SAMURAI masks | 0.753 |

- Masks beat boxes by +0.16 [−0.00, +0.33]: masks carry action information beyond the box.
- Ours − SAMURAI: +0.021 [−0.001, +0.048], so better masks moved recognition only slightly.

### PanAf500 motion audit (`benchmark/panaf/motion_stats.py`)

- **Sitting and standing apes are static.** Median box speed is 0.0011 and 0.0012 body heights/frame; 1–2% of frames
  exceed 0.02.
- **Labels follow motion.** The top transitions are standing↔walking, 189 / 185.
- **Static videos.** 140 / 500 videos are ≥ 95% sitting/standing.
- **Scope.** The integral image needs the ape to move relative to the occluder. A static ape behind static leaves
  exposes nothing new (derived).

### Pre-filter (`benchmark/panaf/filter_moving.py`)

- **Criterion.** Moving frames are box translation + scale change > 0.01 per frame AND a moving-class label; a track
  qualifies with ≥ 48 such frames.
- **Result.** 276 candidate videos: train 217, validation 16, test 43.
- **Triage material.** Contact sheets in `images/panaf/triage/`; the list is `results/panaf/candidates.csv`, whose
  `occluded` column is left for the human triage.

## Work packages

The decisive test of each package comes first. The order is binding.

**WP0, the subset (week 1).** The human triage of the 276 candidates for "fragmented by vegetation: yes/no" is the
occlusion label. It is the only label independent of every compared method. Freeze the list before any method runs on it.
Report the full test split alongside, for no-harm.

**WP1, headroom: kill test, weeks 2–3.**
- Arms on the subset: SAM2.1 video (B0), SAMURAI (T1), and Layer-SAM zero-shot (NFO-trained adapter, human data only).
- Metric: box AUC against the GT boxes.
- Kill if T1 is near ceiling, or if ours ≈ T1 (masks barely differ).
- Two earlier transfers died here: CDnet ("SAM2 too good") and MVOT (no distractor headroom).

**WP2, generalise the motion model (weeks 3–6).**
- Replace the x-only Theil–Sen alignment and tracker with 2-D translation + scale, fitted robustly to the box track.
  SAMURAI's box is the first motion source.
- Background: masked temporal median, excluding ape pixels from SAM2 + GT-box masks.
- Non-rigid alignment (point tracks such as CoTracker) only if an oracle alignment shows the gain.

**WP3, the learned gate (weeks 4–10).** The adapter is used only where it beats SAM2/SAMURAI; see the gate section below.

**WP4, adapter for apes (only if WP1 shows headroom that zero-shot misses).** Two routes:
- synthetic ape sprites (exact visible-mask GT) behind several occluder models;
- a box-supervised fine-tune with the change gate as negative labels: a static-occluder pixel equals the background B.

Reviewer objection to answer: "the gain is domain adaptation". The control is the same fine-tune without the integral
image.

**WP5, downstream recognition (months 3–6).** Follows `sam2_action_recognition_design.md` §4A: frozen SAM2 features
pooled in the mask, plus context and geometry, into a small temporal head.
- Arms: box crop (the control), SAMURAI masks, ours.
- Metrics: top-1 and C-Avg on the subset and on the full test split.
- External baseline: V-JEPA with domain-adaptive pretraining (Mueller et al., arXiv 2509.12193; numbers unverified).

**Out of scope.** Amodal completion (Diffusion-VAS failed on NFO), SAM2 memory variants (closed), PanAf20K video-level
labels, new recognition backbones.

## The learned gate (WP3), design

**Experts.**
- E0: SAM2/SAMURAI output, adapter off.
- E1: adapter on.

**Oracle bound first.** Per frame, keep the better of E0 and E1 according to the GT. That score is the most any gate
can reach. If it is ≈ always-on E1, do not build a gate.

**Gate features.** The decoder runs twice, which is cheap.
- From SAM2's own heads: `pred_iou` (IoU head) and `obj_score` (occlusion head) of E0 and E1.
- Their disagreement IoU(E0, E1).
- Evidence mass Σa.
- Fragmentation of the change map inside the box.

**Labels.** y = 1 if J(E1) > J(E0) + margin. They come from clips with exact visible-mask GT:
- GPJATK synthetic;
- **KTH + synthetic foliage** (SAM2 GT-box silhouette ∧ ¬occluder).

The clips span a density sweep ρ ∈ [0, 0.7], including clean ones.

**Model.**
- First a logistic regression on these features, which is interpretable and has few parameters.
- An MLP only if needed.
- Baseline: a single threshold on `pred_iou`. The literature reports that learned routers often fail to beat a simple
  quality threshold (arXiv 2606.14476; arXiv 2605.06339).

**Check before relying on SAM2's heads.** Reliability diagrams of `pred_iou` and `obj_score` against the true J under
foliage. No calibration study of these heads was found (lit pass).

**Success criteria.**
- The gated J tracks max(E0, E1) across ρ.
- No harm on clean clips.
- It holds on NFO hand frames and the PanAf subset.

### WP3, first result (2026-11-11, synthetic, measured; `benchmark/nfo_vos/eval/dual_expert.py`)

Setup: 24 held-out GPJATK clips × ρ ∈ {0, .15, .3, .45, .6}, 40 frames each. E0 = SAM2.1 video with memory (B0 prompt
protocol, laptop fork with samurai_mode off); E1 = Layer-SAM seed 0. Both share one frame-encoder pass. Metric: J vs the
exact visible mask. No CIs yet (clip bootstrap still to add).

| ρ | E0 | E1 | oracle | LR gate |
|---|---|---|---|---|
| 0 | 0.809 | 0.822 | 0.847 | 0.840 |
| 0.3 | 0.638 | 0.649 | 0.674 | 0.665 |
| 0.6 | 0.271 | 0.361 | 0.377 | 0.362 |
| all | 0.593 | 0.623 | 0.646 | 0.635 |

- **No mean damage on clean clips.** E1 ≥ E0 at every ρ, including ρ = 0 (+0.013). On 9–15% of frames, E1 is worse by
  more than 0.05.
- **Gate headroom is small.** The oracle beats E1 by +0.023. The learned gate recovers about half of that (+0.012).
- **Signals, AUROC for "E1 better"** (inverse values mean a low signal predicts E1):
  - E0's own heads are informative: obj0 0.22, iou0 0.23.
  - E0/E1 agreement is the strongest signal: 0.18. Disagreement means E0 failed.
  - Change-map fragmentation (0.49), gate fraction (0.47) and evidence mass (0.47) carry no information.

**NFO (61 windows, outcome box IoU vs GT box; measured).**
- Means: E0 0.637, E1 0.710, oracle 0.737, LR gate 0.716. On the 38 hand frames: J E0 0.660, E1 0.676.
- **The signal ranking flips relative to synthetic.** E0's heads are uninformative (obj0 0.49, iou0 0.53). The
  adapter-side signals carry the information: evidence mass 0.69, iou1 0.63, gate fraction 0.60. Agreement also
  reverses, to 0.60.
- **A gate fitted on synthetic clips would therefore not transfer.** Caveat: the outcome metric differs between the two
  sources (J vs box IoU).

**SAM2's occlusion head (object-score logit) under fragmented occlusion.**
- It never declares the object absent: 0 negative logits across 4,680 synthetic and 2,440 NFO frames.
- On synthetic clips its median falls from 9.3 (ρ = 0) to 6.0 (ρ = 0.6). Within one density it separates E0's
  failures (J0 < 0.5) only at low ρ: AUROC 0.99 at ρ = 0 (failures there are rare), 0.55 at ρ = 0.6.
- It is a full-disappearance detector, not a fragmentation detector.

**Real-data gate and KTH no-harm (2026-11-11, measured; `dual_expert real`).**

NFO, gate fitted on seq1 and applied to seq2–4, box IoU, CIs from a window bootstrap:

| | box IoU |
|---|---|
| E0 | 0.651 |
| E1 | 0.733 |
| gated | 0.733 |
| oracle | 0.748 |

E1 − E0 = +0.082 [+0.065, +0.101]; gated − E1 = +0.001 [−0.002, +0.003]. On real foliage the gate is useless.

KTH (60 unoccluded crossings, box IoU vs KTH GT boxes):
- E0 0.937, E1 0.883. **E1 − E0 = −0.053 [−0.071, −0.040]**; E1 is worse by more than 0.05 on 28% of frames. So the
  adapter-on pipeline harms clean real video.
- The gate fitted on NFO keeps E1 on 89% of KTH frames and recovers only a third: −0.034 [−0.043, −0.026] vs E0.
  NFO contains no clean frames to learn "switch off" from.
- The clean regime shows in the signals (median NFO / KTH): gate fraction in box 0.24 / 0.47, evidence mass 0.27 / 0.46,
  agreement 0.83 / 0.95. A gate trained on NFO plus clean real video could learn it.

Open before concluding "the adapter harms":
- E1 includes the per-frame tracker prompts. The `trk` arm on KTH (net = None) separates the adapter from the tracker.
- KTH is 160×120, which is out of distribution.
- Shadows pass the change gate.

**The mechanism (2026-11-11, measured, held-out on both sides).**

A logistic gate on three logged signals, trained on NFO seq1 plus the odd-numbered KTH subjects (`class_weight=balanced`):

  Layer-SAM on ⇔ σ(+0.89·iou1 − 0.86·gate_box − 0.57·agree + c) > 0.5   (standardised features)

The rule reads directly: switch Layer-SAM on when the adapter's own IoU head is confident, the change map inside the box
is sparse (fragmented visibility), and E0 and E1 disagree.

| held-out set | E0 | E1 | gated | Layer-SAM on | gated − better fixed choice |
|---|---|---|---|---|---|
| NFO seq2–4 | 0.651 | 0.733 | 0.728 | 86% | −0.005 [−0.008, −0.003] |
| KTH even subjects | 0.932 | 0.873 | 0.919 | 34% | −0.013 [−0.021, −0.006] |

It keeps about 94% of the adapter's NFO gain ((0.728 − 0.651) / (0.733 − 0.651)) and recovers about 78% of its KTH loss
((0.919 − 0.873) / (0.932 − 0.873)). For PanAf: check the same three signals before training anything new.

**Superseded by: a causal switch with ONE model per frame (2026-11-11, measured).** The two-mask gate above is not
deployable, because it computes both masks.

**Policy.**
- Start in SAM2 video mode.
- After each frame, a logistic switch decides the mode for the next frame, using only:
  - the active model's own heads: SAM2 mode `obj0, iou0, fill0`; Layer-SAM mode `iou1, obj1`;
  - change-map statistics inside the tracker box (`gate_box, n_comp, largest, mass_box`). These are pixel operations on
    frame, empty scene and tracker, with no extra network pass.
- Trained on NFO seq1 plus odd KTH subjects.

| held-out set | SAM2 | Layer-SAM | switched | Layer-SAM on | switched − better fixed |
|---|---|---|---|---|---|
| NFO seq2–4 | 0.651 | 0.733 | 0.733 | 97% | +0.001 [−0.000, +0.001] |
| KTH even subjects | 0.932 | 0.873 | 0.932 | 0% | +0.000 |

**Weights.**
- **SAM2 mode → switch on:** many change-map components in the box (`n_comp` +0.51, fragmented visibility) and a low
  SAM2 object score (−0.39).
- **Layer-SAM mode → stay on:** its own IoU head (+0.90). **Switch off:** one dominant change component (`largest` −0.64,
  i.e. the person is fully visible).

**Caveat.** Simulated from logs in which SAM2 ran continuously, so its memory after switching back is idealised. On the
held-out sets the policy barely switches within a clip, so the effect is small. A deployment re-prompts SAM2 from
Layer-SAM's last box when switching back.

**Deployable policy, run for real (2026-11-11, measured; `dual_expert policy`, `results/benchmark/gate/policy.csv`).**
One model per frame. Start in SAM2. Switch back = re-prompt SAM2 from the tracker box at that frame (option c, one
memory-encoder call).

| held-out set | always-SAM2 | always-Layer-SAM | policy | Layer-SAM on | policy − better fixed |
|---|---|---|---|---|---|
| NFO seq2–4 (46 windows) | 0.651 | 0.733 | 0.733 | 97% | +0.000 [−0.001, +0.002] |
| KTH even subjects (32 crossings) | 0.932 | 0.873 | 0.932 | 0% | +0.000 |

- On NFO the policy switches on after the first frame and stays on.
- On KTH it never switches on.
- The switch-back path (c) was barely exercised; it needs clips that alternate between occluded and clear (PanAf).
