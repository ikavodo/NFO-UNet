# Poster plan: seeing people through foliage (2026-10-23)

Small conference. Emphasis: the problem and the ongoing work, not a results table. Audience: scientists outside
video segmentation, so no field jargon on the poster (see Vocabulary). Source of truth for numbers:
`docs/vos_benchmark/nfo_vos_pilot_findings.md`; the published briefing (`docs/briefing/`) is the long version.

**Takeaway under the title:** *A memory built from the model's own guesses compounds its mistakes, while evidence
taken straight from pixels can't.*

**Protocol (decided 2026-10-24): every number and plot on the poster comes from the whole crossings** (full-pass
protocol, findings §9: 32 clips, one per pass, from its earliest benchmark start to the end of the pass, 44–155
frames, every frame once). No 40-frame-window numbers on the poster.

## Punches (one sentence + one figure each; everything else on the poster supports these)

| # | sentence | figure | backed by |
|---|---|---|---|
| 0 | **Title:** Seeing people through foliage, without memory | — | — |
| 1 | Behind branches, a walking person is visible only in pieces. The task: mark exactly the visible pieces. | seq4 00293: frame, crop, human label (7 separate visible pieces) | hand label |
| 2 | Even people can't do this reliably: the annotator marked 79% of this body "unsure". | seq3 00287: crop + label, grey = unsure | hand label |
| 3 | Video models remember their own answers, so their caution sticks: over a crossing their masks end up about a fifth smaller than ours, and on a third of crossings they lose more than 30%. Ours keep their size. | `mask_fill_ratio_fullpass.png` (memory model / ours per frame, 95% band) | `eval/mask_fill.py`, NFO_RUN=fullpass, 32 crossings: from frame 30 to the end, SAM2.1 video ×0.76 [0.63, 0.89], SAMURAI ×0.78 [0.66, 0.92] our mask size; 11 / 10 of 32 crossings below ×0.7, 8 / 32 above ×1; ours +1% per 10 frames [−1%, +3%] |
| 4 | Branches stay still, the person moves: stacking the last frames, aligned to the person, blurs the foliage and builds up the person. | one row of `nfo_inputs.jpg`: frame, empty scene, integral image, our mask | method |
| 5 | We score what masks are for: following the person, and keeping them recognisable. | three icons | benchmark |
| 6 | Clearly better at following the person (0.68 vs 0.58), slightly better at marking the visible pixels, as good as the best video models at keeping the person recognisable, with no memory at all. | three result bars | full-pass job 1950229 |
| 7 | Every ingredient earns its place: the position hint buys tracking, the empty scene removes false pixels, the integral image adds real person evidence. | ablation bars | full-pass rescore with the -noC arms (pending) |
| 8 | Next: how far do two annotators agree? | — | open |

Takeaway line under the title (punch 3 in one sentence): *A memory built from the model's own guesses compounds
its mistakes, while evidence taken straight from pixels can't.*

Example frames chosen 2026-10-23: seq4 00293 (task), seq3 00287 (annotation). Contact sheets:
`images/benchmark/downstream/modal_candidates_seq{2,34}.jpg`.

## Structure: three columns, six panels

Panel titles are the questions a non-specialist would ask.

### 1. Can you see the person?
- Fixed camera; a person walks behind bare branches and shrubs; at any moment only fragments are visible
  (*fragmented occlusion*). Task: mark exactly the visible parts, not the hidden body.
- Figure: one raw NFO frame plus a crop with the visible fragments tinted (new; needs a native frame from the
  cluster, `images/nfo/nfo_*.jpg` carry the old U-Net heatmap).

### 2. Why this is hard, for people and for models
- **People.** Fragments are thinner than a brush stroke: annotators fill small gaps and need an "unsure" class; the
  box around the visible pixels and the hand-drawn person box overlap as little as IoU 0.25. Only 43 frames carry
  mask labels. How far two annotators agree is not yet known (panel 6).
- **Models.** State-of-the-art video segmenters (SAM2.1 video, SAMURAI) play safe: what they mark is nearly always
  person (precision 0.88–0.89) but they miss about a fifth of it (recall 0.80–0.81). They also remember their own
  past masks, so the caution sticks: over a crossing their masks end up about a fifth smaller than ours.
- Figures: `images/benchmark/downstream/hand_vs_gt_box.png` (2–3 tiles), `mask_fill_ratio_fullpass.png`.

| measured (`eval/mask_fill.py`, NFO_RUN=fullpass, 32 crossings; mask area / person box area, frame 30 → end, relative to frame 6) | memory model / ours | 95% CI (over passes) |
|---|---|---|
| SAM2.1 video | ×0.76 | [0.63, 0.89] |
| SAMURAI | ×0.78 | [0.66, 0.92] |

  Held-out seq2–4 alone: ×0.77 / ×0.79. Per crossing it varies: a third lose more than 30%, a quarter end bigger than
  ours. Caveat, not for the poster: area is not recall; the hand frames are too sparse to show that the lost pixels are
  person, though the precision-leaning profile suggests it.

### 3. A benchmark that measures what masks are for (contribution 1)
- Dense pixel labels are infeasible under fragmented occlusion, so every method is scored on three questions with
  real ground truth: are the pixels right (43 hand-labelled frames), does it follow the person (tracking against
  hand-drawn boxes), does the cut-out stay recognisable as a person (a standard person detector on the masked result).
- 4 sequences, 32 whole crossings (44–155 frames), one held-out test (seq2–4).
- Figure: three-icon row; no table.

### 4. Our idea: look through the foliage over time (contribution 2)
- Branches stay put, the person moves. Stack the last six frames shifted to follow the person, keep only what differs
  from the empty scene: foliage blurs out, the person adds up. Same idea as airborne optical sectioning for seeing
  through forests.
- A small trained add-on (0.7 M parameters) feeds that stack and the empty scene into an unchanged SAM2, one frame
  at a time, with no memory. Trained only on synthetic people behind synthetic foliage.
- Figures: simplified diagram (3 inputs → unchanged SAM2 → mask; new, redrawn from the briefing), one row of
  `docs/briefing/figs/nfo_inputs.jpg` (frame, empty scene, stack, our mask).

### 5. What each ingredient adds (the ablation)
- One variable at a time, worded as trade-offs:
  - prompting with the person's position every frame: better tracking, more false pixels;
  - the empty-scene reference: removes the false pixels;
  - the stacked past frames: real person evidence (a detector recognises the person more often), not just bigger masks.
- One headline line: tracking clearly better than SAMURAI, segmentation slightly better, recognition on par, at similar speed
  (61 vs 53 ms per frame), without memory.
- Figure: horizontal "adds / costs" bars instead of the ablation table (new).

Numbers (full-pass job 1950229, 2026-10-24, 32 crossings, seq1–4; recognition at N = 5; ours = mean of 3 seeds):

| question | ours | SAMURAI | SAM2.1 video | verdict |
|---|---|---|---|---|
| follows the person (box AUC) | 0.68 | 0.58 | 0.57 | better, +0.10, every seed's CI excludes 0 |
| visible pixels right (hand J, 19 labelled frames) | 0.76 | 0.73 | 0.73 | slightly better, +0.02 to +0.03, every seed's CI excludes 0 |
| cut-out recognisable (detector confidence, 5-frame stacks) | 0.67 | 0.66 | 0.67 | on par |

| ingredient (32 crossings) | tracking | recognition conf | hand J |
|---|---|---|---|
| position hint every frame (trk − SAMURAI) | +0.05 | −0.07 | −0.05 |
| empty-scene reference (adapter without integral image − trk) | pending | pending | pending |
| integral image (ours − without it, 3 seeds) | pending | pending | pending |

The two pending rows need the -noC arms on the 32 crossings (`pass_run.sh fullpass`, then `fullpass_score.sbatch`).
Their sum is known already: ours − trk = +0.047 tracking, +0.066 recognition conf, +0.072 hand J (seed 0).

### 6. Where this is going
- Second annotator: the human ceiling for segmentation; tells whether "on par" is already at that ceiling.
- Recognition with each method's own motion instead of the true motion: where the tracking lead should turn into
  a recognition lead.
- Does memory ever help? Tested on synthetic clips with exact labels.
- QR code to the briefing.

## Left out on purpose
Loss function and weights, synthetic-data table, per-seed CIs, metric names (J, AUC, AP50), the recognition-vs-stack-
length curve (the stacking picture carries the same point).

## Vocabulary
Replace rather than define. Visible-part segmentation (not *modal*), memory of its own past answers (not *memory
bank*), unchanged (not *frozen*), integral image, i.e. stack of aligned past frames (not *stack*); the recognition metric's frames are a "5-frame stack", position hint (not *prompt*).
A small glossary box for what cannot be avoided: IoU (overlap of two regions, 0–1), SAM2, precision / recall.

## Figures

| use | file | status |
|---|---|---|
| reuse, cropped | `docs/briefing/figs/nfo_inputs.jpg` (one row) | exists |
| reuse, 2–3 tiles | `images/benchmark/downstream/hand_vs_gt_box.png` | exists |
| reuse, one tile | `images/benchmark/downstream/stack_N7_vs_N21.jpg` or `labels_vs_integral_image.jpg` | exists; legend of the cyan / red outlines in the latter to confirm |
| new | `images/benchmark/downstream/mask_fill_trend.png` | done (`python -m benchmark.nfo_vos.eval.mask_fill`) |
| new | raw frame with tinted visible fragments | needs a native frame (cluster) |
| new | simplified method diagram | to draw |
| new | ablation "adds / costs" bars | to draw from the findings table |

## Open decisions
1. Format and size (A0 portrait or landscape; print PDF from a web page, `.pptx`, or LaTeX `tikzposter`).
2. QR target: the briefing artifact is shared with the organisation only; a public poster needs a public link.
