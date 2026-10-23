# Poster plan: seeing people through foliage (2026-10-23)

Small conference. Emphasis: the problem and the ongoing work, not a results table. Audience: scientists outside
video segmentation, so no field jargon on the poster (see Vocabulary). Source of truth for numbers:
`docs/vos_benchmark/nfo_vos_pilot_findings.md`; the published briefing (`docs/briefing/`) is the long version.

**Takeaway under the title:** *A memory built from the model's own guesses compounds its mistakes, while evidence
taken straight from pixels can't.*

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
  person (precision 0.89–0.90) but they miss about a fifth of it (recall 0.81–0.82). They also remember their own
  past masks, so the caution compounds: their masks shrink by about 10% every 10 frames, while ours stay the same size.
- Figures: `images/benchmark/downstream/hand_vs_gt_box.png` (2–3 tiles), `mask_fill_trend.png`.

| measured (`eval/mask_fill.py`, 61 clips, mask area / person box area, frames 6–39) | per 10 frames | 95% CI |
|---|---|---|
| SAM2.1 video (memory) | −10% | [−14%, −6%] |
| SAMURAI (memory) | −9% | [−12%, −4%] |
| ours (no memory, 3 seeds) | −1% | [−4%, +2%] |

  Same on the held-out seq2–4 alone (−10% / −8% / −1%). Caveat, not for the poster: area is not recall; the hand
  frames are too sparse to show that the lost pixels are person, though the precision-leaning profile suggests it.

### 3. A benchmark that measures what masks are for (contribution 1)
- Dense pixel labels are infeasible under fragmented occlusion, so every method is scored on three questions with
  real ground truth: are the pixels right (43 hand-labelled frames), does it follow the person (tracking against
  hand-drawn boxes), does the cut-out stay recognisable as a person (a standard person detector on the masked result).
- 4 sequences, 61 clips of 40 frames, one held-out test (seq2–4).
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
- One headline line: tracking clearly better than SAMURAI (+0.07 box AUC), segmentation on par, at similar speed
  (61 vs 53 ms per frame).
- Figure: horizontal "adds / costs" bars instead of the ablation table (new).

### 6. Where this is going
- Second annotator: the human ceiling for segmentation; tells whether "on par" is already at that ceiling.
- Recognition with each method's own motion instead of the true motion: where the tracking lead should turn into
  a recognition lead.
- Does memory ever help? Tested on synthetic clips with exact labels.
- Why stacks longer than ~7 frames stop helping (seams or misaligned limbs).
- QR code to the briefing.

## Left out on purpose
Loss function and weights, synthetic-data table, per-seed CIs, metric names (J, AUC, AP50), the recognition-vs-stack-
length curve (the stacking picture carries the same point).

## Vocabulary
Replace rather than define. Visible-part segmentation (not *modal*), memory of its own past answers (not *memory
bank*), unchanged (not *frozen*), stack of aligned past frames (not *composite*), position hint (not *prompt*).
A small glossary box for what cannot be avoided: IoU (overlap of two regions, 0–1), SAM2, precision / recall.

## Figures

| use | file | status |
|---|---|---|
| reuse, cropped | `docs/briefing/figs/nfo_inputs.jpg` (one row) | exists |
| reuse, 2–3 tiles | `images/benchmark/downstream/hand_vs_gt_box.png` | exists |
| reuse, one tile | `images/benchmark/downstream/composite_N7_vs_N21.jpg` or `labels_vs_composite.jpg` | exists; legend of the cyan / red outlines in the latter to confirm |
| new | `images/benchmark/downstream/mask_fill_trend.png` | done (`python -m benchmark.nfo_vos.eval.mask_fill`) |
| new | raw frame with tinted visible fragments | needs a native frame (cluster) |
| new | simplified method diagram | to draw |
| new | ablation "adds / costs" bars | to draw from the findings table |

## Open decisions
1. Format and size (A0 portrait or landscape; print PDF from a web page, `.pptx`, or LaTeX `tikzposter`).
2. QR target: the briefing artifact is shared with the organisation only; a public poster needs a public link.
