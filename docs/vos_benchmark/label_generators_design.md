# Label generators under fragmented occlusion: appearance propagation vs shape prior — design (2026-10-13, DRAFT)

Status: draft for sign-off (hypotheses, boundaries, pass/fail). Nothing below has been run unless marked.

## Context

The NFO VOS benchmark needs labels. Two generators exist:

- **S (appearance propagation, the SAM2-ceiling).** SAM2 *video* predictor, prompted with GT box + centre point at
  every confirmed-clear checkpoint, propagated forward and backward up to the neighbouring checkpoint. Then:
  union, clip to the dilated GT box, and removal of components narrower than min_width_px
  (gen_nfo_pseudo_masks.py:230–260).
- **L (shape prior, the silhouette label).** Retrieve and blend clear-frame silhouettes, fitted to the GT box
  (extent_label_test.py).

Measured against the annotator's polygons on 8 held-out frames: S has P 0.920 / R 0.682; L∪S has P 0.820 /
R 0.775 at θ = 0.2. Against the semi-modal brush masks (20 frames, seg 4), S has P 0.766 / R 0.764.

Known terminology: a *modal* mask is the visible part, an *amodal* mask the full extent. Asked to outline "the
person" behind occluders, humans produce near-amodal outlines consistently: Zhu, Tian, Metaxas & Dollár,
"Semantic Amodal Segmentation", CVPR 2017. So "SAM2 is careful, humans are lazy" is, in standard terms, *SAM2 is
modal and human outlines are near-amodal*. That half is expected from SAM2's training data (visible-part masks)
and is not a finding by itself.

## Phase 0 — cheapest kill (run first, minutes, existing data)

H2 below needs only the 20 brush frames and SAM2-ceiling. Run it before any new generator code.

**Confound to remove first:** S's post-processing deletes thin components. If S misses small fragments, the cause
may be our filter, not SAM2. So H2 is computed on **raw** SAM2 propagation (pre-filter). That needs one re-run of
the seg-4 checkpoints with the filter off (GPU, ~minutes), or the stored raw output if one exists.

## Phase 1 — mechanism

Let S_t be the full silhouette, O the occluder, and V_t = S_t ∧ ¬O the visible part. The human polygon Ĥ_t
approximates S_t smoothed at the annotator's scale.

- **S (appearance).** SAM2 decodes at stride 16. The 1024² input is stretched from 800×600, so one cell is
  12.5 × 9.4 native px ≈ 117 px² (derived). A visible fragment f of area a_f is recovered with probability
  p(a_f), increasing in a_f relative to one cell: a fragment much smaller than a cell shares its cell token
  with foliage. Recall vs the polygon is then R_S ≈ v · r̄, where v = |V|/|S| is the visibility and r̄ is the
  area-weighted fragment recall. Precision stays high, because what is decoded is visible person.
- **L (shape prior).** Recall depends on pose and shape mismatch, not on v: R_L ≈ const(v). Precision is
  bounded by the same mismatch.
- **Prediction.** R_S rises with v, and R_L is flat in v. There is a crossover v* below which L (or L∪S) covers
  more than S.

## Phase 2 — hypotheses

**H1 (recall vs visibility).**
- HYPOTHESIS: against polygon labels, S's recall increases with visibility v, while L's recall does not.
- MECHANISM: R_S ≈ v · r̄ and R_L ≈ const (Phase 1).
- BOUNDARY: not expected for fully visible frames (v ≈ 1, where both are near ceiling), nor for running segments
  (larger pose variance, so L degrades).
- KILL: the slope of R_S on v is not positive (bootstrap CI includes 0), or the slope of R_L is at least as large
  as R_S's.

**H2 (SAM2 drops sub-cell visible fragments): the candidate finding about SAM2 itself.**
- HYPOTHESIS: against the brush (modal) masks, the fraction of each visible connected component recovered by
  raw S increases with component area. Components smaller than about one stride-16 cell (~117 px²) are mostly
  missed.
- MECHANISM: Phase 1, fragment term.
- BOUNDARY: not expected for fragments attached to the main body blob, since connectivity lets the decoder
  extend into them. Analyse attached and detached components separately.
- KILL: recall is flat in component area (bootstrap CI of the slope includes 0), or detached sub-cell
  components have recall ≥ 0.5.

## Phase 3 — baseline and the one variable (equalised prior information)

All generators get the **same inputs**:
- frames;
- GT boxes for every frame;
- the same anchor frames: confirmed-clear frames. These are chosen geometrically, from curated clear regions
  and the GT box (nfo_visibility.py:116), with no SAM2 involved;
- the same anchor masks: SAM2 at the anchors (S's own checkpoint output);
- both bidirectional (non-causal).

The variable is the **inference principle**, as a 2×2 design:

|                           | no shape prior | shape prior (anchor silhouettes) |
|---------------------------|----------------|----------------------------------|
| no per-frame appearance   | box only (B)   | L: retrieval by box aspect only   |
| per-frame appearance      | S (SAM2-ceiling)  | L|S ∪ S: retrieval conditioned on S, union (current) |

Depth cleaning and the empty-scene penalty are extra information neither S nor L has. They are a separate,
reported add-on, not part of the 2×2.

## Phase 4 — ground-truth audit

- Brush masks (20 frames, seg 4): semi-modal (thin branches ignored), one annotator. Usable as a modal
  reference for H2, with that caveat.
- Polygons (17 frames, segs 4/5/0): near-amodal outlines, one annotator. The reference for H1.
- **Blocking caveat: one annotator.** "Humans annotate extent" from n = 1 is an anecdote, and brush vs polygon
  (J 0.58–0.69) mixes style with retest. Recommendation: a second annotator draws polygons on 10 of the 17 frames
  (~20 min). Without that, H1 is stated against "this annotator's polygons" only, and the human-behaviour claim
  rests on the citation, not on our data.
- **v for H1** needs an independent visibility per frame. v from SAM2-ceiling area would be circular for S. Use
  v = |brush ∩ polygon| / |polygon| on frames that have both: 5 now (1164, 1183, 1214, 1224, 1263). Brush the 12
  remaining polygon frames for n = 17 (~1.5–2 h), or accept n = 5 as a pilot.

## Phase 5 — feasibility

Done today: L∪S runs and is scored (extent_label_test.py). Remaining: a raw-S re-run with the filter off for seg 4
(H2), and the box-only generator B (trivial).

## Phase 6 — pass/fail (to fix at sign-off)

- **H2 PASS:** logistic slope of "component recovered (≥ 50% of its pixels)" on log area > 0 with 95% bootstrap
  CI (over frames) excluding 0, AND detached components < 117 px² recovered at < 0.5.
- **H1 PASS:** slope of R_S on v > 0 with CI excluding 0, AND slope of R_L on v < half of R_S's.
- The label decision (which generator is the benchmark's label) is separate. It stays with the pre-registered
  2026-10-13 test (L∪S passed F₂) plus the ranking-robustness check.

## Phase 7 — scope-cutting order (cut first → last)

1. Running segments (boundary of H1 anyway).
2. The box-only arm B.
3. Brushing all 12 extra frames (fall back to n = 5 for H1, as a pilot).
4. The second annotator (fall back to the citation; weakens the human-behaviour claim).
5. Never cut: H2 on raw S, and the ranking-robustness check of B0/T1 under both labels.

## Execution order (each gated on the previous)

1. Raw S re-run on seg 4 (filter off) → H2 component analysis. If H2 is killed, the "SAM2 behaviour" finding
   reduces to "SAM2 is modal" (expected) and this study becomes a paragraph, not a section.
2. Ranking robustness: B0, T1, T1-oracle-init rescored under S, L∪S and the hand frames. This is the benchmark's
   decisive question, independent of 1.
3. H1: only if 1 passes, with whatever v sample exists.

## Phase 8 — adversarial notes

- Strongest objection: the label study is about our *label generators*, not about SVOS methods. A claim
  about "how SVOS methods behave under fragmented occlusion" needs the methods themselves (B0, T1, …) scored
  against the human references. Step 2 provides exactly that. The generator comparison can only motivate it.
- S is a non-causal, multi-anchor, box-gated *oracle-style* SAM2. Its behaviour bounds causal SAM2 from
  above; it does not equal it.
- Pattern watch: this study rhymes with "breadth over follow-through". Step 2 (ranking robustness) was the
  named next decisive test and is still unrun.
