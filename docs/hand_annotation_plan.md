# Hand masks for one NFO segment, and auto-completing the others — plan (2026-10-12)

## Why

Every number on NFO is scored against SAM2-made pseudo-GT. Two open questions decide what the numbers
mean:

1. **Does the pseudo-GT miss visible person (low recall)?** If it does, methods that add real person
   pixels are penalised.
2. **Is V̄ a better visible-part label?** V̄ is the motion-aligned average of the past 6 frames' pseudo-GT;
   it combines body parts seen in different frames, so it may be closer to extent than to "visible now".

Hand masks at native resolution answer both: compare pseudo-GT and V̄ against them, then rescore
B0/T1.

## Segment

**seq1 segment 4**: frames 1154–1308, right-to-left label "R", walk, backup half. 155 frames, 3 benchmark
windows (t₀ = 1154, 1194, 1234), **2 of them hard starts**.

Auto-completion targets on the same scene and path:
- seg 5 (1476–1621, L walk, the **opposite direction**);
- seg 0 (11–137, R walk, the same direction, the other half).

## Protocol (write it before opening a frame)

- **Person (1):** body, hair, clothing or carried object seen directly through a gap. **Shadows are
  background.**
- **Void (2):** a genuinely ambiguous pixel (leaf edge over the body, blur). Excluded from scoring;
  report the void fraction.
- **Edits go both ways:** you may erase pseudo-GT pixels, not only add. Store additions and removals as
  separate layers, so pseudo-GT precision = 1 − removed/pseudo and recall = kept/hand.
- **Frames:** every 5th frame plus every t₀, 33 frames.
- **Anchoring check:** annotate 3 frames **from scratch first**, before seeing their pseudo-GT, then the
  same 3 starting from the pseudo-GT. Their disagreement is the anchoring bias.
- **Test–retest:** redo 3 frames on another day for your own agreement. That agreement is the noise floor.
- **Tool:** CVAT, brush/eraser on the native frame. Import the native pseudo-GT as the starting mask
  (e.g. CVAT's segmentation-mask import format), with a separate `void` label. Use neighbouring frames,
  since motion is how fragments are seen.
- **SAM / SAM2 / SAM3 clicks: not for the evaluation segment.** Any assistant's biases enter the labels
  and favour methods that share them:
  - SAM clicks → SAM-like blobs, which favour B0/T1;
  - negative clicks placed from the background model → the empty-scene cue, which favours our own
    cue-based methods;
  - positive clicks inside the pseudo-GT → the pseudo-GT itself.
  Assisted labelling is fine for *auto-completion and training* labels, and only with its bias measured
  on the 3 from-scratch frames.
- **Save:** `data/nfo_processed/seq1_gt/<idx>_handmask.png` (0/1/2 at 800×600). Data, not code:
  gitignored, back it up.

## What it produces

- Pseudo-GT precision / recall / J vs hand, by visibility v and by distance to the edge.
- **V̄ vs hand**, in the same terms: settles whether V̄ is a better label or extent in disguise.
- B0, T1, T1-om rescored on segment 4 against hand vs pseudo-GT: does any difference or ranking
  change?

## Auto-completing other segments (same scene, static camera)

### Model 1 (first): a static occluder layer, the layered model with a fixed foreground

Camera and foliage are static, so the occluder in front of the walking path is (almost) the same in every
segment. With S_t the person's silhouette (full extent) at frame t and O the occluder matte (1 = occluder
in front):

  visible_t(x) = S_t(x) · (1 − O(x))

This is the layered representation of Wang & Adelson (IEEE TIP 1994), with the front layer known to be
static.

- **Estimate O from the hand segment.** Wherever the person's extent covered pixel x, the hand mask says
  "visible" (evidence O = 0) or "not visible" (evidence O = 1).

  Beta–Bernoulli per pixel: P(O(x) = 1) = (n_occ(x) + α) / (n_obs(x) + α + β).

  The posterior also captures wind (foliage that is sometimes in front), and n_obs shows where we have
  any evidence at all: only where the person passed.
- **Transfer.** For seg 5 / seg 0 frames: visible = Ŝ_t ∧ (P(O) < ½), restricted to pixels with evidence.
  Elsewhere fall back to the pseudo-GT.
- **The hard part is Ŝ_t.** It is the silhouette, not a box. Options:
  - SAM2 on the motion-aligned composite (the A/4 extent: +0.06–0.08 box IoU on NFO, amodal IoU ~0.75 on
    GPJATK; finally a use for it);
  - the hand-segment silhouettes shifted along the path, since the person walks at a similar height;
  - the union of pseudo-GT over the window.
- **Validation (cheap):** hand-annotate 5 frames of seg 5. Measure transfer vs pseudo-GT precision and
  recall against them. Bar: recall up without precision falling below the pseudo-GT's.

### Model 2 (if 1 is not enough): one-shot fine-tuning on the hand segment

Fine-tune a segmenter on the ~33 hand masks and apply it to the other segments: OSVOS (Caelles et al.,
CVPR 2017, arXiv 1611.05198) is the classic recipe of learning from one annotated example. With SAM2 frozen,
the adapter from the master_thesis plan can be fine-tuned instead. Inputs should include the occluder
prior Ô and the position, so it learns the scene's fixed structure. Few frames: evaluate on seg 5's 5 hand
frames only.

### Annotation tools that propagate a few hand frames

XMem++ (Bekuzarov et al., ICCV 2023, arXiv 2307.15958) segments a whole video from a few user-chosen
annotated frames held in a permanent memory, and suggests which frame to annotate next. A tool like this
could speed up annotating seg 4 itself. But it is a learned VOS model, so its output inherits
VOS-model errors. Use it to propose, never as the final label.

### Diffusion

A conditional generative model of masks ("sample a completion given frame, position, prior") is possible,
but it is data-hungry. With one segment of hand masks the static-occluder structure is a far stronger
prior than anything a diffusion model could learn from 33 masks. The closest diffusion work, pix2gestalt
(CVPR 2024, arXiv 2401.14398), synthesises amodal **wholes**: the hidden parts. That is the opposite of our
target (visible now). Not recommended at this stage.

## Auto-completion strategy (revised 2026-10-12, measured path overlap)

**Fact (measured):** every seq1 segment walks through the same pixels as segment 4. 91–99% of each
segment's box-swept area lies inside segment 4's, and 99–100% of each segment's visible (pseudo-GT) pixels
do. So one hand-labelled segment observes the whole occluder in front of the path, for the whole
sequence.

**Key idea:** "visible person" is the product of a **static** property and a **per-frame** property:
- static: is pixel x a **gap** in the foliage? Fixed for the sequence, learnable once from the hand
  masks.
- per-frame: is **something new** at x now? The frame differs from the empty scene.

Inside a gap the scene behind is static, so "something new" there is the person, or their shadow on the
ground. The cue's earlier failure, false positives on moving foliage, disappears once it is restricted
to gap pixels.

1. **Gap map Ŵ(x) from segment 4.** Ŵ(x) = n_visible(x) / n_box-covered(x) over the hand frames: how
   often x showed the person when the person passed. A Beta prior shrinks rarely covered pixels.
   - To generalise beyond observed pixels, train a tiny classifier on the **empty-scene image itself**
     (pixel colour/texture + SAM2 stride-16 background features), "gap vs leaf", with labels from Ŵ.
   - Then the gap map is defined everywhere, including pixels segment 4 never covered.
2. **Per-pixel labeller**, trained on segment 4's hand frames:
   - logistic regression or a small gradient-boosted classifier;
   - features: Ŵ(x), pixel z and feature z16 vs empty scene, the pseudo-GT, position in the GT box
     (relative y), distance to the pseudo-GT;
   - target: hand mask (void excluded).

   Few parameters, interpretable, and it learns e.g. that new pixels at the box bottom are shadow.
3. **Apply to the other segments.** GT boxes exist for every frame, so restrict to the box. Output: an
   auto label per frame.
4. **Validate before trusting:**
   - Leave-frames-out within segment 4 (train on 2/3, test on 1/3) gives an immediate estimate.
   - Then hand-label 5 frames of segment 5 (opposite direction) and 3 of segment 6 (run).
   - Bar: auto label J vs hand ≥ pseudo-GT J vs hand + 0.05, with precision no more than 0.02 below
     the pseudo-GT's.
5. **Human in the loop:** import the auto labels into CVAT and correct only the errors, which is
   much faster than labelling. Log correction time per frame.

**Circularity, stated up front:** the auto labels contain the empty-scene cue. They are fine for training
and transfer. But evaluating cue-based methods against *uncorrected* auto labels would favour them, so
benchmark verdicts use only hand-corrected frames.
