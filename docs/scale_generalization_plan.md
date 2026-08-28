# Scale generalization: what the experiments established, and what to build next

Authoritative summary. `deepsort_blob_scoring_compatibility.md` is the chronological log of
how these results were arrived at (including the wrong turns); this file is the standing
conclusion and the plan. If the two disagree, this one is newer.

All numbers below are hit@0.1 (reported position within 10% of the frame of the true
position, no-track counted as a miss) unless stated. Synthetic results are 853 windows per
scale bucket over 6 KTH `d1` sequences; NFO results are 3495 windows over 4 sequences.

---

## Part 1: What the experiments established

### F1. Absolute-pixel constants fail catastrophically, and asymmetrically

Same tracker, same footage, people rendered 60 / 120 / 240 px tall. One scale's constants
applied everywhere:

| person size | constants correct for this size | one fixed size's constants |
|---|---|---|
| 240 px | 72-91% | **2-6%** (79-95% of windows produce no track at all) |
| 120 px | 61-74% | identical (this is the calibration point) |
| 60 px | 37-38% | 79-80% (**better** than "correct") |

Reproduced in five independent variant runs (static occluder, swaying occluder, frozen
front-end, weak distractor, competitive distractor). Two things to take from it:

- Failure is a cliff, not a slope, and it is silent: no error, no crash, just an empty
  result. Anything that ships with a pixel constant carries an undeclared assumption about
  camera distance.
- The direction matters. **Too-tight is fatal, too-loose is cheap.** People bigger than
  calibration kills the tracker; people smaller merely degrades it. At 60 px the "correctly
  re-measured" gate was *worse* than the wrong one, because it was measured from how far the
  person truly moves - while the tracker actually follows *fragments*, whose centroids jump
  between pieces by an amount that does not shrink with the person.

### F2. Essentially all the damage is in one parameter, and it is not the scorer

Correcting exactly one constant at a time and leaving the rest wrong:

| corrected | share of the gap recovered (240 px bucket) |
|---|---|
| association gate (`max_dist`) | **70-78%** |
| merge radius | 2-6% |
| `expected_height` (the scoring term) | **0%** |
| Kalman `P`/`Q`/`R` | **0%** |

The break is at *candidate generation*, not scoring: when the gate is too tight nothing links
into a track, so there is nothing to score. Sensitivity ordering, measured:
**gate >> merge radius > everything else.**

### F3. A dimensionless re-parameterization fixes it, with no learning

Express every pixel quantity - gate, merge radius, expected height, min area, both morphology
kernels, all three Kalman variances - as a multiple of one measured person height, with
coefficients fixed across all scales:

| | worst bucket | spread across buckets |
|---|---|---|
| one size's absolute constants | 2.1% | 77 pp |
| per-size ground-truth-measured constants | 37.0% | 36 pp |
| **one measured scale + fixed coefficients** | **74.6%** | **21 pp** |

It also beat the per-size ground-truth-measured recipe at two of three buckets. Calibrated
coefficients: gate `0.25 x h`, merge `0.75 x h`, expected height `0.95 x h`. The gate is very
flat (0.15-0.40 moves results ~2 pp); the merge radius is the sensitive one and is where the
residual scale-dependence still lives.

### F4. The scale estimate itself does not transfer - and it did not need to

`estimate_person_height` was within ~5% of truth at every scale on synthetic KTH. On real
NFO it reports 315 / 61 / 129 / 200 px against a true ~195 px. NFO's occlusion is far denser
than the synthetic generator's, so fragment-bridging either welds the person to surrounding
foliage or finds too little connected foreground.

Yet NFO accuracy barely moved: mean residual 0.0698 (eight hand-tuned constants) -> 0.0762
(zero dataset-specific constants), no-track 0.2% -> 0.0%. **The robustness comes from the
parameterization, not from the accuracy of the estimate**, because the gate is flat over a
wide band and too-loose is cheap (F1). Practical rule: never build anything that needs an
accurate absolute scale; do build things that only need a scale within a factor of ~2.

### F5. Consistency across scale is not invariance

The first scale proxy (p95 of raw blob heights) grew only 1.35x when real person size grew
2x, because a fixed-pixel front-end fragments a big person into relatively smaller pieces.
Downstream, that produced ~30% accuracy *flat at every scale*: perfectly consistent and
uniformly wrong. Low variance across scale is not evidence of invariance, and it is the
easiest way to fool yourself here. Any future invariance claim needs both a spread number
and a level number.

### F6. The shape term does fragment selection, not distractor rejection

The premise all along was that `expected_height` exists to reject swaying foliage. On real
NFO, switching it off leaves the no-track rate **unchanged** (0.2% either way) but worsens
mean residual **2.7x** (0.0698 -> 0.1912). The tracker does not get captured by foliage and
lose the person; it reports a much worse position while still tracking. That is the signature
of anchoring on the wrong *fragment of the person*.

Corroborating: two synthetic distractor designs (swaying occluder; independent full-height
swaying object beside the frame, at 5x sway amplitude) both left the shape term worth ~2-3 pp.
A rigid shear moves branch tips but barely moves a dense blob's centroid, and the scorer
scores centroids. **Our synthetic data does not reproduce NFO's failure mode**, and the
failure mode is heavy fragmentation, not foliage that translates convincingly.

### F7. NFO's merge radius looks too large - test this on its own

seq2's wrongly-*small* measured height (61 px) produced a smaller merge radius and made that
sequence **better** (0.1095 -> 0.0783). An over-large merge sweeps neighbouring foliage blobs
into the person's box and drags the reported centre off. Current NFO value is 100 px
(= height/2); the synthetic sweep prefers `0.75 x h`. Cheap standalone experiment.

---

## Part 2: What to take from this for the non-learned implementation

In value order. Items 1-3 need no learning and no new data.

1. **Keep the dimensionless parameterization** (already in `tracking/core/track_sequence.py`
   `scale_relative_params`, driven by `preprocess.estimate_person_height`). Treat the measured
   height as a scale *proxy*, not a measurement - do not report it as person height (F4).
2. **Replace the association gate with a formulation that needs no scale at all.** This is the
   single highest-value change left, because the gate holds 70-78% of the sensitivity (F2) and
   it is the one constant whose failure is fatal rather than degrading (F1). Three scale-free
   options, all standard:
   - ratio test: accept the best match only if it is clearly better than the runner-up;
   - gate at `k x` the median nearest-neighbour distance among this frame's detections;
   - Mahalanobis distance against the Kalman innovation covariance with `Q`/`R` estimated
     online from observed residuals, gated by a chi-square quantile.
   All three use the data's own distance distribution as the yardstick, so there is no pixel
   constant and no person-height dependence. Bias every choice toward *too loose* (F1).
3. **Do not derive the gate from ground-truth motion statistics.** Measured GT displacement is
   ~0.10-0.13 of a body height, and that value is a lower bound that performs badly (F1);
   the fitted 0.25 works because fragment centroids jump more than people move.
4. **Retune the merge radius, and check NFO specifically** (F7). Also note a wrong shape prior
   is worse than no shape prior (82.2% vs 88.6% measured): if a term cannot be scaled
   correctly, switching it off beats leaving it mis-scaled.
5. **Change how results are reported.** Per-scale numbers plus spread plus worst bucket, never
   pooled accuracy alone (F5). The kill test's leave-one-in ablation is the diagnostic that
   actually localized the problem; keep using it.

---

## Part 3: The learned components

Approved direction: learned ranking and a learned association gate. The measured evidence
constrains the design more than it constrains the ambition, so the constraints come first.

### Design constraints that follow from the findings

- **Learn where the sensitivity is.** Gate first (F2: 70-78%), fragment grouping second
  (F6: the real NFO error), track ranking third. Building the ranker first would repeat this
  session's mistake of optimizing the 0%-of-the-gap component.
- **Every feature dimensionless.** Ratios, angles, ranks, correlation coefficients,
  time-normalized quantities. Not pixels, not areas, not raw speeds. This is what replaces
  the constants; a single pixel-valued feature reintroduces the whole problem.
- **Prefer features that need no scale estimate at all.** Normalize *within the window*: a
  track's height over the median candidate height, its displacement over the median candidate
  displacement, its rank among candidates. All candidates in a window share a frame, so scale
  cancels algebraically - which matters because the explicit estimator is exactly the piece
  that failed to transfer (F4).
- **Mechanically test invariance, do not assume it.** Unit test: for every feature `f`,
  assert `f(video) ~= f(resize(video, s))` for `s` in {0.5, 1, 2} within tolerance. Given F5,
  this guard is not optional - it is the cheapest possible protection against shipping a
  confidently non-invariant feature.
- **Small models.** Logistic regression or a depth-2/3 gradient-boosted tree. The point is
  removing constants, not capacity, and a large model will fit the occluder generator.
- **No CNN appearance embedding yet.** F2 and F6 both say the wins are not in appearance
  re-ID. Masked-region intensity mean/std and a gradient histogram are cheap and can go in as
  ordinary features if wanted, without a backbone.

### Component A: learned association gate (build first)

Replace the `max_dist` threshold and the hand-set `Q`/`R` with a learned match probability.

- **Formulation.** Binary verification over (track, detection) pairs: is this detection the
  continuation of this track? Assignment cost `-log p(match)`; Hungarian on that; gate at a
  probability threshold. A probability threshold is dimensionless and transfers; a pixel
  threshold does not. This is a strict generalization of Mahalanobis gating - it learns the
  weighting instead of assuming it.
- **Features** (all dimensionless): displacement over the frame's median nearest-neighbour
  distance; displacement over the track's own recent mean step; displacement normalized by
  the Kalman innovation covariance; cosine between this step and the track's recent velocity;
  relative change in area, height, aspect; frames since last match; this detection's rank
  among candidates for this track, and the reverse rank (is the match mutual?).
- **Labels are free.** Synthetic data knows exactly where the person is in every frame, so a
  pair is positive when both blobs belong to the person. Real KTH ground-truth boxes give the
  same labels for the unoccluded case.
- **Success gate:** worst-bucket >= 74.6% (what the two-coefficient formula already achieves,
  F3), with no per-dataset constant, and NFO mean residual <= 0.0762. If a learned gate cannot
  beat two dimensionless numbers, that is a real result - report it and stop.

### Component B: learned fragment grouping and anchor selection (build second)

This is where NFO's actual error lives (F6), and it removes the merge radius (F7) entirely.

- **Formulation.** Pairwise affinity: do these two blobs belong to the same object? Then
  cluster (connected components over the affinity graph, or agglomerative with a probability
  cut) and report the cluster's centroid. Replaces the fixed-radius merge and the
  "which fragment do I anchor on" choice in one model.
- **Features:** gap between blobs over the larger blob's height; vertical overlap fraction;
  height and area ratios; horizontal offset over height; motion coherence over the window
  (do the two blobs move together?); optionally masked-intensity similarity.
- **Labels are free** from synthetic data; on NFO the ground-truth box gives blob membership,
  so NFO can be used as a *test* set with real labels without ever tuning on it.

### Component C: learned track ranking (build third)

Only after A and B, and with the measured ceiling in mind (2-9 pp synthetic; the real NFO
contribution turned out to be fragment selection, which is Component B's job).

- **Rank, do not score.** The output is only consumed through an argmax within one window, so
  absolute calibration is irrelevant and is the part that would not transfer. Pairwise ranking
  loss on (person track, distractor track) pairs from the same window.
- **Features:** within-window normalized size and displacement (above), net displacement over
  path length (translation vs oscillation), gait periodicity of width/aspect oscillation
  (dimensionless in space *and* diagnostic of people specifically), residual of the linear fit
  over the track's own displacement.

### Data: the plan, and the one thing that must happen first

Data volume is not the constraint - KTH has 298 usable sequences and the occluder is
synthetic, so the effective dataset is essentially unbounded. **Fidelity is the constraint**,
and F6 says so concretely: our current occluders do not produce NFO's failure mode, so a model
trained on them will optimize the wrong thing however much data we generate.

1. **Calibrate the occlusion generator against NFO first.** Measure on real NFO: blobs per
   person, largest-blob height over person height, total foreground area over person-box area,
   and the distribution of gaps between a person's own fragments. Then tune density / occluder
   model / morphology until the synthetic distributions match. This is the "density not
   calibrated against real coverage statistics" gap the log has flagged three times, and it is
   now the blocking item rather than a footnote. It is also cheap: it needs no training, and
   it gives a quantitative match criterion instead of visual judgement.
2. **Randomize continuously, not in buckets.** Person scale 0.4-2.5x (continuous), occluder
   density 0.1-0.6, occluder model (branches / morphological blobs / texture patches),
   occluder motion (static / sway / cover-and-reveal), frame stride, contrast and noise,
   background. Held-out *scales* matter more than held-out sequences: train on a scale range,
   validate at scales never trained on.
3. **Protocol.** Train on synthetic only. Validate on synthetic at unseen scales. Test on real
   NFO, once, without tuning. Report per-scale spread and worst bucket (F5). A second gait
   dataset is valuable but as a *second test set* - different camera geometry and appearance -
   not as more training data; cross-dataset transfer is the claim being made.
4. **Keep a non-learned baseline in every table.** The two-coefficient dimensionless formula
   from F3 is the thing to beat, not the original hand-tuned constants.

### What the contribution is

Not "learned scorer beats heuristic". The defensible framing, supported by these
measurements, is: **a fragmented-occlusion tracker with no dataset-specific pixel constants,
which matches hand-tuned performance on the target data and degrades gracefully over a 4x
range of person sizes** - plus the negative results, which are genuinely useful: which
parameter actually carries the scale sensitivity, that a fragment-statistics scale proxy does
not transfer to dense real occlusion, that consistency across scale is not invariance, and
that the shape term's real job was fragment selection rather than the distractor rejection its
docstring claims.
