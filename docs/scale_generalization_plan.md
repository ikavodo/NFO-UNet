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

### F6. ~~The shape term does fragment selection, not distractor rejection~~ SUPERSEDED

**Wrong - see the calibration addendum at the end of this file.** Measured NFO fragmentation
is mild (1.85 blobs per person, tallest blob spanning 86% of the box), so there is little
fragment ambiguity to resolve, and the p90 residual of 0.5929 without the term means the
tracker locks onto a *different object*, not a different fragment. The term rejects non-person
movers, as its docstring always said. The paragraph below is kept for the record.

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
2. ~~**Replace the association gate with a formulation that needs no scale at all.**~~
   **WITHDRAWN - measured and wrong, see the Stage 1 result at the end of this file.** The
   median-nearest-neighbour rule loses 9.7-11.3% of person detections where the existing
   `0.25 x h` rule loses 0.0%, because the median nearest-neighbour distance is a statistic of
   the clutter rather than of the person. Distance relative to body height is the right
   reference and the gate is already correct. Original text kept below for the record; the
   ratio-test and online-Mahalanobis variants remain untested and must clear the same 0.0% bar.
   Three scale-free options, all standard:
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

---

## Addendum: occluder calibration, and a correction to F6

`tracking/eval/occluder_calibration.py` measures four dimensionless fragmentation statistics
strictly inside the person's own ground-truth box, on real NFO and on synthetic KTH, with the
segmentation front-end derived from each dataset's own ground-truth person height so neither
gets a front-end tuned to it. Person height comes from ground truth, not from
`estimate_person_height` - per F4 that estimator has no business inside a calibration loop.

### What real NFO fragmentation actually looks like

| | blobs per person | tallest blob / box height | fill of box | inter-blob gap / box height |
|---|---|---|---|---|
| NFO seq1 | 1.65 | 0.873 | 0.452 | 0.381 |
| NFO seq2 | 1.93 | 0.870 | 0.504 | 0.350 |
| NFO seq3 | 2.02 | 0.798 | 0.429 | 0.364 |
| NFO seq4 | 1.84 | 0.873 | 0.508 | 0.363 |
| **NFO pooled** | **1.85** | **0.855** | **0.475** | **0.365** |

**NFO people are only mildly fragmented.** Under two blobs on average, and the tallest single
blob spans 86% of the person's box height. This contradicts an assumption carried through this
whole investigation.

### The synthetic occluder was ~7x too aggressive

Synthetic KTH, 1x scale, sweeping density and line thickness (mismatch = mean relative error
over the three shape statistics; `fill_frac` excluded, see below):

| thickness | density | blobs | tallest/box | fill | gap | mismatch |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 1.66 | 0.873 | 0.230 | 0.392 | 0.064 |
| 1 | **0.05** | **1.75** | **0.838** | 0.220 | 0.394 | **0.050** |
| 1 | 0.10 | 1.91 | 0.790 | 0.211 | 0.398 | 0.063 |
| 1 | 0.15 | 2.02 | 0.731 | 0.198 | 0.409 | 0.113 |
| 1 | **0.35** (used all along) | 2.42 | 0.552 | 0.149 | 0.416 | 0.238 |
| 5 | 0.05 | 1.82 | 0.809 | 0.211 | 0.418 | 0.067 |

**Calibrated value: density 0.05, thickness 1.** `OCC_DENSITY` default changed accordingly;
pass `--density 0.35` to reproduce any earlier result. Line thickness barely matters at
matched density, so the few-thick-bands idea is not needed.

`fill_frac` is excluded from the score for a reason that is itself a finding: adding occlusion
can only lower it, and synthetic KTH's **unoccluded** fill (0.230) is already less than half
NFO's **occluded** fill (0.475). MOG2 recovers a much smaller share of the person on KTH than
on NFO. So on the statistic that measures how much of the person survives segmentation,
**un-occluded synthetic KTH is already harder than real occluded NFO** - our benchmark's
difficulty comes mostly from KTH's own segmentation noise, not from the occluder we added. No
density setting can fix that; it is a property of the source footage.

### Correction to F6

F6 claimed the shape term does *fragment selection* on heavily-fragmented NFO rather than
distractor rejection. **That was wrong**, and this measurement is why: NFO people are not
heavily fragmented (1.85 blobs, 86% intact), so there is little fragment ambiguity to resolve.

Re-reading the same NFO run with that in mind, the tell was already there and I read it
wrongly: without the shape term the **p90 residual is 0.5929** - more than half the frame
away. That is not a different fragment of the person, it is a different object. The no-track
rate stays at 0.2% because a track is always found; it is simply the wrong one. So
`score_and_fit`'s docstring was right all along: **the term rejects non-person movers**, and
on NFO that is worth taking p90 from 0.59 to 0.10.

Why the synthetic distractor experiments still measured nothing is then a separate, mechanical
failure, already diagnosed: a rigid shear of a dense branch band moves branch tips but barely
moves the blob centroid, and the scorer scores centroids. The concept was fine; the stimulus
was not.

### What this does and does not change

- **F1, F2, F3, F5 stand.** The scale cliff, the gate holding 70-78% of the sensitivity, the
  dimensionless fix, and consistency-is-not-invariance were all measured across five variant
  runs and are properties of pixel thresholds, not of occluder density.
- **F4 stands**, and gains support: the estimator over-read seq1 as 315px partly because dense
  occlusion welds a person to nearby foliage - but NFO's *box-internal* fragmentation is mild,
  so the failure is about foliage outside the person, i.e. about blob identity, not breakage.
- **F6 is replaced** by the corrected reading above: the shape term rejects wrong objects.
- **Every scoring-term measurement in this log was taken at density 0.35**, i.e. in a
  7x-over-fragmented regime that does not represent NFO. The scoring-term numbers (2-9pp) are
  therefore measured in the wrong regime and should be re-taken at the calibrated density
  before any conclusion about the scorer is trusted. The gate numbers are unaffected.
- **Data-strategy consequence:** KTH + synthetic occlusion is a weak proxy for NFO
  fragmentation, and the mismatch is dominated by KTH's base segmentation rather than the
  occluder. Generating more of it will not fix that. Either improve the base segmentation on
  KTH, or accept the mismatch and lean on dimensionless features to bridge it - and in either
  case validate on NFO, never tune on it.

---

## Stage 1: the smallest experiment that moves toward learned gating

Deliberately local: no tracker changes, no training pipeline, no new data generation, one
script, one number that decides whether to continue. It targets the gate because that is where
70-78% of the scale sensitivity is (F2), and it is offline so a negative result costs nothing.

**Question.** Can a learned match rule hold its operating point across scale better than the
best non-learned dimensionless rule can?

That is the only question worth asking first. If the fixed dimensionless threshold already
transfers perfectly, a learned gate has nothing to add and the whole direction can be dropped
for the cost of an afternoon.

**Setup.**
- Data: 3 KTH sequences, 3 scales (0.5x / 1x / 2x), calibrated occluder (density 0.05).
  Everything already exists in `kill_test_scale.build_bucket`.
- Examples: for each frame pair `(t, t + nth_frame)`, every (detection in `t`, detection in
  `t + nth_frame`) pair. Label 1 if both detections lie inside the person's ground-truth box,
  0 otherwise. That is exactly the decision `track_blobs`'s gate makes, in isolation.
- Features, 6, all dimensionless, none needing a person-height estimate:
  1. centroid displacement / median nearest-neighbour distance among frame `t`'s detections
  2. centroid displacement / this detection's own blob height
  3. `|dheight| / mean height`
  4. `|darea| / mean area`
  5. `|daspect| / mean aspect`
  6. rank of this candidate by distance (1 = nearest) + whether the match is mutual
- Model: logistic regression (sklearn 1.6.1 is already in the env).
- Baselines: (i) today's rule, `distance < 0.25 x person_height`; (ii) the scale-free
  alternative, `distance < k x median nearest-neighbour distance`, with `k` tuned on 1x.

**Protocol.** Fit on 1x only. Evaluate at 0.5x and 2x. This is the scale-transfer test in
its cheapest possible form.

**Metric.** Not accuracy - the classes are wildly imbalanced and the costs are asymmetric.
Report the **false-negative rate on true person pairs at a fixed false-positive rate**, per
scale, because F1 established that missing true matches is what kills the tracker while
extra candidates are cheap.

**Go/no-go.** Continue only if the learned rule's false-negative rate at 0.5x and 2x is
materially better than both baselines' when all three are tuned on 1x alone. If baseline (ii)
already transfers flat, implement baseline (ii) in the tracker (it is a ~5-line change) and
stop - that is a win without any learned component.

**By-products worth having either way.** The dimensionless feature extractor, and the
resize-invariance unit test on it (`f(video) ~= f(resize(video, s))` for `s` in {0.5, 1, 2}),
which components A/B/C all need and which F5 says must exist before any invariance claim.

**Only if stage 1 passes,** stage 2 plugs `-log p(match)` into `track_blobs`'s cost matrix and
re-runs the kill test, with the bar being `scale_rel`'s worst bucket of 74.6% (F3) and NFO mean
residual <= 0.0762 (F4). Not before.

---

## Stage 1 result: NO-GO on a learned gate, and a retraction

`tracking/eval/stage1_gate_learning.py`, 3 sequences, calibrated occluder (density 0.05),
injected swaying distractor, logistic regression on 7 dimensionless features fit on 1x only,
every rule's threshold chosen on 1x only and then frozen.

Loss rate = fraction of person detections with **no** accepted person match (the failure that
kills tracks, per F1), at a matched 1% contamination budget:

| bucket | fixed-h (0.25 x h) | median-nn (scale-free) | learned |
|---|---|---|---|
| 0.5x | **0.0%** [1.0%] | 9.7% [0.8%] | 6.9% [1.3%] |
| 1.0x | **0.0%** [1.0%] | 11.3% [1.0%] | 10.2% [1.0%] |
| 2.0x | **0.0%** [1.8%] | 11.3% [2.1%] | 10.6% [2.3%] |

**The plain distance threshold in the right units loses nothing, at any scale.** The learned
rule loses ~10% of person detections at the same contamination cost, and so does the
"scale-free" median-nearest-neighbour alternative. Verdict: NO-GO. All three transfer *flatly*
across scale (drift ~0pp), so scale transfer was not the differentiator - absolute quality was,
and the simple rule wins outright.

### Retraction: Part 2, item 2 was wrong

Part 2 recommended replacing the gate with a scale-free formulation, `k x` median
nearest-neighbour distance, as "the single highest-value change left". **Measured, that rule is
worse than what is already there** - 9.7-11.3% loss against 0.0%. The reason in hindsight is
simple: the median nearest-neighbour distance among detections is a statistic of the *clutter*,
not of the person, so when the person is cleanly separated from a few distant blobs the
reference becomes large and the gate lets in nonsense, forcing a tighter operating point.
Distance relative to *body height* is the right reference. Item 2 is withdrawn; the ratio-test
and online-Mahalanobis variants remain untested but should be held to the same 0.0% bar.

### What this says about the gate, and about the earlier 85pp cliff

The cliff (F1/F2) was never a *discrimination* problem - it was a *units* problem. Given a
threshold expressed in the right units and calibrated once, the gate makes no mistakes worth
learning away: with 2-4 detections per frame and >90% of them belonging to the person, the
association decision is nearly trivial. Getting the units right is the whole fix, and it is
already implemented (`scale_relative_params`). **The gate is done.** Nothing learned belongs
here.

One caveat kept honest: `fixed-h` is handed a ground-truth person height in this experiment,
where deployment gets `estimate_person_height` (3x off on NFO, per F4). That is tolerable only
because too-loose is cheap (F1) - but it means the gate's remaining risk is the estimator, not
the rule.

### Why KTH cannot answer the next question either

The measurement that matters for what comes next: with the occluder calibrated to NFO and a
swaying distractor injected, only **3.1% / 4.6% / 13.8%** of candidate destinations (at
0.5x / 1x / 2x) are non-person. KTH's background is essentially clean, so it contains almost no
negatives. That is why the first version of this experiment was saturated - at a 10%
contamination budget, accepting *everything* was within budget at every scale.

So on the data-strategy question: **the constraint was never quantity, and it is not even
occluder fidelity any more - it is the absence of clutter.** Augmenting KTH harder will not
manufacture distractors that a clean-background dataset does not contain, and the two synthetic
distractor designs tried so far are trivially separable. Real NFO, by contrast, is full of real
clutter - that is precisely why its shape term is worth taking p90 from 0.59 to 0.10.

### The next smallest experiment

The remaining ambiguity is **which object**, on cluttered data, which is the ranking problem
and not the gate. NFO is the only data available that actually contains it, so the next
experiment has to use NFO for fitting, with leave-one-sequence-out so nothing is tuned and
tested on the same footage:

- Run the existing tracker on NFO, keep **all** candidate tracks per window rather than only
  the argmax winner.
- Label each track by whether its centre lands within the eval threshold of the ground truth.
- Features: dimensionless and within-window normalized - track height over the median candidate
  height in that window, net displacement over path length (translation vs oscillation),
  residual of the linear fit over the track's own displacement, span, size stability over the
  track's running median.
- Fit a pairwise ranker on 3 of the 4 NFO sequences, evaluate on the held-out one, rotate.
- Baseline to beat: `score_and_fit` as it stands, i.e. mean residual 0.0698 / p90 0.1049.
- Go/no-go: a learned ranker must beat that on held-out sequences. With 4 sequences the
  variance will be high, so treat a small win as inconclusive rather than as success.

This keeps the same discipline as stage 1 - offline, no tracker changes, one number - while
moving to the only stage where the evidence says learning has something to bite on.
