# MVOT oracle: does the integral image remove distractors for SAM2? (design, 2026-11-10)

Status: **H0 failed (2026-11-10): no distractor headroom on MVOT; see nfo_failure_log.md. H1 not run.** Was: design, awaiting sign-off. Nothing built beyond the toy below.

## Context

Reviewer request: show the integral-image idea beyond fragmented occlusion. MVOT (SatSAM2's synthetic satellite
benchmark, arXiv 2511.18264; HF `Frank0666/MVOT`, local `~/PycharmProjects/MVOT`) has a per-video "similar object"
(distractor) label: 170 of 1,579 videos, 166 in the regular subset; the label covers whole videos (median 100% of
frames). Measured: static camera (phase correlation, shift 0.00 px over 50 frames, 24 videos), 100 frames at 20 fps,
1024×1024, targets median 19×20 px (p10–p90 10–30), speeds 1–3 px/frame. Boxes only, no masks.

## Mechanism (derived)

Integral image of past frames aligned on the target (`adapter/integral_image.py`): evidence mass a = Σ_k w_k G_k(x + s_k).
For a distractor of length L with velocity Δv relative to the target, alignment on the target shifts it by Δv·k in the
k-th past frame, so its mass is a_d ≈ Σ_k w_k · max(0, 1 − |Δv|·k / L); the target keeps a ≈ G. Separation needs
|Δv|·T_eff > L, T_eff ≈ τ. MVOT: L ≈ 20 px, |Δv| ≲ 3 px/frame → T_eff ≳ 7–10 frames (NFO's τ = 3.5 is too short).
Co-moving distractors (|Δv| ≈ 0: same lane, same speed) are not separable at any window (boundary).

## Feasibility (Phase 5, done; 10 random similar-object videos, script inline in the session, traces `images/benchmark/mvot/`)

1. NFO gate as is: robust scale collapses on rendered video (median |F − B| = 0 → MAD → 1e-6 floor, z ~ 10⁶), gate
   fires on 1–2 gray-level compression noise. **Fix: scale floor 2 gray levels, fixed z0 = 3** (changes > ~6 levels).
2. τ = 3.5, 6 frames: non-target moving blobs (a > 0.5, ≥ 60 px, outside 3× target box) 1,103 → 1,128 (×1.02).
3. **τ = 12, 30 frames: 870 → 527 (×0.61), lower in 10/10 videos; target mass unchanged** (G 0.21–0.79 ≈ a).
   Window chosen from the |Δv|·T > L condition before running, not tuned.

## Hypotheses

- **H0 (headroom):** SAM2.1 video loses the target more on similar-object videos than on matched videos without the label.
  KILL: box AUC(similar) ≥ AUC(matched) − 0.03 → there is no distractor problem for SAM2 to solve on MVOT; stop.
- **H1 (oracle):** feeding SAM2.1 video the integral image C (GT-motion aligned, τ = 12, 30 frames) instead of the raw
  frame raises box AUC on similar-object videos, and more than on matched videos.
  BOUNDARY: co-moving distractors; occlusion subset (target absent) excluded; frames with < 30 frames of history use raw.
  KILL: ΔAUC(similar) ≤ +0.01 or its paired CI includes 0.

- **H0b (failure anatomy, same run as H0, no new method):** at every SAM2.1 / SAMURAI failure on MVOT (IoU < 0.1 for
  ≥ 5 consecutive frames) the wrongly tracked object's velocity (centroid motion of the predicted mask) is compared with
  the target's (GT). Classes: *co-moving* |Δv| < L/30 (no window can separate it, see Mechanism), *static* (mask still,
  target moving), *different* (separable by the integral image), *empty* (mask lost). Decision: mostly *different* /
  *static* → evidence-mass selection term (H1); mostly *co-moving* → appearance template from the integral image first.
- **H1 (revised 2026-11-10):** the integral image as an extra **candidate-mask selection term** in SAMURAI's
  Kalman-weighted score (the mechanism SatSAM2 builds on), not as the input frame. Thresholds to be learned, see
  "Learned formulation" (session notes); arms T1 vs T1 + evidence term, same videos.

## Baseline and the one variable

Both arms: SAM2.1-b+ video predictor (laptop env), t0 prompt = GT box, same frames, **grayscale** input replicated to
3 channels (C is grayscale, so the baseline is too). Arm `raw`: frame t. Arm `C`: integral image of t−30..t−1 aligned by
GT box-centre motion. One variable: the input image. Reference only: SatSAM2's published numbers
(different protocol; not compared).

Oracle caveats, stated: alignment uses GT motion (upper bound for a Kalman-aligned version, as SatSAM2 has), and the
empty scene B is the median of all 100 frames (non-causal). Generic effects of C (static background blurred by the
shift, ±60 px) are measured by the matched control: the distractor claim rests on the difference-in-differences.

## Data

30 similar-object videos (regular subset, random seed 0) and 30 matched videos without the label (same lighting, tilt
and view, random). Fixed as the rerun subset. Metric: one-pass box success AUC (tight box of the mask vs GT,
`eval/box_metrics` logic), plus lost-target rate (IoU < 0.1 for ≥ 5 consecutive frames). Paired video bootstrap CIs.

## Pass / fail (pre-registered)

- H0 passes if AUC_raw(similar) < AUC_raw(matched) − 0.03 with CI excluding 0. Otherwise stop: report "no headroom".
- H1 passes if ΔAUC(similar) = AUC_C − AUC_raw ≥ +0.03 with paired CI > 0, **and** ΔAUC(similar) − ΔAUC(matched) > 0
  with CI > 0. ΔAUC(similar) > 0 without the difference-in-differences = generic effect of C, not distractor removal.

## Cost and scope cuts

60 videos × 2 arms × 100 frames ≈ 12k SAM2 video frames (~11 min at 53 ms) + C (30 warps of 1024² per frame, est.
~0.2 s → ~20 min, cached to disk). Cut order if short: (1) night/dusk videos, (2) 20 per group instead of 30,
(3) never drop the matched control (it is what separates distractor removal from generic blur).

## Execution order

1. H0 alone (raw arm, 60 videos). Stop if it fails.
2. Cache C for the 60 videos (`images/benchmark/mvot/` montages for 3 videos per group).
3. C arm, score, report table + CIs.
4. If H1 passes: the causal version (alignment from a constant-velocity / Kalman fit on SAM2's own boxes) and DiDi.

## Adversarial notes

- C changes more than distractors: the shifted static background is blurred, which may help or hurt SAM2 everywhere;
  the matched control is the guard. A sharper isolation (C only inside gated pixels, raw elsewhere) is the fallback if
  the generic effect dominates.
- The "similar" label is undocumented beyond its name; it may mark co-moving traffic, which the mechanism cannot remove.
- An idea-critic pass could test whether SatSAM2's Kalman + state machine already removes the same failures (then the
  oracle gain would not survive against SatSAM2). Not run; ask first.
