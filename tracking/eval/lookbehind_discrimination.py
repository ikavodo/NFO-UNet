"""Does ACCUMULATING blob tracks (long lookbehind) discriminate the person from foliage
better than the current 7-frame window does?

    python -m tracking.eval.lookbehind_discrimination
    python -m tracking.eval.lookbehind_discrimination --seqs seq1_gt --depths 7 15 31

This is the cheap kill/confirm test that comes BEFORE building a persistent-track engine.
No engine is written: tracks are linked once over the whole sequence offline with the
existing Kalman/Hungarian code, and then each candidate's history is TRUNCATED to a trailing
window of N strided samples. N=7 reproduces what the streaming tracker does today, so it is
the do-nothing baseline, and every larger N is the same tracks seen with more past evidence.
One variable changes.

WHY IT SHOULD WORK, AND WHAT WOULD FALSIFY IT

Lookahead and lookbehind cost different things. Lookahead costs latency, which is why
SPAN=6 is capped at 6 frames. Lookbehind is free: a persistent track can carry a hundred
frames of past evidence at zero added delay. The current design throws all of it away -
track_blobs rebuilds every track from scratch inside each 7-frame window.

Two closed-form reasons more history should separate a walking person from swaying foliage:

  1. Velocity precision improves as N^(-3/2). For a linear fit over N samples spaced D,
     Var[v] = 12 s^2 / (N (N^2 - 1) D^2). From N=7 to N=31 the velocity standard error falls
     by (31/7)^1.5 ~ 9.3x, for free.
  2. Bounded oscillation and unbounded translation separate LINEARLY in N. Foliage sway has
     displacement bounded by its amplitude, net_disp <= 2A for any N, while a person gives
     net_disp = v N D. The ratio therefore grows like N. At N=7 (12 real frames, 0.5s) a sway
     with a ~1s period has not completed a half cycle, so it is indistinguishable from
     translation - which is the confusion actually observed. By N=31 (2.6s) it has completed
     two cycles and its net displacement collapses.

FALSIFIED IF: AUC does not rise with N for any motion feature. That would mean the confusion
is not about observation length, and a persistent-track engine would buy nothing but speed
we already have (52 fps at 1920x1080 measured).

THE TRAP THIS ALSO TESTS. Naive accumulation makes things WORSE: the current score is
span * net_disp / (1 + resid_std), and both span and net_disp grow without bound with
history, so a long-lived foliage track outscores a briefly-seen person on persistence alone.
So cur_score's hit@1 and the length-normalised features' AUC are reported separately - they
answer different questions:
  - cur_score hit@1 rising with N  => accumulation helps the tracker AS IT STANDS.
  - only the normalised features' AUC rising => accumulation helps only with a new score.

FEATURES, all oriented so higher = more person-like, and all named rather than invented:
  cur_score     span * net_disp / (1 + resid_std) - the current design, for reference
  net_disp      raw end-to-end displacement; grows with N for translation, bounded for sway
  mean_speed    net_disp / (n-1); length-normalised
  straightness  net_disp / path_length - the directionality ratio (a.k.a. confinement ratio,
                Batschelet 1981; standard in cell-migration analysis). 1 = perfectly straight,
                ~0 = returns to where it started.
  msd_alpha     slope of log MSD vs log lag, i.e. the anomalous-diffusion exponent. 2 =
                ballistic/directed, 1 = diffusive, ~0 = confined. This is the canonical
                formalisation of "bounded oscillation vs unbounded translation".
                Caveat: centroid noise inflates MSD at lag 1 and biases alpha DOWN, more so
                at small N, so treat its absolute value as indicative and its trend as the
                signal.
  straight_run  1 - (fraction of consecutive x-steps that reverse sign); an oscillation count.

SCORING. Labels come from the ground truth ONLY as a scoring target, never as an input:
person_height is measured from the footage with estimate_person_height, and the GT-derived
height is printed alongside purely as a cross-check. A candidate counts as the person when
its merged-blob centre lands within max_dist_error = 0.1 of the GT box centre in normalised
diagonal units, matching tracking/eval/eval_nfo.py's own convention.

RESULT, 2026-08-31, all four NFO sequences (224x224, measured person height 53-81px):

    AUC (person vs distractor), 2250 person samples vs only 110 distractor samples
    feature            N=7     N=15     N=31     N=51
    cur_score        0.774    0.758    0.769    0.769     flat
    net_disp         0.734    0.777    0.788    0.788     +0.054
    mean_speed       0.563    0.527    0.528    0.528     -0.035
    straightness     0.456    0.434    0.435    0.434     -0.022
    msd_alpha        0.572    0.583    0.604    0.605     +0.033
    straight_run     0.658    0.662    0.658    0.656     flat

    hit@1 (top-ranked track is the person, n=1486)
    cur_score        0.991    0.986    0.988    0.988

**Accumulation does not help on NFO, because there is almost nothing to discriminate
against.** Only 1.2-1.8 candidate tracks per frame, 20-56% of frames have >=2 candidates at
all, and 110 distractor samples against 2250 person samples. cur_score's hit@1 is already
0.991 against a ceiling of 0.985-0.994 (how often the person is among the candidates), so
there is no headroom for a better ranking to occupy.

The two closed-form predictions were directionally right and quantitatively irrelevant: the
only features that improve are exactly the two that formalise "bounded oscillation vs
unbounded translation" - net_disp (+0.054) and msd_alpha (+0.033) - while every
length-normalised feature degrades, because normalising by observation length is precisely
what removes the signal that grows with it. So the mechanism is real; the problem it solves is
not present here.

WHY NFO DIFFERS FROM ido_walk.mkv, which is the scene where foliage confusion WAS observed:
NFO's occluders are bare trees - rigid and static, so MOG2 assigns them to the background and
they spawn no tracks. ido_walk's occluder is an indoor plant whose leaves move, which does
spawn competing tracks. The discrimination problem is therefore a property of MOVING
occluders, and NFO has none. Testing accumulation against it needs footage with moving
distractors AND ground truth; NFO has the ground truth but not the distractors, ido_walk has
the distractors but no ground truth.

SELECTION BIAS, and it points at the better experiment. This evaluates only frames where a
candidate track has a detection at the readout frame itself (86% of strided GT frames; 14%
are dropped). Those dropped 14% are exactly the extrapolation cases the real pipeline handles
by evaluating a fitted line, so the 0.991 here is measured on a favourable subset and is NOT
comparable to eval_nfo.py's 90.0% hit rate on the 800x600 data. It also means the untested
frames are the ones where lookbehind should pay MOST: with no observation at the readout
frame the position comes entirely from the fit, and Var[v] ~ N^-3 makes a longer history
directly sharper there. The measurement that would settle it is the localisation residual on
extrapolation frames only, fit over N=7 versus N=31 - a different experiment from this one.

DATA NOTE. This reads data/nfo_processed/<seq>_gt/, which is 224x224 with a 5-column
`frame,x,y,w,h` groundtruth.txt. That is NOT the layout tracking/eval/eval_nfo.py expects
(800x600 under data/nfo_final/nfo_final, a 4-column file, plus a second groundtruth* file), so
eval_nfo.py will not run on this directory unmodified. The SAM masks here are *_sammask.png,
so the *.jpg frame globs are unaffected by them.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from scipy.stats import rankdata

from tracking.core.blob_tracker import _Track, detect_blobs, merged_center, track_blobs
from tracking.core.preprocess import (estimate_person_height, filter_by_shape,
                                      foreground_mask, refine_mask)
from tracking.core.track_sequence import scale_relative_params

NTH_FRAME = 2          # matches the streaming tracker's stride
MIN_TRACK_LENGTH = 3
MAX_AGE = 6
MAX_DIST_ERROR = 0.1   # eval_nfo's hit threshold, normalised diagonal units
DEPTHS = (7, 15, 31, 51)
FEATURES = ('cur_score', 'net_disp', 'mean_speed', 'straightness', 'msd_alpha', 'straight_run')


def load_sequence(seq_dir: str):
    """Frames (contiguous, greyscale) and ground-truth boxes keyed by frame index."""
    jpgs = sorted(glob.glob(os.path.join(seq_dir, '*_or.jpg')))
    assert jpgs, f"no *_or.jpg in {seq_dir}"
    frames = np.stack([cv2.imread(f, 0) for f in jpgs])
    gt = {}
    with open(os.path.join(seq_dir, 'groundtruth.txt')) as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) == 5:
                gt[int(parts[0])] = tuple(float(v) for v in parts[1:])
    return frames, gt


def gt_runs(gt: dict):
    """The ground truth is contiguous in runs (one person pass each) separated by gaps where
    nobody is in frame. A trailing window must never reach across a gap - the history there
    belongs to a different traversal - so return the runs and clip against them."""
    ks = sorted(gt)
    runs, start = [], ks[0]
    for a, b in zip(ks, ks[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, ks[-1]))
    return runs


def trajectory_features(hist_items):
    """hist_items: [(strided_index, x, y), ...] in increasing index order.
    Returns a dict of features, all oriented so higher = more person-like."""
    idx = np.array([i for i, _, _ in hist_items], dtype=float)
    xs = np.array([x for _, x, _ in hist_items], dtype=float)
    ys = np.array([y for _, _, y in hist_items], dtype=float)
    n = len(idx)

    span = idx[-1] - idx[0] + 1
    net_disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    steps = np.hypot(np.diff(xs), np.diff(ys))
    path = float(steps.sum())

    A = np.vstack([idx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    resid_std = float((xs - A @ coef).std())

    dx = np.diff(xs)
    reversals = float(np.mean(np.sign(dx[1:]) != np.sign(dx[:-1]))) if n >= 3 else 0.0

    # anomalous-diffusion exponent: MSD(lag) ~ lag^alpha
    alpha = float('nan')
    lags = np.arange(1, max(2, n // 2) + 1)
    lags = lags[lags < n]
    if len(lags) >= 2:
        msd = np.array([np.mean((xs[l:] - xs[:-l]) ** 2 + (ys[l:] - ys[:-l]) ** 2)
                        for l in lags])
        ok = msd > 0
        if ok.sum() >= 2:
            alpha = float(np.polyfit(np.log(lags[ok]), np.log(msd[ok]), 1)[0])

    return dict(cur_score=span * net_disp / (1.0 + resid_std),
                net_disp=net_disp,
                mean_speed=net_disp / max(n - 1, 1),
                straightness=net_disp / path if path > 0 else 0.0,
                msd_alpha=alpha,
                straight_run=1.0 - reversals,
                n=n)


def auc(pos, neg) -> float:
    """Mann-Whitney U / |pos||neg|. Higher than 0.5 means the feature ranks the person above
    the distractors. NaNs are dropped rather than imputed."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate_sequence(seq_dir: str, depths=DEPTHS, verbose=True):
    frames, gt = load_sequence(seq_dir)
    H, W = frames.shape[1], frames.shape[2]
    runs = gt_runs(gt)

    person_height = float(estimate_person_height(frames, bg_frames=30))
    gt_height = float(np.median([b[3] for b in gt.values()]) * H)   # cross-check ONLY
    kw, kalman = scale_relative_params(person_height)
    _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = kalman
    if verbose:
        print(f"{os.path.basename(seq_dir)}: {len(frames)} frames {W}x{H}, {len(gt)} gt rows in "
              f"{len(runs)} runs; measured person height {person_height:.0f}px "
              f"(gt median box height {gt_height:.0f}px, cross-check only, not an input)")

    masks = filter_by_shape(refine_mask(foreground_mask(frames, bg_frames=30),
                                        kw['close_kernel_size'], kw['open_kernel_size']),
                            min_area=kw['min_area'], min_solidity=0.1)
    dets_per_frame = detect_blobs(masks, min_area=kw['min_area'])

    # link ONCE over the whole strided sequence; history keys are strided indices
    strided = list(range(0, len(frames), NTH_FRAME))
    tracks = track_blobs([dets_per_frame[i] for i in strided],
                         max_dist=kw['max_dist'], max_age=MAX_AGE)
    at_index = {}
    for tr in tracks:
        for s in tr.history:
            at_index.setdefault(s, []).append(tr)

    rows = {N: {f: ([], []) for f in FEATURES} for N in depths}   # (person, distractor)
    hits = {N: {f: [0, 0] for f in FEATURES} for N in depths}     # (top-1 correct, evaluated)
    reachable, n_eval, cands, multi = 0, 0, [], 0

    for c in sorted(gt):
        if c % NTH_FRAME:
            continue                      # only frames that are strided samples
        s = c // NTH_FRAME
        run_lo = next(lo for lo, hi in runs if lo <= c <= hi)
        s_lo_run = -(-run_lo // NTH_FRAME)
        gx = (gt[c][0] + gt[c][2] / 2) * W
        gy = (gt[c][1] + gt[c][3] / 2) * H

        for N in depths:
            lo = max(s_lo_run, s - N + 1)
            entries = []
            for tr in at_index.get(s, []):
                hist = [(i, tr.history[i][0], tr.history[i][1])
                        for i in sorted(tr.history) if lo <= i <= s]
                if len(hist) < MIN_TRACK_LENGTH:
                    continue
                ax, ay = tr.history[s][0], tr.history[s][1]
                cx, cy = merged_center(dets_per_frame[c], ax, ay, kw['merge_radius'])
                is_person = np.hypot((cx - gx) / W, (cy - gy) / H) <= MAX_DIST_ERROR
                entries.append((trajectory_features(hist), is_person))
            if not entries:
                continue
            if N == depths[0]:
                n_eval += 1
                cands.append(len(entries))
                if any(p for _, p in entries):
                    reachable += 1
                if len(entries) >= 2:
                    multi += 1
            for f in FEATURES:
                pos = [e[f] for e, p in entries if p]
                neg = [e[f] for e, p in entries if not p]
                rows[N][f][0].extend(pos)
                rows[N][f][1].extend(neg)
                if pos:                       # hit@1 only where the person is findable
                    vals = [(e[f], p) for e, p in entries if np.isfinite(e[f])]
                    if vals:
                        hits[N][f][1] += 1
                        hits[N][f][0] += int(max(vals, key=lambda v: v[0])[1])
    if verbose:
        print(f"  evaluated {n_eval} strided gt frames, mean {np.mean(cands):.1f} candidate "
              f"tracks/frame ({100 * multi / max(n_eval, 1):.0f}% of frames have >=2, i.e. any "
              f"discrimination to do at all), person among candidates on "
              f"{100 * reachable / max(n_eval, 1):.1f}% (the hit@1 ceiling)")
    return rows, hits, depths


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='data/nfo_processed')
    p.add_argument('--seqs', nargs='*', default=['seq1_gt', 'seq2_gt', 'seq3_gt', 'seq4_gt'])
    p.add_argument('--depths', type=int, nargs='*', default=list(DEPTHS))
    p.add_argument('--out-dir', default='images/stream')
    a = p.parse_args()

    pooled = {N: {f: ([], []) for f in FEATURES} for N in a.depths}
    pooled_hits = {N: {f: [0, 0] for f in FEATURES} for N in a.depths}
    for seq in a.seqs:
        rows, hits, _ = evaluate_sequence(os.path.join(a.root, seq), a.depths)
        for N in a.depths:
            for f in FEATURES:
                pooled[N][f][0].extend(rows[N][f][0])
                pooled[N][f][1].extend(rows[N][f][1])
                pooled_hits[N][f][0] += hits[N][f][0]
                pooled_hits[N][f][1] += hits[N][f][1]

    n_pos = len(pooled[a.depths[0]][FEATURES[0]][0])
    n_neg = len(pooled[a.depths[0]][FEATURES[0]][1])
    print(f"\nAUC (person vs distractor tracks), pooled over {len(a.seqs)} sequences; "
          f"{n_pos} person samples vs {n_neg} distractor samples")
    print(f"{'feature':<14}" + "".join(f"{'N=' + str(N):>10}" for N in a.depths)
          + f"{'real span':>12}")
    aucs = {}
    for f in FEATURES:
        vals = [auc(*pooled[N][f]) for N in a.depths]
        aucs[f] = vals
        print(f"{f:<14}" + "".join(f"{v:>10.3f}" for v in vals)
              + f"{'':>12}")
    print(f"{'':14}" + "".join(f"{(N - 1) * NTH_FRAME:>10}" for N in a.depths) + "  frames")

    print(f"\nhit@1 (top-ranked track is the person, where the person is findable)")
    print(f"{'feature':<14}" + "".join(f"{'N=' + str(N):>10}" for N in a.depths) + f"{'n':>9}")
    for f in FEATURES:
        print(f"{f:<14}" + "".join(
            f"{pooled_hits[N][f][0] / max(pooled_hits[N][f][1], 1):>10.3f}" for N in a.depths)
              + f"{pooled_hits[a.depths[0]][f][1]:>9}")

    base = a.depths[0]
    print(f"\nverdict, against the N={base} do-nothing baseline:")
    for f in FEATURES:
        d = aucs[f][-1] - aucs[f][0]
        arrow = "improves" if d > 0.02 else "degrades" if d < -0.02 else "flat"
        print(f"  {f:<14} AUC {aucs[f][0]:.3f} -> {aucs[f][-1]:.3f}  ({d:+.3f}, {arrow})")
    print("  cur_score rising means accumulation helps the tracker as it stands; only the\n"
          "  normalised features rising means it helps only with a rewritten score.")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for f in FEATURES:
            ax.plot(a.depths, aucs[f], 'o-', label=f)
        ax.axhline(0.5, ls=':', c='grey', lw=1)
        ax.set_xlabel('lookbehind N (strided samples; real span = 2(N-1) frames)')
        ax.set_ylabel('AUC, person vs distractor tracks')
        ax.set_title('Does accumulating track history discriminate the person? (NFO, GT-scored)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = f'{a.out_dir}/21_lookbehind_auc.png'
        plt.savefig(out, dpi=110)
        print(f"\nwrote {out}")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
