"""Stage 2: can a learned ranker over dimensionless track features beat score_and_fit's
hand-picked formula at picking the right candidate track?

Runs on REAL NFO, not synthetic KTH, for a measured reason (docs/scale_generalization_plan.md,
Stage 1 result): with the occluder calibrated to NFO and a distractor injected, only 3-14% of
KTH's candidate detections are non-person - its background is clean, so it contains almost no
negatives and cannot pose the "which object is the person" question at all. NFO is full of real
clutter, which is exactly why its shape term is worth taking p90 from 0.59 to 0.10.

Because NFO is also the only test set, everything here is leave-one-sequence-out: fit on three
sequences, evaluate on the fourth, rotate. Nothing is tuned and tested on the same footage.

The tracker is untouched and runs with NFO's own hand-tuned constants, so the baseline
reproduces the published 0.0698 mean residual and the learned ranker sees exactly the same
candidate sets. Only the choice among candidates differs.

Features (6) are dimensionless and normalized WITHIN the window - a track's height against the
median candidate height in its own window, etc. - so no person-height estimate appears anywhere
(F4: the explicit estimator does not transfer to NFO).

Reports, per held-out sequence:
  baseline   score_and_fit's own winner (the thing to beat)
  learned    argmax of the fitted ranker
  oracle     the best available candidate, i.e. the headroom. If oracle is barely better than
             baseline, there is nothing to learn and no ranker can help.

    python -m tracking.eval.stage2_rank_learning
"""
import os

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from tracking.core.blob_tracker import detect_blobs, track_blobs, score_and_fit
from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.track_window import position_from_track
from tracking.eval.eval_nfo import (IN_DIR, SEQS, SPAN, NTH_FRAME, BG_FRAMES, MAX_DIST,
                                    MERGE_RADIUS, EXPECTED_HEIGHT, parse_normalized_bbs)

HIT = 0.1
BASE_FEATURES = ['span_frac', 'height_rel', 'disp_rel', 'straightness', 'resid_rel',
                 'size_stability']
# Shape/gait block. True gait *periodicity* is not measurable here and deliberately is not
# attempted: a window is 7 sampled frames spanning 13 real ones, while a walking cycle is
# ~25 frames at NFO's frame rate, so a window sees about half a cycle. What IS measurable is
# how the silhouette's proportions vary over the window, which is the same signal minus the
# frequency estimate: a person's aspect ratio wobbles about a person-like value, a
# foliage-revealed background patch does not.
GAIT_FEATURES = ['aspect_mean', 'aspect_cv', 'width_cv', 'aspect_rel']
# Appearance block, from the raw greyscale under each blob's own pixels - previously discarded
# entirely by detect_blobs. Normalized within the window so overall scene brightness and
# contrast cancel.
APP_FEATURES = ['app_mean_rel', 'app_std_rel', 'app_consistency']
# score_and_fit's own score, handed to the ranker as evidence rather than as a hard fallback
# branch. Motivation: on the one NFO sequence where the formula nearly saturates (seq1,
# baseline 98.5% against an oracle of 99.3% - 0.8pp of headroom), every learned feature block
# hurts, because there is nothing to win and only variance to add. A count-based "few
# candidates -> use the formula" guard would fix that with a threshold fitted to a single
# sequence; giving the model the formula's own opinion instead needs no constant and lets it
# learn when to defer. Both features are PER-CANDIDATE: a per-window quantity such as the
# top-to-runner-up margin would cancel in a pairwise-difference ranker.
#   base_score_rel  the candidate's score over the window's best score (1.0 = the formula's pick)
#   base_rank_frac  its rank by score over the number of candidates (small = the formula likes it)
# Note the score is a nonlinear combination of quantities the other features already carry
# (span x net_disp / (1 + resid_std), times a Gaussian in height), so this mainly gives a
# linear model access to that nonlinearity.
BASE_SCORE_FEATURES = ['base_score_rel', 'base_rank_frac']
FEATURES = BASE_FEATURES + GAIT_FEATURES + APP_FEATURES + BASE_SCORE_FEATURES


def _track_arrays(cand):
    """(xs, ys, heights, widths, app_means, app_stds) over the track's own frames."""
    frames, hist = cand['frames'], cand['history']
    cols = [[hist[f][k] if len(hist[f]) > k else None for f in frames] for k in range(6)]
    out = []
    for c in cols:
        v = np.array([np.nan if x is None else float(x) for x in c], dtype=float)
        out.append(v)
    return out


def _cv(v):
    """Coefficient of variation - dimensionless spread. 0 when undefined."""
    v = v[np.isfinite(v)]
    if len(v) < 2 or abs(v.mean()) < 1e-6:
        return 0.0
    return float(v.std() / abs(v.mean()))


def _nanmed(vals, default=1.0):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    m = float(np.median(vals)) if vals else default
    return m if abs(m) > 1e-6 else default


def track_features(cand, n_frames, all_cands):
    """Dimensionless, within-window-normalized description of one candidate track."""
    xs, ys, hs, ws, ams, asds = _track_arrays(cand)
    frames = cand['frames']

    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) if len(xs) > 1 else 0.0
    med_h = _nanmed([c['mean_height'] for c in all_cands])
    med_disp = _nanmed([c['net_disp'] for c in all_cands])

    aspect = ws / np.where(np.isfinite(hs) & (hs > 0), hs, np.nan)
    aspect_mean = float(np.nanmean(aspect)) if np.isfinite(aspect).any() else 0.0
    app_mean = float(np.nanmean(ams)) if np.isfinite(ams).any() else 0.0
    app_std = float(np.nanmean(asds)) if np.isfinite(asds).any() else 0.0

    # within-window references for the new blocks, computed once per candidate set
    med_aspect = _nanmed([c.get('_aspect_mean') for c in all_cands], default=aspect_mean or 1.0)
    med_app = _nanmed([c.get('_app_mean') for c in all_cands], default=app_mean or 1.0)
    med_app_std = _nanmed([c.get('_app_std') for c in all_cands], default=app_std or 1.0)

    return [
        len(frames) / max(n_frames, 1),                                  # span_frac
        (cand['mean_height'] or 0.0) / med_h,                            # height_rel
        cand['net_disp'] / med_disp,                                     # disp_rel
        cand['net_disp'] / max(path, 1e-6),                              # straightness
        cand['resid_std'] / max(cand['net_disp'], 1e-6),                 # resid_rel
        _cv(hs),                                                         # size_stability
        aspect_mean,                                                     # aspect_mean
        _cv(aspect),                                                     # aspect_cv
        _cv(ws),                                                         # width_cv
        aspect_mean / med_aspect,                                        # aspect_rel
        app_mean / med_app,                                              # app_mean_rel
        app_std / med_app_std,                                           # app_std_rel
        _cv(ams),                                                        # app_consistency
        cand['_score_rel'],                                              # base_score_rel
        cand['_rank_frac'],                                              # base_rank_frac
    ]


def annotate_candidates(cands):
    """Pre-compute each candidate's raw aspect/appearance means so the within-window medians
    used for normalization are available while featurizing any single candidate."""
    best_score = max([c['score'] for c in cands] + [0.0])
    for rank, c in enumerate(cands, start=1):   # score_and_fit returns candidates best-first
        _, _, hs, ws, ams, asds = _track_arrays(c)
        aspect = ws / np.where(np.isfinite(hs) & (hs > 0), hs, np.nan)
        c['_aspect_mean'] = float(np.nanmean(aspect)) if np.isfinite(aspect).any() else None
        c['_app_mean'] = float(np.nanmean(ams)) if np.isfinite(ams).any() else None
        c['_app_std'] = float(np.nanmean(asds)) if np.isfinite(asds).any() else None
        c['_score_rel'] = c['score'] / best_score if best_score > 1e-12 else 0.0
        c['_rank_frac'] = rank / len(cands)


def collect_sequence(seq):
    """-> list of windows, each dict(X [k,6], resid [k], baseline_idx). One entry per window
    that produced at least one candidate track."""
    seq_in = os.path.join(IN_DIR, seq)
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    norm_file = next(f for f in os.listdir(seq_in)
                     if f != 'groundtruth.txt' and f.startswith('groundtruth'))
    raw = parse_normalized_bbs(os.path.join(seq_in, norm_file))
    n = min(len(jpgs), len(raw))
    frames_all = np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(n)], axis=0)
    H, W = frames_all.shape[1:]
    centers = [c for c in range(SPAN, n - SPAN) if raw[c] is not None]

    masks = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))

    windows = []
    for c in centers:
        idx = list(range(c - SPAN, c + SPAN + 1, NTH_FRAME))
        dets = detect_blobs(masks[idx], raw_frames=frames_all[idx])
        tracks = track_blobs(dets, max_dist=MAX_DIST)
        cands = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT, return_all=True)
        if not cands:
            continue
        annotate_candidates(cands)
        gt = raw[c]
        gx, gy = (gt[0] + gt[2] / 2) * W, (gt[1] + gt[3] / 2) * H
        X, resid = [], []
        for cand in cands:
            pos = position_from_track(cand, dets, MERGE_RADIUS)
            resid.append(float(np.hypot((pos['x'] - gx) / W, (pos['y'] - gy) / H)))
            X.append(track_features(cand, len(idx), cands))
        windows.append(dict(X=np.array(X, dtype=float), resid=np.array(resid),
                            baseline_idx=0))  # score_and_fit returns best-first
    return windows


def pairs_from(windows):
    """RankNet-style: one row per (better, worse) candidate pair, as a feature difference."""
    D, y = [], []
    for w in windows:
        best = int(np.argmin(w['resid']))
        for j in range(len(w['resid'])):
            if j == best or w['resid'][j] <= w['resid'][best] + 1e-9:
                continue
            D.append(w['X'][best] - w['X'][j]); y.append(1)
            D.append(w['X'][j] - w['X'][best]); y.append(0)
    return np.array(D, dtype=float), np.array(y)


def summarize(resids, label):
    r = np.array(resids)
    return (f"{label:>9}: mean={r.mean():.4f} median={np.median(r):.4f} "
            f"p90={np.percentile(r, 90):.4f} hit@{HIT}={100 * (r < HIT).mean():.1f}%")


def main():
    data = {}
    for seq in SEQS:
        data[seq] = collect_sequence(seq)
        ncand = np.mean([len(w['resid']) for w in data[seq]])
        print(f"  {seq}: {len(data[seq])} windows, {ncand:.1f} candidate tracks per window")

    idx = {name: i for i, name in enumerate(FEATURES)}
    # app_mean_rel encodes a brightness POLARITY ("the person is darker than the other
    # candidates"), which may be true of NFO's four sequences and false of other footage.
    # Leave-one-sequence-out cannot detect a shortcut shared by every sequence in one scene,
    # so the *_nopol blocks drop it and keep only the polarity-free appearance features
    # (texture amount and temporal stability). The gap between 'all' and 'all_nopol' measures
    # how much of the win rests on that shortcut.
    APP_NOPOL = [f for f in APP_FEATURES if f != 'app_mean_rel']
    ALL_NOPOL = BASE_FEATURES + GAIT_FEATURES + APP_NOPOL
    BLOCKS = {
        'base': BASE_FEATURES,
        'base+gait': BASE_FEATURES + GAIT_FEATURES,
        'base+app_nopol': BASE_FEATURES + APP_NOPOL,
        'all_nopol': ALL_NOPOL,
        'all': BASE_FEATURES + GAIT_FEATURES + APP_FEATURES,
        # + the formula's own score as evidence, on top of each of the two headline sets
        'all_nopol+bscore': ALL_NOPOL + BASE_SCORE_FEATURES,
        'all+bscore': FEATURES,
    }
    pooled = {k: [] for k in ('baseline', 'oracle', *BLOCKS)}
    weights = {}
    print()
    for held in SEQS:
        train = [w for s in SEQS if s != held for w in data[s]]
        D_full, y = pairs_from(train)
        base, orac = [], []
        for win in data[held]:
            base.append(win['resid'][win['baseline_idx']])
            orac.append(win['resid'].min())
        pooled['baseline'] += base; pooled['oracle'] += orac
        print(f"held-out {held} ({len(base)} windows, {len(D_full)} training pairs)")
        print("   " + summarize(base, 'baseline'))
        for bname, names in BLOCKS.items():
            cols = [idx[n] for n in names]
            D = D_full[:, cols]
            sd = D.std(axis=0)
            sd[sd < 1e-9] = 1.0          # differences are mean-zero by construction
            model = LogisticRegression(max_iter=5000, fit_intercept=False).fit(D / sd, y)
            w_vec = model.coef_[0] / sd
            weights.setdefault(bname, []).append(dict(zip(names, model.coef_[0])))
            vals = [win['resid'][int(np.argmax(win['X'][:, cols] @ w_vec))] for win in data[held]]
            pooled[bname] += vals
            print("   " + summarize(vals, bname))
        print("   " + summarize(orac, 'oracle'))

    print("\nPOOLED over all four held-out sequences:")
    for label in ('baseline', *BLOCKS, 'oracle'):
        print("   " + summarize(pooled[label], label))

    print("\nstandardized weights for the full feature set (mean over folds, "
          "positive = evidence this candidate is the right one):")
    mean_w = {k: float(np.mean([f[k] for f in weights['all+bscore']])) for k in FEATURES}
    for k, v in sorted(mean_w.items(), key=lambda t: -abs(t[1])):
        block = ('gait' if k in GAIT_FEATURES else 'app' if k in APP_FEATURES
                 else 'score' if k in BASE_SCORE_FEATURES else 'base')
        print(f"  {k:>16} [{block:>4}]: {v:+.3f}")

    b = np.array(pooled['baseline']); l = np.array(pooled['all+bscore'])
    o = np.array(pooled['oracle'])
    print(f"\nheadroom (baseline -> oracle): mean {b.mean():.4f} -> {o.mean():.4f}, "
          f"hit {100 * (b < HIT).mean():.1f}% -> {100 * (o < HIT).mean():.1f}%")
    print(f"captured by the learned ranker: mean {b.mean():.4f} -> {l.mean():.4f}, "
          f"hit {100 * (b < HIT).mean():.1f}% -> {100 * (l < HIT).mean():.1f}%")
    print("\nfraction of the available ranking headroom captured, by feature block:")
    for bname in BLOCKS:
        v = np.array(pooled[bname])
        frac_mean = (b.mean() - v.mean()) / max(b.mean() - o.mean(), 1e-9)
        frac_hit = ((v < HIT).mean() - (b < HIT).mean()) / \
                   max((o < HIT).mean() - (b < HIT).mean(), 1e-9)
        print(f"  {bname:>10}: {100 * frac_mean:>5.0f}% of the mean-residual headroom, "
              f"{100 * frac_hit:>5.0f}% of the hit-rate headroom")
    print("\n-> " + ("GO: learned ranker beats the hand-picked formula out of sample"
                     if l.mean() < b.mean() and (b < HIT).mean() < (l < HIT).mean()
                     else "NO-GO: hand-picked formula is not beaten out of sample"))


if __name__ == '__main__':
    main()
