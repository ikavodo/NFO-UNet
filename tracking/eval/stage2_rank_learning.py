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
import sys

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from tracking.core.blob_tracker import detect_blobs, track_blobs, score_and_fit
from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.track_window import position_from_track
from tracking.eval.eval_nfo import (IN_DIR, SEQS, SPAN, NTH_FRAME, BG_FRAMES, MAX_DIST,
                                    MERGE_RADIUS, EXPECTED_HEIGHT, parse_normalized_bbs)

HIT = 0.1
FEATURES = ['span_frac', 'height_rel', 'disp_rel', 'straightness', 'resid_rel', 'size_stability']


def track_features(cand, n_frames, all_cands):
    """Dimensionless, within-window-normalized description of one candidate track."""
    frames = cand['frames']
    hist = cand['history']
    xs = np.array([hist[f][0] for f in frames])
    ys = np.array([hist[f][1] for f in frames])
    hs = np.array([hist[f][2] for f in frames if hist[f][2] is not None], dtype=float)

    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) if len(xs) > 1 else 0.0
    med_h = np.median([c['mean_height'] for c in all_cands if c['mean_height']]) or 1.0
    med_disp = np.median([c['net_disp'] for c in all_cands]) or 1.0
    return [
        len(frames) / max(n_frames, 1),                                  # span_frac
        (cand['mean_height'] or 0.0) / max(med_h, 1e-6),                 # height_rel
        cand['net_disp'] / max(med_disp, 1e-6),                          # disp_rel
        cand['net_disp'] / max(path, 1e-6),                              # straightness
        cand['resid_std'] / max(cand['net_disp'], 1e-6),                 # resid_rel
        float(hs.std() / max(hs.mean(), 1e-6)) if len(hs) else 0.0,      # size_stability
    ]


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
        dets = detect_blobs(masks[idx])
        tracks = track_blobs(dets, max_dist=MAX_DIST)
        cands = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT, return_all=True)
        if not cands:
            continue
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

    pooled = {k: [] for k in ('baseline', 'learned', 'oracle')}
    print()
    for held in SEQS:
        train = [w for s in SEQS if s != held for w in data[s]]
        D, y = pairs_from(train)
        model = LogisticRegression(max_iter=5000, fit_intercept=False).fit(D, y)
        w_vec = model.coef_[0]

        base, learn, orac = [], [], []
        for win in data[held]:
            base.append(win['resid'][win['baseline_idx']])
            learn.append(win['resid'][int(np.argmax(win['X'] @ w_vec))])
            orac.append(win['resid'].min())
        pooled['baseline'] += base; pooled['learned'] += learn; pooled['oracle'] += orac
        print(f"held-out {held} ({len(base)} windows, {len(D)} training pairs)")
        for label, vals in (('baseline', base), ('learned', learn), ('oracle', orac)):
            print("   " + summarize(vals, label))

    print("\nPOOLED over all four held-out sequences:")
    for label in ('baseline', 'learned', 'oracle'):
        print("   " + summarize(pooled[label], label))

    b = np.array(pooled['baseline']); l = np.array(pooled['learned']); o = np.array(pooled['oracle'])
    print(f"\nheadroom (baseline -> oracle): mean {b.mean():.4f} -> {o.mean():.4f}, "
          f"hit {100 * (b < HIT).mean():.1f}% -> {100 * (o < HIT).mean():.1f}%")
    print(f"captured by the learned ranker: mean {b.mean():.4f} -> {l.mean():.4f}, "
          f"hit {100 * (b < HIT).mean():.1f}% -> {100 * (l < HIT).mean():.1f}%")
    frac = (b.mean() - l.mean()) / max(b.mean() - o.mean(), 1e-9)
    print(f"fraction of available headroom captured: {100 * frac:.0f}%")
    print("\n-> " + ("GO: learned ranker beats the hand-picked formula out of sample"
                     if l.mean() < b.mean() and (b < HIT).mean() < (l < HIT).mean()
                     else "NO-GO: hand-picked formula is not beaten out of sample"))


if __name__ == '__main__':
    main()
