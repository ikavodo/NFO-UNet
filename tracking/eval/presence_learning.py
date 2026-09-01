"""Can a learned rule tell "a person is in frame" from "only the plant is"?

    python -m tracking.eval.presence_learning

This is the PRESENCE question, not the ranking question, and they are different problems with
different headroom:

  RANKING - which of several candidate tracks is the person? Already measured in
  tracking/eval/stage2_rank_learning.py: on NFO the oracle is 98.8% hit@0.1 against a 90.0%
  baseline, so the right candidate is nearly always present and merely mis-picked. That is real
  headroom and a learned ranker captured 13-52% of it.

  PRESENCE - is any candidate a person at all? score_and_fit has no null output: it returns its
  best candidate unconditionally. This is where the false positives on person-absent frames come
  from, and no existing script targets it.

WHY THIS IS THE HARDER PROBLEM TO LEARN, stated before the numbers so the numbers can refute it.
The scoring protocol here is leave-one-CLIP-out over three clips, which sounds like three folds
but is not three independent samples. Frames at 24 fps are massively correlated: one walk-past is
~120 near-identical frames. The real unit of independence is the WALK, and there are six of them
across all three clips, against nine person-absent stretches - all in one room, one plant, one
subject, one camera position. A model fitted on that learns THIS plant, not plants. So the honest
question is not "does a fitted model beat the hand formula on held-out frames" (it will, by
memorising the scene) but "does it beat it on a held-out CLIP", and even that shares the scene.

Features are dimensionless and normalised by the measured person height, following the same
principle as stage1/stage2: nothing here may carry an absolute pixel constant, or it cannot
transfer to another scene even in principle.

Baseline: the tracker's own score, thresholded. That is what a deployment would do today, and
any learned rule has to beat it to be worth the machinery.

Reported per held-out clip:
  auc_score     AUC of log(1+score) alone - the baseline discriminator
  auc_best1     AUC of the best single feature chosen ON THE TRAINING CLIPS
  auc_learned   AUC of L2-regularised logistic regression fitted on the training clips
  fpr@90        fraction of person-ABSENT frames accepted at a threshold keeping 90% of
                person-present frames - the operationally meaningful number
"""
import argparse

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from tracking.stream.stream import (StreamPipeline, bootstrap_person_height, frames_from_source)

# person-in-frame intervals, supplied by the person who was in them
LABELS = {
    'walk_noisy1': [(65, 195), (353, 453)],
    'walk_noisy2': [(70, 160), (270, 373)],
    'ido_walk':    [(31, 194), (278, 391)],
}
FEATURES = ['log_score', 'box_h', 'box_w', 'aspect', 'net_disp', 'resid', 'speed',
            'span', 'mean_h', 'extrap', 'support']


def features_for(clip: str, scale: float = 0.5):
    """One feature row per emission, plus the present/absent label and the frame index."""
    frames = list(frames_from_source(f'data/{clip}.mkv', scale))
    h = bootstrap_person_height(np.stack(frames[:240]))
    pipe = StreamPipeline(h)
    rows, labels, idx = [], [], []
    present = LABELS[clip]
    for f in frames:
        r = pipe.step(f)
        if r is None:
            continue
        w = r.winner
        if r.box is None or w is None:
            bh = bw = nd = rs = sp = mh = su = 0.0      # no candidate at all is itself evidence
        else:
            bh = (r.box[3] - r.box[1]) / h
            bw = (r.box[2] - r.box[0]) / h
            nd = w['net_disp'] / h
            rs = w['resid_std'] / h
            sp = w['span'] / 7.0
            mh = (w['mean_height'] or 0.0) / h
            su = len(w['frames']) / 7.0
        rows.append([np.log1p(max(r.score or 0.0, 0.0)), bh, bw,
                     bw / bh if bh > 0 else 0.0, nd, rs,
                     abs(w['vx']) / h if w else 0.0, sp, mh,
                     float(r.extrapolated), su])
        labels.append(int(any(a <= r.frame_index <= b for a, b in present)))
        idx.append(r.frame_index)
    return np.array(rows, float), np.array(labels, int), np.array(idx, int)


def auc(scores, y) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fpr_at_recall(scores, y, recall: float = 0.7) -> float:
    """Fraction of person-ABSENT frames accepted at the threshold keeping `recall` of the
    person-present ones - how often it would report a person in an empty room, at a sensitivity
    you would ship.

    RECALL MUST BE BELOW THE DETECTION CEILING or this number is meaningless. A present frame
    with no track at all scores 0, so if more than (1-recall) of present frames are untracked the
    threshold lands at 0 and every absent frame passes, giving a spurious fpr of exactly 1.000.
    A first version of this asked for recall=0.9 on ido_walk, where only 81% of present frames
    carry a box, and got precisely that artifact. recall_ceiling() below reports the limit.
    """
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    thr = np.quantile(pos, 1.0 - recall)
    return float((neg >= thr).mean())


def recall_ceiling(scores, y) -> float:
    """Fraction of person-present frames with any track at all. No threshold can exceed it."""
    return float((scores[y == 1] > 0).mean())


def recall_at_fpr(scores, y, fpr: float = 0.05) -> float:
    """The dual, and always well posed: what recall is available at a fixed false-positive rate."""
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    thr = np.quantile(neg, 1.0 - fpr)
    return float((pos > thr).mean())


def fit_logistic(X, y, l2: float = 1.0):
    """L2-regularised logistic regression by direct minimisation - scipy only, because sklearn
    is not installed in any environment on this machine (which is also why
    stage2_rank_learning.py cannot currently run)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])

    def loss(w):
        z = np.clip(Z @ w, -30, 30)
        ll = np.mean(np.log1p(np.exp(z)) - y * z)
        return ll + l2 * np.sum(w[:-1] ** 2) / len(w)

    w = minimize(loss, np.zeros(Z.shape[1]), method='L-BFGS-B').x
    return lambda Xn: (np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ w), w


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--l2', type=float, default=1.0)
    p.add_argument('--depth-sweep', action='store_true',
                   help='instead of the fit, sweep lookbehind depth and report presence AUC')
    a = p.parse_args()

    if a.depth_sweep:
        depth_sweep()
        return

    data = {c: features_for(c) for c in LABELS}
    for c, (X, y, _) in data.items():
        runs = 1 + int(np.sum(np.diff(y) != 0) / 2)
        print(f'{c:13s} {len(y):4d} emissions, {y.sum():4d} present / {len(y)-y.sum():4d} absent, '
              f'{len(LABELS[c])} walks')

    n_walks = sum(len(v) for v in LABELS.values())
    print(f'\nEFFECTIVE SAMPLE SIZE: {n_walks} independent walks and '
          f'{sum(len(v)+1 for v in LABELS.values())} absent stretches, all one room / plant / '
          f'subject / camera. Frames are not samples.\n')

    print(f"{'held-out clip':15s}{'ceiling':>9}{'auc_score':>11}{'auc_best1':>11}"
          f"{'auc_learned':>13}{'fpr@70 sc':>11}{'fpr@70 lrn':>12}"
          f"{'rec@fpr5 sc':>13}{'rec@fpr5 lrn':>14}")
    agg = {}
    for held in data:
        Xtr = np.vstack([data[c][0] for c in data if c != held])
        ytr = np.concatenate([data[c][1] for c in data if c != held])
        Xte, yte, _ = data[held]
        # best single feature, chosen on the TRAINING clips only
        aucs_tr = [auc(Xtr[:, j], ytr) for j in range(Xtr.shape[1])]
        j = int(np.argmax([max(v, 1 - v) for v in aucs_tr]))
        sgn = 1.0 if aucs_tr[j] >= 0.5 else -1.0
        pred, _ = fit_logistic(Xtr, ytr, a.l2)
        s_learn, s_score, s_best = pred(Xte), Xte[:, 0], sgn * Xte[:, j]
        row = (recall_ceiling(s_score, yte), auc(s_score, yte), auc(s_best, yte),
               auc(s_learn, yte), fpr_at_recall(s_score, yte), fpr_at_recall(s_learn, yte),
               recall_at_fpr(s_score, yte), recall_at_fpr(s_learn, yte))
        agg[held] = row
        print(f'{held:15s}{row[0]:9.3f}{row[1]:11.3f}{row[2]:11.3f}{row[3]:13.3f}'
              f'{row[4]:11.3f}{row[5]:12.3f}{row[6]:13.3f}{row[7]:14.3f}'
              f'  (best1={FEATURES[j]})')
    m = np.mean(list(agg.values()), axis=0)
    print(f'{"mean":15s}{m[0]:9.3f}{m[1]:11.3f}{m[2]:11.3f}{m[3]:13.3f}'
          f'{m[4]:11.3f}{m[5]:12.3f}{m[6]:13.3f}{m[7]:14.3f}')



# ---------------------------------------------------------------------------------------------
# Does a LONGER BLOB TRAJECTORY make the person/plant distinction easier?
#
# Two different things get confused under "enlarge the window", and only one is cheap:
#
#   LOOKAHEAD - growing SEQ_SIZE grows SPAN, so it costs latency frame for frame, and it
#   invalidates the ALPHA_* coefficients (calibrated at SEQ_SIZE=7) and the bit-exact parity with
#   track_windows_in_sequence.
#
#   LOOKBEHIND - a longer PAST trajectory for the same emitted frame. Free in latency, because
#   the person already walked. Needs persistent tracks, which the current per-window engine does
#   not keep - but it can be measured offline without writing that engine, by linking tracks once
#   over the whole sequence and TRUNCATING each candidate's history to a trailing window of N.
#
# The mechanism that should make it work: fan-driven foliage OSCILLATES, so its net displacement
# is bounded by its amplitude for any N, while a walking person TRANSLATES, so theirs grows like
# N. The ratio therefore grows linearly in N, and at N=7 (12 real frames, 0.5s) a leaf has not
# completed a period so the two are indistinguishable. Falsified if AUC does not rise with N.
def depth_sweep(depths=(7, 15, 31, 51), nth_frame=2, max_age=6, min_len=3):
    from tracking.core.blob_tracker import _Track, detect_blobs, track_blobs
    from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask
    from tracking.core.track_sequence import scale_relative_params
    from tracking.eval.lookbehind_discrimination import trajectory_features

    feats = ('cur_score', 'net_disp', 'straightness', 'msd_alpha')
    out = {}
    for clip, present in LABELS.items():
        frames = np.stack(list(frames_from_source(f'data/{clip}.mkv', 0.5)))
        h = bootstrap_person_height(frames[:240])
        kw, kalman = scale_relative_params(h)
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = kalman
        masks = filter_by_shape(refine_mask(foreground_mask(frames, bg_frames=30),
                                            kw['close_kernel_size'], kw['open_kernel_size']),
                                min_area=kw['min_area'], min_solidity=0.1)
        dets = detect_blobs(masks, min_area=kw['min_area'])
        strided = list(range(0, len(frames), nth_frame))
        tracks = track_blobs([dets[i] for i in strided], max_dist=kw['max_dist'], max_age=max_age)
        at = {}
        for tr in tracks:
            for s in tr.history:
                at.setdefault(s, []).append(tr)
        for N in depths:
            rows, lab = [], []
            for s, f_idx in enumerate(strided):
                cands = []
                for tr in at.get(s, []):
                    hist = [(i, tr.history[i][0], tr.history[i][1])
                            for i in sorted(tr.history) if s - N + 1 <= i <= s]
                    if len(hist) >= min_len:
                        cands.append(trajectory_features(hist))
                if not cands:
                    continue
                best = max(cands, key=lambda d: d['cur_score'])          # the tracker's own rule
                rows.append([best[f] / (h if f == 'net_disp' else 1.0) for f in feats])
                lab.append(int(any(a <= f_idx <= b for a, b in present)))
            X, y = np.array(rows, float), np.array(lab, int)
            out[(clip, N)] = [auc(X[:, j], y) for j in range(len(feats))]
        del frames, masks
    print(f"\nAUC(person-present vs absent) of the WINNING track's features, by lookbehind depth")
    print(f"{'clip':14s}{'N':>4}{'real span':>11}" + ''.join(f'{f:>14}' for f in feats))
    for clip in LABELS:
        for N in depths:
            v = out[(clip, N)]
            print(f'{clip if N == depths[0] else "":14s}{N:>4}{(N-1)*nth_frame:>11}'
                  + ''.join(f'{x:>14.3f}' for x in v))
    return out

if __name__ == '__main__':
    main()
