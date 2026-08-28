"""Stage 1 of the learned-tracker plan: can a learned match rule hold its operating point
across person scale better than the best non-learned dimensionless rule?

Deliberately offline and local - no tracker changes, no training pipeline, no new data. It
targets the association gate because that is where 70-78% of the measured scale sensitivity
lives (docs/scale_generalization_plan.md, F2), and because the gate's failure mode is fatal
rather than degrading (F1): too tight and no tracks form at all.

The decision in isolation: given a detection in frame t and a detection in frame t+stride,
are they the same object? That is exactly what track_blobs's `cost <= max_dist` test decides,
so it can be studied without touching the tracker.

Labels: positive iff BOTH detections' centroids fall inside their frame's ground-truth person
box. Note this deliberately counts a head-at-t / legs-at-t+stride pair as positive: the
tracker follows fragments and merges later, so keeping a person's track alive across a
fragment switch is a correct association for the gate's purposes.

Features (7) are all dimensionless and NONE uses a person-height estimate - the point being
that a within-frame-normalized rule needs no scale measurement at all, which matters because
the explicit estimator is the component that failed to transfer to real NFO (F4).

Protocol: fit on the 1x bucket ONLY, pick the decision threshold on the 1x bucket only, then
apply that frozen threshold at 0.5x and 2x.

Metric: per-pair error rates are the WRONG scoring (a first version of this script used them
and produced a misleading verdict). The label marks every person-to-person pair positive,
head-to-legs included, but the tracker does not need to accept all of them - it needs, for each
person detection, at least one accepted match that is also the person. So:
  loss_rate      fraction of person detections with NO accepted person match - what kills
                 tracks, and per F1 the failure that actually matters
  contamination  fraction of accepted matches out of person detections landing on a non-person
Thresholds are chosen to equalize contamination on 1x, so loss rate is compared at a matched
cost.

Baselines, both given the same 1x-only tuning budget:
  fixed-h     today's rule, distance < ALPHA_MAX_DIST * person_height, where person_height
              comes from GROUND TRUTH (i.e. the baseline is handed a perfect scale estimate -
              deliberately generous)
  median-nn   the scale-free alternative, distance < k * median nearest-neighbour distance
              among frame t's detections, k tuned on 1x

Go/no-go: continue to stage 2 only if the learned rule's held-out-scale FNR is materially
better than BOTH baselines'. If median-nn already transfers flat, implement median-nn in the
tracker (a ~5-line change) and stop - that would be a win with no learned component at all.

    python -m tracking.eval.stage1_gate_learning [n_sequences]
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from tracking.core.blob_tracker import detect_blobs
from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.track_sequence import scale_relative_params, ALPHA_MAX_DIST
from tracking.eval import kill_test_scale as kt

TARGET_CONTAMINATION = 0.10   # operating point, chosen on the 1x bucket then frozen
FEATURES = ['d_over_median_nn', 'd_over_own_height', 'rel_dheight', 'rel_darea',
            'rel_daspect', 'dist_rank', 'is_mutual']


def _nn_distances(dets):
    """Nearest-neighbour distance for each detection within its own frame."""
    if len(dets) < 2:
        return None
    pts = np.array([[d['x'], d['y']] for d in dets])
    dm = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(dm, np.inf)
    return dm.min(axis=1)


def _in_box(det, box, H, W):
    x1, y1 = box[0] * W, box[1] * H
    return x1 <= det['x'] <= x1 + box[2] * W and y1 <= det['y'] <= y1 + box[3] * H


def pair_features(detections, boxes, H, W, stride, seq_i=0):
    """-> (X [n,7], y [n], dists [n], meta [n,4]). One row per candidate pair.

    meta columns: source-group id (unique per (sequence, frame, source detection)),
    source-is-person, destination-is-person, unused. The group id is what lets the caller
    score the decision the way the tracker experiences it - per source detection, did ANY
    accepted match land on the person - rather than per candidate pair."""
    X, y, dists, meta = [], [], [], []
    for t in range(len(detections) - stride):
        u = t + stride
        if t not in boxes or u not in boxes or not detections[t] or not detections[u]:
            continue
        a, b = detections[t], detections[u]
        nn = _nn_distances(a)
        pts_b = np.array([[d['x'], d['y']] for d in b])
        for i, di in enumerate(a):
            dd = np.hypot(pts_b[:, 0] - di['x'], pts_b[:, 1] - di['y'])
            order = np.argsort(dd)
            rank_of = {j: r + 1 for r, j in enumerate(order)}
            # scale reference from the frame's own geometry: the typical distance between
            # detections, falling back to this blob's own height when it is alone
            hi = di['bbox'][3] - di['bbox'][1]
            ref = float(np.median(nn)) if nn is not None else float(hi)
            ref = max(ref, 1e-6)
            for j, dj in enumerate(b):
                hj = dj['bbox'][3] - dj['bbox'][1]
                wi = di['bbox'][2] - di['bbox'][0]
                wj = dj['bbox'][2] - dj['bbox'][0]
                aspi, aspj = wi / max(hi, 1e-6), wj / max(hj, 1e-6)
                # is di also the nearest detection in frame t to dj?
                back = np.hypot(np.array([d['x'] for d in a]) - dj['x'],
                                np.array([d['y'] for d in a]) - dj['y'])
                mutual = float(int(np.argmin(back) == i and rank_of[j] == 1))
                X.append([
                    dd[j] / ref,
                    dd[j] / max(hi, 1e-6),
                    abs(hj - hi) / max((hi + hj) / 2, 1e-6),
                    abs(dj['area'] - di['area']) / max((di['area'] + dj['area']) / 2, 1e-6),
                    abs(aspj - aspi) / max((aspi + aspj) / 2, 1e-6),
                    float(rank_of[j]),
                    mutual,
                ])
                src_p = int(_in_box(di, boxes[t], H, W))
                dst_p = int(_in_box(dj, boxes[u], H, W))
                y.append(int(src_p and dst_p))
                dists.append(dd[j])
                meta.append([seq_i * 10 ** 7 + t * 10 ** 3 + i, src_p, dst_p, 0])
    return (np.array(X, dtype=float), np.array(y), np.array(dists),
            np.array(meta, dtype=np.int64).reshape(-1, 4))


def build_scale(seqs, scale, distractor_px=0.0):
    """-> (X, y, dists, gt person height per row, meta)."""
    Xs, ys, ds, refs, metas = [], [], [], [], []
    for seq_i, name in enumerate(seqs):
        frames_native, boxes = kt.load_sequence(name)
        built = kt.build_bucket(frames_native, boxes, scale, seed=seq_i,
                                distractor_px=distractor_px * scale)
        frames, boxes_b = built['frames'], built['boxes']
        H, W = frames.shape[1:]
        person_h = float(np.mean([b[3] * H for b in boxes_b.values()]))
        pre, _ = scale_relative_params(person_h)
        first = next(c for c in sorted(boxes_b) if c >= kt.SPAN)
        masks = filter_by_shape(refine_mask(foreground_mask(frames, bg_frames=int(max(5, min(30, first))),),
                                            close_kernel_size=pre['close_kernel_size'],
                                            open_kernel_size=pre['open_kernel_size']),
                                min_area=pre['min_area'], min_solidity=0.1)
        dets = detect_blobs(masks, min_area=pre['min_area'])
        X, y, d, meta = pair_features(dets, boxes_b, H, W, kt.NTH_FRAME, seq_i=seq_i)
        if len(X):
            Xs.append(X); ys.append(y); ds.append(d); metas.append(meta)
            refs.append(np.full(len(X), person_h))
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(ds),
            np.concatenate(refs), np.vstack(metas))


def tracker_rates(scores, meta, thr):
    """Score the gate the way the tracker experiences it, not per candidate pair.

    Per-pair FNR is the wrong metric: the label marks every person-to-person pair positive,
    including head-at-t to legs-at-t+stride, but the tracker does not need to accept all of
    them - it needs, for each person detection, at least ONE accepted match that is also the
    person. Hence:

      loss_rate     fraction of person detections with NO accepted person match. This is what
                    kills tracks, and F1 says it is the failure that matters.
      contamination fraction of accepted matches out of person detections that land on a
                    NON-person detection. This is the cost of a looser gate.
    """
    accepted = scores >= thr
    gid, src_p, dst_p = meta[:, 0], meta[:, 1].astype(bool), meta[:, 2].astype(bool)
    src = src_p
    if not src.any():
        return float('nan'), float('nan')
    good = accepted & src & dst_p
    lost_groups, all_groups = set(np.unique(gid[src])), None
    kept = set(np.unique(gid[good]))
    all_groups = lost_groups
    loss_rate = 1.0 - len(kept) / max(len(all_groups), 1)
    acc_src = accepted & src
    contamination = float((~dst_p[acc_src]).mean()) if acc_src.any() else 0.0
    return loss_rate, contamination


def threshold_at_contamination(scores, meta, target):
    """Lowest (most permissive) threshold whose contamination stays <= target."""
    order = np.unique(scores)[::-1]
    best = order[0]
    for thr in order:
        _, cont = tracker_rates(scores, meta, thr)
        if cont > target:
            break
        best = thr
    return float(best)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    n = int(args[0]) if args else 3
    dist = float(sys.argv[sys.argv.index('--distractor') + 1]) if '--distractor' in sys.argv else 0.0
    global TARGET_CONTAMINATION
    if '--contamination' in sys.argv:
        TARGET_CONTAMINATION = float(sys.argv[sys.argv.index('--contamination') + 1])
    seqs = kt.SEQS[:n]
    print(f"stage 1: learned association gate vs non-learned dimensionless rules")
    print(f"{len(seqs)} sequences, occluder density {kt.OCC_DENSITY} (NFO-calibrated), "
          f"distractor {dist}px@1x, target contamination {TARGET_CONTAMINATION} "
          f"chosen on 1x only\n")

    data = {}
    for s in kt.SCALES:
        X, y, d, ph, meta = build_scale(seqs, s, distractor_px=dist)
        data[s] = (X, y, d, ph, meta)
        n_person = int(meta[:, 1].sum())
        print(f"  {s}x: {len(y)} pairs, {y.mean() * 100:.1f}% person-to-person, "
              f"{100 * (1 - meta[:, 2].mean()):.1f}% of destinations are non-person, "
              f"gt person height {ph[0]:.0f}px")

    Xtr, ytr, dtr, phtr, mtr = data[kt.REF_SCALE]
    model = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
    print("\nlearned weights (positive = evidence for a match):")
    for name, w in sorted(zip(FEATURES, model.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:>20}: {w:+.3f}")

    def scores_for(rule, X, d, ph):
        if rule == 'learned':
            return model.decision_function(X)
        if rule == 'fixed-h':
            return -d / (ALPHA_MAX_DIST * ph)   # today's rule, given a perfect scale estimate
        return -X[:, 0]                          # distance / median nearest-neighbour distance

    RULES = ('fixed-h', 'median-nn', 'learned')
    # every rule's operating point is chosen on the 1x bucket, then frozen
    thr = {r: threshold_at_contamination(scores_for(r, Xtr, dtr, phtr), mtr,
                                         TARGET_CONTAMINATION) for r in RULES}

    print(f"\nloss rate = person detections with NO accepted person match, threshold frozen "
          f"from 1x\n(contamination in brackets = accepted matches landing on a non-person)")
    print(f"{'bucket':>7} " + ' '.join(f"{r:>22}" for r in RULES))
    loss = {}
    for s in kt.SCALES:
        X, y, d, ph, meta = data[s]
        cells = []
        for rule in RULES:
            lr, cont = tracker_rates(scores_for(rule, X, d, ph), meta, thr[rule])
            loss[(s, rule)] = lr
            cells.append(f"{100 * lr:>13.1f}% [{100 * cont:>4.1f}%]")
        print(f"{s:>6}x " + ' '.join(cells))

    print("\nheld-out-scale summary (0.5x and 2x; 1x is the tuning bucket):")
    held = [s for s in kt.SCALES if s != kt.REF_SCALE]
    for rule in RULES:
        vals = [loss[(s, rule)] for s in held]
        print(f"  {rule:>10}: worst held-out loss {100 * max(vals):.1f}%, "
              f"drift from 1x {100 * (max(vals) - loss[(kt.REF_SCALE, rule)]):+.1f}pp")

    best_base = min(('fixed-h', 'median-nn'), key=lambda r: max(loss[(s, r)] for s in held))
    lw = max(loss[(s, 'learned')] for s in held)
    bw = max(loss[(s, best_base)] for s in held)
    print(f"\nGO/NO-GO: learned worst held-out loss {100 * lw:.1f}% vs best baseline "
          f"({best_base}) {100 * bw:.1f}%")
    print("  -> " + ("GO: learned rule transfers better, proceed to stage 2"
                     if lw < bw - 0.02 else
                     f"NO-GO: use {best_base} in the tracker instead, no learned gate needed"))
    if min(1 - data[s][4][:, 2].mean() for s in kt.SCALES) < 0.10:
        print("\nWARNING: fewer than 10% of candidate destinations are non-person, so there is"
              "\nalmost nothing for a gate to discriminate against and both contamination and"
              "\nthe GO/NO-GO margin rest on very few samples. Re-run with --distractor, or on"
              "\ndata with a cluttered background, before believing this verdict.")


if __name__ == '__main__':
    main()
