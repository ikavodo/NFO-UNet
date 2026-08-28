"""Sweep MERGE_RADIUS on real NFO, settling the open finding F7 in
docs/scale_generalization_plan.md: NFO's value of 100px (= measured person height / 2) looked
too large, because seq2's wrongly-small measured person height produced a smaller radius and
made that sequence better (0.1095 -> 0.0783 mean residual).

Cheap by construction. `merge_radius` is used in exactly one place - `position_from_track` ->
`merged_center`, which merges detections near the winning track's anchor into one box and
reports its centre. It does not affect background subtraction, association, or scoring. So the
tracker only has to run ONCE and every radius can be evaluated on the same winning tracks.

Radii are swept as multiples of NFO's measured mean person height (195px) so the result is
directly usable as an ALPHA_MERGE coefficient in
tracking/core/track_sequence.scale_relative_params, rather than as another absolute pixel
constant.

    python -m tracking.eval.merge_radius_sweep
"""
import os

import cv2
import numpy as np

from tracking.core.blob_tracker import detect_blobs, track_blobs, score_and_fit
from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.track_window import position_from_track
from tracking.eval.eval_nfo import (IN_DIR, SEQS, SPAN, NTH_FRAME, BG_FRAMES, MAX_DIST,
                                    MERGE_RADIUS, EXPECTED_HEIGHT, parse_normalized_bbs)

HIT = 0.1
# multiples of person height; 0.5 is what eval_nfo currently uses, 0.75 is what the synthetic
# sweep preferred, 0.0 means "no merge at all - report the winning fragment's own anchor"
ALPHAS = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5]


def sweep_sequence(seq):
    """-> {alpha: [residuals]} plus the measured mean GT person height in px."""
    seq_in = os.path.join(IN_DIR, seq)
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    norm_file = next(f for f in os.listdir(seq_in)
                     if f != 'groundtruth.txt' and f.startswith('groundtruth'))
    raw = parse_normalized_bbs(os.path.join(seq_in, norm_file))
    n = min(len(jpgs), len(raw))
    frames_all = np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(n)], axis=0)
    H, W = frames_all.shape[1:]
    centers = [c for c in range(SPAN, n - SPAN) if raw[c] is not None]
    person_h = float(np.mean([raw[c][3] * H for c in centers]))

    masks = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)),
                            min_area=50)
    out = {a: [] for a in ALPHAS}
    n_no_track = 0
    for c in centers:
        idx = list(range(c - SPAN, c + SPAN + 1, NTH_FRAME))
        dets = detect_blobs(masks[idx], min_area=50)
        winner = score_and_fit(track_blobs(dets, max_dist=MAX_DIST),
                              expected_height=EXPECTED_HEIGHT)
        if winner is None:
            n_no_track += 1
            continue
        gt = raw[c]
        gx, gy = (gt[0] + gt[2] / 2) * W, (gt[1] + gt[3] / 2) * H
        for a in ALPHAS:
            pos = position_from_track(winner, dets, a * person_h)
            out[a].append(float(np.hypot((pos['x'] - gx) / W, (pos['y'] - gy) / H)))
    return out, person_h, n_no_track, len(centers)


def line(vals, label):
    r = np.array(vals)
    return (f"{label:>18}: mean={r.mean():.4f} median={np.median(r):.4f} "
            f"p90={np.percentile(r, 90):.4f} hit@{HIT}={100 * (r < HIT).mean():.1f}%")


def main():
    pooled = {a: [] for a in ALPHAS}
    print(f"sweeping merge_radius as a multiple of person height; eval_nfo currently uses "
          f"{MERGE_RADIUS:.0f}px\n")
    for seq in SEQS:
        out, person_h, nnt, ntot = sweep_sequence(seq)
        cur = MERGE_RADIUS / person_h
        print(f"{seq}: person_h={person_h:.0f}px, current {MERGE_RADIUS:.0f}px = "
              f"{cur:.2f}x height, no_track={nnt}/{ntot}")
        best = min(ALPHAS, key=lambda a: np.mean(out[a]))
        for a in ALPHAS:
            mark = '  <- best here' if a == best else ''
            print("   " + line(out[a], f"{a:.3f}x = {a * person_h:.0f}px") + mark)
            pooled[a] += out[a]
        print()

    print("POOLED over all four sequences:")
    for a in ALPHAS:
        print("   " + line(pooled[a], f"{a:.3f}x height"))
    best_mean = min(ALPHAS, key=lambda a: np.mean(pooled[a]))
    best_hit = max(ALPHAS, key=lambda a: (np.array(pooled[a]) < HIT).mean())
    cur_alpha = min(ALPHAS, key=lambda a: abs(a - MERGE_RADIUS / 195.0))
    print(f"\nbest by mean residual: {best_mean:.3f}x height")
    print(f"best by hit@{HIT}:       {best_hit:.3f}x height")
    print(f"nearest swept value to eval_nfo's current 100px: {cur_alpha:.3f}x height "
          f"(mean {np.mean(pooled[cur_alpha]):.4f})")
    print(f"ALPHA_MERGE in track_sequence.py is currently 0.75")


if __name__ == '__main__':
    main()
