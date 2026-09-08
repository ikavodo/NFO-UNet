"""Motion-compensated temporal integration driven by GROUND TRUTH position, not the tracker.

    python -m tracking.eval.gt_integrated_image --seq seq1_gt

For each contiguous ground-truth run (one traversal - see 20_nfo_gt_check.png, the same
sequence: f11 through f2102 are frames within these runs), fits a single constant x-velocity to
the WHOLE segment from the GT box centres (the "ground truth velocity"), then re-uses
align_frames/fuse (tracking/core/integrate_image.py) exactly as the tracker's own pipeline does -
same crop-following-a-constant-velocity-anchor mechanism, same median fusion - but fed the true
position instead of an estimated one. This is the ORACLE version of the integration step: whatever
still looks bad here cannot be blamed on tracking error, only on the fusion step itself (long-run
gait articulation, illumination, occluder density).

BASELINE INCLUDED, NOT OPTIONAL: alongside the GT-aligned integration, the SAME frames are also
fused with vx=0 (a world-fixed window at the segment's mean position) - the "do-nothing" control.
align_frames' own docstring already establishes why these differ: a world-fixed window keeps
static occluders sharp and blurs the (moving) person, exactly backwards from what alignment buys.
Showing both is the only way this claim is checked visually rather than assumed.

Segments here run 83-155 frames - an order of magnitude longer than the tracker's own 7-31 frame
windows this project has tested elsewhere. That is a real difference worth watching for in the
result, not swept under the rug: fusing across a whole gait-cycle-heavy traversal gives the fitted
velocity far more chances to drift from the true (non-constant-velocity, oscillating-gait) path
than a short window does, and if the aligned panel looks smeared rather than sharp, segment length
is the first suspect, not the method.
"""
import argparse
import os

import cv2
import numpy as np

from tracking.core.integrate_image import align_frames, fuse
from tracking.eval.lookbehind_discrimination import gt_runs, load_sequence
from tracking.eval.yolo_vs_tracker import yolo_boxes


def build_gt_winner(gt: dict, run: tuple, w: int, h: int):
    """An align_frames-compatible 'winner' dict built from GT boxes for one run, plus the
    absolute frame indices it corresponds to.

    gt values are normalised (x, y, bw, bh) as fractions of frame size (nfo_processed's own
    groundtruth.txt convention - see parse_normalized_bbs in eval_nfo.py).

    History is keyed by LOCAL index 0..T-1 within the segment, matching the frame stack
    align_frames will receive (frames[abs_idx]) - align_frames indexes both by the same loop
    variable `t`, so absolute video frame numbers here would silently misalign every crop.

    vx is an OLS fit of x-centre against local frame index over the WHOLE run - one velocity per
    segment, exactly what a constant-velocity crop path needs and what align_frames applies from a
    single center anchor. y is stored TRUE per frame (not pre-averaged): align_frames computes its
    own ay = mean(history[f][1] for f in frames) once it receives this dict, matching the tracker's
    horizontal-only motion model - collapsing y here would just duplicate that averaging one layer
    early and discard the one place the true y-trajectory is recorded.
    """
    abs_idx = list(range(run[0], run[1] + 1))
    cx = np.array([(gt[f][0] + gt[f][2] / 2) * w for f in abs_idx])
    cy = np.array([(gt[f][1] + gt[f][3] / 2) * h for f in abs_idx])
    ht = np.array([gt[f][3] * h for f in abs_idx])
    local = np.arange(len(abs_idx))
    if len(local) > 1:
        A = np.vstack([local, np.ones(len(local))]).T
        vx = float(np.linalg.lstsq(A, cx, rcond=None)[0][0])
    else:
        vx = 0.0
    history = {int(t): (float(cx[t]), float(cy[t]), float(ht[t])) for t in local}
    return dict(frames=list(int(t) for t in local), history=history, vx=vx), abs_idx


def draw_yolo(img, boxes):
    """Boxes + confidence drawn ONTO the integrated image itself (not a separate panel) - the
    question here is whether YOLO fires on THIS pixel content, so the boxes need to sit directly
    over the crop that produced them, not beside it."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for (x1, y1, x2, y2), c in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f'{c:.2f}', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   (255, 0, 255), 1)
    return vis


def panel(img, subtitle):
    """One crop with a SHORT caption underneath it - not the full run description, which does not
    fit in a single crop_size-wide panel and was clipping off-image (crop=224 read as "crop" with
    no number). The full description goes on a header spanning BOTH panels instead, in main()."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    vis = cv2.copyMakeBorder(vis, 0, 18, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(vis, subtitle, (4, vis.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
               (255, 255, 255), 1)
    return vis


def header(width, text):
    bar = np.zeros((22, width, 3), np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return bar


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', default='seq1_gt')
    p.add_argument('--data-dir', default='data/nfo_processed')
    p.add_argument('--method', default='median', choices=('median', 'mean'))
    p.add_argument('--crop-mult', type=float, default=4.5,
                   help='crop_size = crop_mult * segment mean GT height, clipped to the frame')
    p.add_argument('--limit', type=int, default=None,
                   help='only render this many runs (the first N in the sequence), for a short '
                        'demo montage instead of every traversal')
    p.add_argument('--caption', choices=('full', 'simple'), default='full',
                   help="'simple' drops the run header and technical detail (frame range, n, "
                        "height, crop size) and labels each panel just 'unaligned (vx=0)' / "
                        "'aligned (vx=...)' - for a non-technical audience where the mechanism, "
                        "not the measurement, is the point")
    p.add_argument('--window', type=int, default=None,
                   help='restrict each run to a centered sub-window of this many frames before '
                        'fitting velocity/aligning, instead of the whole traversal - e.g. 31 to '
                        "match the tracker's own tested window sizes elsewhere in this project. "
                        'Segments here are otherwise 83-155 frames, 5-20x longer, which gives '
                        'gait articulation far more room to smear the median than a short window '
                        'does - if the aligned panel looks smeared rather than sharp, this is the '
                        'first thing to change before concluding anything about the METHOD.')
    p.add_argument('--layout', choices=('rows', 'grid'), default='rows',
                   help="'rows' (default): one row per segment, unaligned then aligned side by "
                        "side. 'grid': one COLUMN per segment instead - all unaligned panels "
                        "across the top row, all aligned panels across the bottom row - so N "
                        "segments render as a 2xN grid rather than an Nx2 stack. No per-run "
                        "header in this mode (there is no single column it could span).")
    p.add_argument('--yolo', nargs='?', const='data/yolo11m.pt', default=None, metavar='WEIGHTS',
                   help='run YOLO on both the unaligned and aligned integrated image per segment '
                        'and draw its boxes/confidence directly on them (bare flag uses '
                        'data/yolo11m.pt). Prints a recall summary: how many segments get a '
                        'detection unaligned vs aligned - the actual test of whether integration '
                        'helps a downstream detector, which is what mask_background=False in '
                        'integrate_image.py was built for and never checked against a detector.')
    p.add_argument('--yolo-conf', type=float, default=0.25)
    p.add_argument('--yolo-device', default='cuda')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    model = None
    if a.yolo:
        import os as _os
        _os.environ.setdefault('YOLO_VERBOSE', 'False')
        from ultralytics import YOLO
        model = YOLO(a.yolo)

    seq_dir = os.path.join(a.data_dir, a.seq)
    frames, gt = load_sequence(seq_dir)
    T, H, W = frames.shape
    runs = gt_runs(gt)
    print(f'{a.seq}: {T} frames {W}x{H}, {len(runs)} GT runs (traversals)')

    rows, tops, bottoms, yolo_tally = [], [], [], []
    for run in (runs[:a.limit] if a.limit else runs):
        if a.window and (run[1] - run[0] + 1) > a.window:
            mid = (run[0] + run[1]) // 2
            half = a.window // 2
            run = (max(run[0], mid - half), min(run[1], mid - half + a.window - 1))
        winner, abs_idx = build_gt_winner(gt, run, W, H)
        seg = frames[abs_idx]
        mean_h = float(np.mean([winner['history'][t][2] for t in winner['history']]))
        crop = int(np.clip(round(a.crop_mult * mean_h), 60, min(W, H)))

        aligned = align_frames(seg, winner, crop_size=crop)
        gt_fused = fuse(aligned, method=a.method)

        unaligned_winner = dict(winner, vx=0.0)          # the do-nothing control: fixed window
        static = fuse(align_frames(seg, unaligned_winner, crop_size=crop), method=a.method)

        det_static = det_aligned = []
        if model is not None:
            det_static = yolo_boxes(model, static, a.yolo_conf, a.yolo_device)
            det_aligned = yolo_boxes(model, gt_fused, a.yolo_conf, a.yolo_device)
            static, gt_fused = draw_yolo(static, det_static), draw_yolo(gt_fused, det_aligned)
            yolo_tally.append((bool(det_static), bool(det_aligned)))
            best = lambda d: max((c for _, c in d), default=0.0)
            print(f'    YOLO: unaligned {"fires " + format(best(det_static), ".2f") if det_static else "nothing"}'
                  f'   aligned {"fires " + format(best(det_aligned), ".2f") if det_aligned else "nothing"}')

        n = run[1] - run[0] + 1
        left_cap = 'unaligned (vx=0)' if a.caption == 'simple' else \
            'world-fixed (vx=0, the do-nothing control)'
        right_cap = (f'aligned (vx={winner["vx"]:+.2f})' if a.caption == 'simple' else
                    f'GT-aligned  vx={winner["vx"]:+.2f}px/frame')
        if a.layout == 'grid':
            tops.append(panel(static, left_cap))
            bottoms.append(panel(gt_fused, right_cap))
        else:
            left, right = panel(static, left_cap), panel(gt_fused, right_cap)
            pair = np.hstack([left, right])
            if a.caption == 'simple':
                rows.append(pair)
            else:
                rows.append(np.vstack([header(pair.shape[1], f'f{run[0]}-{run[1]}  n={n}  '
                                                              f'mean height {mean_h:.0f}px  '
                                                              f'crop {crop}px'), pair]))
        print(f'  run {run}: n={n} mean_height={mean_h:.1f}px fitted_vx={winner["vx"]:+.3f}px/frame '
              f'crop_size={crop}')

    if a.layout == 'grid':
        montage = np.vstack([np.hstack(tops), np.hstack(bottoms)])
    else:
        w = max(r.shape[1] for r in rows)
        rows = [r if r.shape[1] == w else cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                                                              cv2.BORDER_CONSTANT) for r in rows]
        montage = np.vstack(rows)
    out = a.out or f'images/stream/{a.seq}_gt_integrated.png'
    cv2.imwrite(out, montage)
    print(f'wrote {out}')
    if yolo_tally:
        n = len(yolo_tally)
        fu = sum(u for u, _ in yolo_tally)
        fa = sum(g for _, g in yolo_tally)
        print(f'\nYOLO fired on {fu}/{n} unaligned segments, {fa}/{n} aligned segments')


if __name__ == '__main__':
    main()
