"""Multi-checkpoint SAM2 pseudo-mask generation for NFO.

A single prompt propagated across a whole continuously-visible segment is not reliable (see
prototype_sam2_video_segment.py's diagnostic: 41/155 frames came back empty on seq1's longest
segment, all in one contiguous block, no self-recovery). This seeds one checkpoint per
geometrically-confirmed clear-visibility crossing within each segment (nfo_visibility.py -
GT-trajectory cross-referenced against fixed occluder positions, no detector needed), then
propagates each in both directions, bounded by its neighbors: checkpoint i propagates outward
to its own segment edge (if it has no neighbor on that side) or only as far as the neighboring
checkpoint (if it has one). This generalizes cleanly to however many checkpoints a segment
actually has (0, 1, or several) instead of a fixed count, and gives every frame between two
consecutive checkpoints double coverage (real agreement, not just a single meeting point).

If a segment has zero confirmed-clear frames anywhere, falls back to a single naive-middle
checkpoint (same behavior as the original single-prompt approach for that segment only).

Requires a GPU and the sam2 package (not available in this dev environment):
    git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .

Usage (one sequence per process - run seq1..seq4 in parallel, see module docstring below):
    python3 -m gen_data.gen_nfo_pseudo_masks --seq seq1
"""
import argparse
import itertools
import os

import cv2
import numpy as np
import torch

from gen_data.nfo_segment_utils import find_segments, stage_frames
from gen_data.nfo_visibility import default_clear_regions, geometric_checkpoints
from utils.bb_utils import parse_bbs

IN_DIR = 'data/nfo_processed'
OUT_TAG = 'sammask'
MAX_CONSECUTIVE_EMPTY = 5  # early-stop a direction once it's clearly dead, don't burn compute


def point_and_box_from_gt(bbs, raw_idx, img_w, img_h):
    """Returns (center_point, box) in pixel coords. Passing the full box (not just its center
    point) to SAM2 lets it use the GT's known vertical extent directly - a point-only prompt has
    to grow into low-contrast regions (e.g. pants the same color as foliage) via appearance
    similarity alone, which is exactly where it can fail to reach the feet; a box prompt tells
    it explicitly where the object's extent actually is."""
    bb = bbs[raw_idx][0]
    point = ((bb.x + bb.w / 2) * img_w, (bb.y + bb.h / 2) * img_h)
    box = [bb.x * img_w, bb.y * img_h, (bb.x + bb.w) * img_w, (bb.y + bb.h) * img_h]
    return point, box


def compute_bounds(checkpoints):
    """checkpoint i propagates outward unbounded on the side with no neighbor, and only as far
    as the neighboring checkpoint on the side that has one - generalizes the old fixed 3-way
    (outer/middle/outer) scheme to any number of checkpoints, including 1 (fully unbounded both
    ways, same as a lone single-checkpoint run)."""
    bounds = {}
    for i, cp in enumerate(checkpoints):
        max_forward = checkpoints[i + 1] - cp if i + 1 < len(checkpoints) else None
        max_backward = cp - checkpoints[i - 1] if i > 0 else None
        bounds[cp] = dict(max_forward=max_forward, max_backward=max_backward)
    return bounds


def propagate_one_checkpoint(predictor, frame_dir, checkpoint_local_idx, point, box, device,
                             max_forward=None, max_backward=None):
    """Returns {local_idx: bool_mask}. max_forward/max_backward cap how many frames to track
    in each direction (None = unbounded, subject only to the empty-run early-stop)."""
    results = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        state = predictor.init_state(frame_dir)
        predictor.add_new_points_or_box(state, frame_idx=checkpoint_local_idx, obj_id=1,
                                        points=[point], labels=[1], box=box)

        for reverse, max_num in ((False, max_forward), (True, max_backward)):
            if max_num == 0 and checkpoint_local_idx in results:
                continue  # already have the checkpoint's own frame from the other direction
            consecutive_empty = 0
            for frame_idx, _, masks in predictor.propagate_in_video(
                    state, start_frame_idx=checkpoint_local_idx,
                    max_frame_num_to_track=max_num, reverse=reverse):
                mask = (masks[0] > 0).cpu().numpy().squeeze()
                if mask.sum() == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                        break
                else:
                    consecutive_empty = 0
                results[frame_idx] = mask
    return results


def combine_checkpoint_masks(per_checkpoint_results, n_seg_frames):
    """Majority vote per frame across whichever checkpoints reached it.

    Returns (combined: {local_idx: bool_mask}, diagnostics: [(local_idx, n_reached, min_iou)]).
    """
    combined = {}
    diagnostics = []
    for local_idx in range(n_seg_frames):
        masks = [r[local_idx] for r in per_checkpoint_results if local_idx in r and r[local_idx].sum() > 0]
        n_reached = len(masks)
        if n_reached == 0:
            diagnostics.append((local_idx, 0, float('nan')))
            continue
        if n_reached == 1:
            combined[local_idx] = masks[0]
            diagnostics.append((local_idx, 1, float('nan')))
            continue

        vote = np.zeros(masks[0].shape, dtype=np.int32)
        for m in masks:
            vote += m.astype(np.int32)
        combined[local_idx] = vote > (n_reached / 2)

        ious = []
        for a, b in itertools.combinations(masks, 2):
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            ious.append(inter / union if union else 1.0)
        diagnostics.append((local_idx, n_reached, min(ious)))
    return combined, diagnostics


def combine_checkpoint_masks_union_gt_outlier(per_checkpoint_results, n_seg_frames, bbs, start,
                                              img_w, img_h, gt_dist_factor=1.25, min_width_px=4):
    """TEMPORARY alternate combination strategy for comparison against the majority-vote
    default: union of all available masks per frame (maximizes recall/coverage instead of the
    default's intersection-like behavior, which trades recall for precision) - then reject
    outlier connected components using the GT box as an anchor, since we actually have ground
    truth for every frame in these labeled segments (that's what seeds the checkpoints in the
    first place) - a much stronger reference than a statistic like "median of 2 masks" would be
    at this sample size. A component survives if its centroid is within gt_dist_factor times
    the GT box's own size of the GT box center; anything farther is almost certainly drift onto
    the wrong object, not the person.

    gt_dist_factor was 2.0 initially; tightened to 1.25 after visual review of real seq1 output
    (segment 7) showed SAM2's memory-based tracking getting "stuck" on a frozen spatial location
    after the person walks past - since the filter compares against the *current frame's* GT
    position (not a fixed reference), it does eventually reject a stuck blob once the person has
    walked far enough away, but 2.0x box-size was generous enough to let it survive for several
    frames after the tracker had already lost the real target. 1.25x (roughly one body-width of
    tolerance) catches this faster while still allowing normal partial-visibility offset.

    Also rejects components narrower than min_width_px (bounding-box width, not area) - a
    recurring artifact (visually confirmed on real seq1 output) is a persistent thin vertical
    sliver, likely a trunk edge one checkpoint's mask keeps including, 1-3px wide even where
    it's tall. A real person fragment - even a partial one, occluded by branches - has
    meaningfully more width than that; this only removes hairline slivers, not body parts (see
    the module's test for the calibration check on synthetic near/far/thin cases).

    Returns (combined, diagnostics) with the same shape as combine_checkpoint_masks.
    """
    combined = {}
    diagnostics = []
    for local_idx in range(n_seg_frames):
        masks = [r[local_idx] for r in per_checkpoint_results if local_idx in r and r[local_idx].sum() > 0]
        n_reached = len(masks)
        if n_reached == 0:
            diagnostics.append((local_idx, 0, float('nan')))
            continue

        union = np.zeros(masks[0].shape, dtype=bool)
        for m in masks:
            union |= m

        raw_idx = start + local_idx
        gt_list = bbs.get(raw_idx)
        if gt_list and gt_list[0].x >= 0:
            bb = gt_list[0]
            gt_cx, gt_cy = (bb.x + bb.w / 2) * img_w, (bb.y + bb.h / 2) * img_h
            # width, not max(w,h): drift/stuck-tracker artifacts are horizontal (the person
            # walks sideways), so tolerance should scale with the horizontal body size, not
            # height (~2.5x larger for a standing person, which made the old max(w,h) version
            # far looser than the "1.25x" multiplier suggested - see gt_dist_factor docstring)
            max_dist = gt_dist_factor * bb.w * img_w
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(union.astype(np.uint8))
            keep = np.zeros_like(union)
            for lbl in range(1, n_labels):  # 0 is background
                cx, cy = centroids[lbl]
                width = stats[lbl, cv2.CC_STAT_WIDTH]
                if width >= min_width_px and np.hypot(cx - gt_cx, cy - gt_cy) <= max_dist:
                    keep |= (labels == lbl)
            combined[local_idx] = keep
        else:
            combined[local_idx] = union  # no GT available - trust the union as-is

        if n_reached >= 2:
            ious = []
            for a, b in itertools.combinations(masks, 2):
                inter = np.logical_and(a, b).sum()
                u = np.logical_or(a, b).sum()
                ious.append(inter / u if u else 1.0)
            diagnostics.append((local_idx, n_reached, min(ious)))
        else:
            diagnostics.append((local_idx, n_reached, float('nan')))
    return combined, diagnostics


def process_segment(predictor, seq_dir, bbs, start, end, seg_idx, out_dir_frames, device,
                    clear_regions, combine_method='majority'):
    n_seg_frames = end - start + 1
    frame_dir = os.path.join(out_dir_frames, f'seg{seg_idx}_frames')
    stage_frames(seq_dir, start, end, frame_dir)
    img_h, img_w = cv2.imread(os.path.join(seq_dir, f'{start:05d}_or.jpg'), 0).shape

    geo_raw = geometric_checkpoints(bbs, start, end, clear_regions, img_w)
    if geo_raw:
        checkpoints = sorted(idx - start for idx in geo_raw)
        print(f'  segment {seg_idx}: {len(checkpoints)} geometric checkpoint(s) at local {checkpoints}')
    else:
        checkpoints = [n_seg_frames // 2]
        print(f'  segment {seg_idx}: no confirmed-clear frame found, falling back to naive '
              f'middle at local {checkpoints[0]}')

    bounds = compute_bounds(checkpoints)
    per_checkpoint_results = []
    for cp_local_idx in checkpoints:
        raw_idx = start + cp_local_idx
        point, box = point_and_box_from_gt(bbs, raw_idx, img_w, img_h)
        per_checkpoint_results.append(
            propagate_one_checkpoint(predictor, frame_dir, cp_local_idx, point, box, device,
                                     **bounds[cp_local_idx]))

    if combine_method == 'union_gt_outlier':
        combined, diagnostics = combine_checkpoint_masks_union_gt_outlier(
            per_checkpoint_results, n_seg_frames, bbs, start, img_w, img_h)
    else:
        combined, diagnostics = combine_checkpoint_masks(per_checkpoint_results, n_seg_frames)

    for local_idx, mask in combined.items():
        raw_idx = start + local_idx
        out_path = os.path.join(seq_dir, f'{raw_idx:05d}_{OUT_TAG}.png')
        cv2.imwrite(out_path, (mask * 255).astype(np.uint8))

    return diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', required=True, help='e.g. seq1')
    parser.add_argument('--model-id', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--tmp-dir', default='tracking/sam2_pseudo_mask_tmp',
                        help='scratch space for staged per-segment frame symlinks')
    parser.add_argument('--segment-idx', type=int, default=None,
                        help='only process this one segment instead of the whole sequence')
    parser.add_argument('--combine-method', choices=['majority', 'union_gt_outlier'],
                        default='majority',
                        help='majority: intersection-like agreement (default, high precision, '
                             'sparse coverage). union_gt_outlier: TEMPORARY comparison method - '
                             'union of masks, GT-box-anchored outlier rejection (more coverage, '
                             'trades on trusting GT-anchored connected components)')
    args = parser.parse_args()

    seq_dir = os.path.join(IN_DIR, f'{args.seq}_gt')
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    n_frames = max(bbs.keys()) + 1
    img_h, img_w = cv2.imread(os.path.join(seq_dir, '00000_or.jpg'), 0).shape
    segments = find_segments(bbs)
    if args.segment_idx is not None:
        segments = [segments[args.segment_idx]]
        print(f'{args.seq} segment {args.segment_idx}: frames {segments[0]}')
    else:
        print(f'{args.seq}: {len(segments)} continuously-visible segments, '
              f'lengths {[e - s + 1 for s, e in segments]}')

    clear_regions = default_clear_regions(seq_dir, bbs, n_frames, img_h, img_w)
    print(f'{args.seq}: clear regions {clear_regions}')

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir_frames = os.path.join(args.tmp_dir, args.seq)
    suffix = f'_seg{args.segment_idx}' if args.segment_idx is not None else ''
    csv_path = f'{args.seq}{suffix}_pseudo_mask_diagnostics.csv'
    with open(csv_path, 'w') as f:
        f.write('segment_idx,local_idx,raw_idx,n_checkpoints_reached,min_pairwise_iou\n')
        for seg_idx, (start, end) in enumerate(segments):
            diagnostics = process_segment(predictor, seq_dir, bbs, start, end, seg_idx,
                                          out_dir_frames, device, clear_regions,
                                          combine_method=args.combine_method)
            for local_idx, n_reached, min_iou in diagnostics:
                f.write(f'{seg_idx},{local_idx},{start + local_idx},{n_reached},{min_iou}\n')

            n_zero = sum(1 for _, n, _ in diagnostics if n == 0)
            print(f'  segment {seg_idx}: {n_zero}/{len(diagnostics)} frames unlabeled '
                  f'({100 * n_zero / len(diagnostics):.1f}%)')

    print(f'wrote diagnostics to {csv_path}')


if __name__ == '__main__':
    main()
