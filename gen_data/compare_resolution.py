"""One-off experiment: does running SAM2 at native NFO resolution (800x600, then downsampled
to 224 for comparison) produce better masks than running at the pipeline's current 224x224?

Reuses gen_nfo_pseudo_masks.py's checkpoint selection, propagation, and combination logic
unchanged - only the frame source and prompt pixel coordinates differ. Doesn't touch the real
pipeline's output files; computes box-recovery IoU (bbox of the final mask vs. the GT box - a
free, no-extra-annotation quality check the mask "should" reproduce) in-memory and prints a
summary, for direct comparison against the same metric computed on the existing 224-resolution
masks already on disk.

Requires a GPU and the sam2 package (not available in this dev environment).

Usage:
    python3 -m gen_data.compare_resolution --seq seq1 --segment-idx 3
"""
import argparse
import os

import cv2
import numpy as np
import torch

from gen_data.gen_kth_data.kth_utils import scale_and_pad_img_to_square
from gen_data.gen_nfo_pseudo_masks import (combine_checkpoint_masks_union_gt_outlier,
                                           compute_bounds, propagate_one_checkpoint)
from gen_data.nfo_segment_utils import find_segments
from gen_data.nfo_visibility import default_clear_regions, geometric_checkpoints
from utils.bb_utils import BoundingBox, parse_bbs

NATIVE_DIR = 'data/nfo_final/nfo_final'
PROCESSED_DIR = 'data/nfo_processed'
OUT_SIZE = 224


def bbox_iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def stage_native_frames(seq, start, end, frame_dir):
    if os.path.exists(frame_dir):
        import shutil
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)
    for local_idx, raw_idx in enumerate(range(start, end + 1)):
        src = os.path.abspath(os.path.join(NATIVE_DIR, seq, f'{raw_idx:05d}.jpg'))
        os.symlink(src, os.path.join(frame_dir, f'{local_idx}.jpg'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', required=True)
    parser.add_argument('--segment-idx', type=int, required=True)
    parser.add_argument('--model-id', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--tmp-dir', default='tracking/compare_resolution_tmp')
    parser.add_argument('--save-masks', action='store_true',
                        help='write native-res and 224-downsampled masks to <tmp-dir>/'
                             '<seq>_seg<idx>_masks/ for visual comparison against the real '
                             'pipeline output')
    args = parser.parse_args()

    processed_dir = os.path.join(PROCESSED_DIR, f'{args.seq}_gt')
    bbs = parse_bbs(os.path.join(processed_dir, 'groundtruth.txt'))
    n_frames = max(bbs.keys()) + 1
    img_h_224, img_w_224 = cv2.imread(os.path.join(processed_dir, '00000_or.jpg'), 0).shape
    segments = find_segments(bbs)
    start, end = segments[args.segment_idx]
    n_seg_frames = end - start + 1
    print(f'{args.seq} segment {args.segment_idx}: raw {start}-{end} ({n_seg_frames} frames)')

    # checkpoint selection is resolution-independent (based on normalized GT + curated
    # 224-space clear regions, which only decide *which frame* to seed, not pixel coords)
    clear_regions = default_clear_regions(processed_dir, bbs, n_frames, img_h_224, img_w_224)
    geo_raw = geometric_checkpoints(bbs, start, end, clear_regions, img_w_224)
    checkpoints = sorted(idx - start for idx in geo_raw) if geo_raw else [n_seg_frames // 2]
    print(f'checkpoints (local): {checkpoints}')
    bounds = compute_bounds(checkpoints)

    native_h, native_w = cv2.imread(
        os.path.join(NATIVE_DIR, args.seq, f'{start:05d}.jpg'), 0).shape
    print(f'native resolution: {native_w}x{native_h}')

    frame_dir = os.path.join(args.tmp_dir, f'{args.seq}_seg{args.segment_idx}')
    stage_native_frames(args.seq, start, end, frame_dir)

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    per_checkpoint_results = []
    for cp_local_idx in checkpoints:
        raw_idx = start + cp_local_idx
        bb = bbs[raw_idx][0]
        point = ((bb.x + bb.w / 2) * native_w, (bb.y + bb.h / 2) * native_h)
        box = [bb.x * native_w, bb.y * native_h, (bb.x + bb.w) * native_w, (bb.y + bb.h) * native_h]
        per_checkpoint_results.append(
            propagate_one_checkpoint(predictor, frame_dir, cp_local_idx, point, box, device,
                                     **bounds[cp_local_idx]))

    combined, _ = combine_checkpoint_masks_union_gt_outlier(
        per_checkpoint_results, n_seg_frames, bbs, start, native_w, native_h)

    mask_dir = os.path.join(args.tmp_dir, f'{args.seq}_seg{args.segment_idx}_masks')
    if args.save_masks:
        os.makedirs(os.path.join(mask_dir, 'native'), exist_ok=True)
        os.makedirs(os.path.join(mask_dir, '224'), exist_ok=True)

    ious = []
    for local_idx, mask in combined.items():
        raw_idx = start + local_idx
        if raw_idx not in bbs or bbs[raw_idx][0].x < 0:
            continue
        # downsample this frame's native-resolution mask to 224 the same way the real
        # pipeline's frames/boxes are prepared, for a fair comparison against the 224-based GT
        mask_224, _ = scale_and_pad_img_to_square(
            (mask * 255).astype(np.uint8), BoundingBox(0, 0, 0, 0), OUT_SIZE)
        if args.save_masks:
            cv2.imwrite(os.path.join(mask_dir, 'native', f'{raw_idx:05d}.png'),
                       (mask * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(mask_dir, '224', f'{raw_idx:05d}.png'), mask_224)
        m = mask_224 > 127
        if not m.any():
            ious.append(0.0)
            continue
        ys, xs = np.nonzero(m)
        mask_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        bb = bbs[raw_idx][0]
        gt_box = (bb.x * OUT_SIZE, bb.y * OUT_SIZE, (bb.x + bb.w) * OUT_SIZE, (bb.y + bb.h) * OUT_SIZE)
        ious.append(bbox_iou(mask_box, gt_box))

    ious = np.array(ious)
    print(f'native-resolution box-recovery IoU: n={len(ious)} mean={ious.mean():.3f} '
          f'median={np.median(ious):.3f} frac_zero={(ious == 0).mean():.3f}')
    if args.save_masks:
        print(f'masks saved to {mask_dir}/native/ (native-res) and {mask_dir}/224/ '
              f'(downsampled, directly comparable to data/nfo_processed/{args.seq}_gt/*_sammask.png)')


if __name__ == '__main__':
    main()
