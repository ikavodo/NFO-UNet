"""Multi-checkpoint SAM2 pseudo-mask generation for NFO.

A single prompt propagated across a whole continuously-visible segment is not reliable (see
prototype_sam2_video_segment.py's diagnostic: 41/155 frames came back empty on seq1's longest
segment, all in one contiguous block, no self-recovery). This spreads N_CHECKPOINTS independent
prompts (seeded from ground truth) across each segment, propagates each in both directions, and
combines the overlapping per-frame masks by simple per-pixel majority vote. v1: simplest
reasonable combination rule - refine later if the false-negative rate is still too high.

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
from utils.bb_utils import parse_bbs

IN_DIR = 'data/nfo_processed'
OUT_TAG = 'sammask'
CHECKPOINT_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
MAX_CONSECUTIVE_EMPTY = 5  # early-stop a direction once it's clearly dead, don't burn compute


def point_from_gt(seq_dir, bbs, raw_idx, img_w, img_h):
    bb = bbs[raw_idx][0]
    return ((bb.x + bb.w / 2) * img_w, (bb.y + bb.h / 2) * img_h)


def propagate_one_checkpoint(predictor, frame_dir, checkpoint_local_idx, point, n_seg_frames, device):
    """Returns {local_idx: bool_mask} for every frame this checkpoint's forward+backward
    propagation reached before either direction was early-stopped."""
    results = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        state = predictor.init_state(frame_dir)
        predictor.add_new_points_or_box(state, frame_idx=checkpoint_local_idx, obj_id=1,
                                        points=[point], labels=[1])

        for reverse in (False, True):
            consecutive_empty = 0
            for frame_idx, _, masks in predictor.propagate_in_video(
                    state, start_frame_idx=checkpoint_local_idx, reverse=reverse):
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


def process_segment(predictor, seq_dir, bbs, start, end, seg_idx, out_dir_frames, device):
    n_seg_frames = end - start + 1
    frame_dir = os.path.join(out_dir_frames, f'seg{seg_idx}_frames')
    stage_frames(seq_dir, start, end, frame_dir)

    img_h, img_w = cv2.imread(os.path.join(seq_dir, f'{start:05d}_or.jpg'), 0).shape
    checkpoints = sorted(set(int(round((n_seg_frames - 1) * f)) for f in CHECKPOINT_FRACS))

    per_checkpoint_results = []
    for cp_local_idx in checkpoints:
        raw_idx = start + cp_local_idx
        point = point_from_gt(seq_dir, bbs, raw_idx, img_w, img_h)
        print(f'  segment {seg_idx}: checkpoint at local {cp_local_idx} (raw {raw_idx})')
        per_checkpoint_results.append(
            propagate_one_checkpoint(predictor, frame_dir, cp_local_idx, point, n_seg_frames, device))

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
    args = parser.parse_args()

    seq_dir = os.path.join(IN_DIR, f'{args.seq}_gt')
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    segments = find_segments(bbs)
    print(f'{args.seq}: {len(segments)} continuously-visible segments, '
          f'lengths {[e - s + 1 for s, e in segments]}')

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir_frames = os.path.join(args.tmp_dir, args.seq)
    csv_path = f'{args.seq}_pseudo_mask_diagnostics.csv'
    with open(csv_path, 'w') as f:
        f.write('segment_idx,local_idx,raw_idx,n_checkpoints_reached,min_pairwise_iou\n')
        for seg_idx, (start, end) in enumerate(segments):
            diagnostics = process_segment(predictor, seq_dir, bbs, start, end, seg_idx,
                                          out_dir_frames, device)
            for local_idx, n_reached, min_iou in diagnostics:
                f.write(f'{seg_idx},{local_idx},{start + local_idx},{n_reached},{min_iou}\n')

            n_zero = sum(1 for _, n, _ in diagnostics if n == 0)
            print(f'  segment {seg_idx}: {n_zero}/{len(diagnostics)} frames unlabeled '
                  f'({100 * n_zero / len(diagnostics):.1f}%)')

    print(f'wrote diagnostics to {csv_path}')


if __name__ == '__main__':
    main()
