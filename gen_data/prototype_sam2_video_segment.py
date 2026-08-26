"""Diagnostic: single-segment SAM2 video tracking on NFO.

Prompts once at the middle frame of one continuously-visible NFO segment (a run with no
'-1' sentinel gaps), then propagates the mask forward to the segment's end and backward to
its start using SAM2's video memory. Purpose: see whether/where tracking drifts over a long,
fragmented-occlusion segment before deciding whether smaller/overlapping chunks are needed -
this is a single-segment sanity check, not the production pseudo-labeling pipeline.

Requires a GPU and the sam2 package (not available in this dev environment):
    git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .

Usage:
    python3 -m gen_data.prototype_sam2_video_segment --seq seq1 --out-dir tracking/sam2_video_prototype
"""
import argparse
import os

import cv2
import numpy as np
import torch

from gen_data.nfo_segment_utils import find_segments, stage_frames
from utils.bb_utils import parse_bbs

IN_DIR = 'data/nfo_processed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', required=True, help='e.g. seq1')
    parser.add_argument('--segment-idx', type=int, default=None,
                        help='which continuously-visible segment to use (default: the longest one)')
    parser.add_argument('--prompt-frac', type=float, default=0.5,
                        help='where in the segment to prompt, as a fraction of its length')
    parser.add_argument('--model-id', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--out-dir', default='tracking/sam2_video_prototype')
    args = parser.parse_args()

    seq_dir = os.path.join(IN_DIR, f'{args.seq}_gt')
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    segments = find_segments(bbs)
    if not segments:
        raise RuntimeError(f'no continuously-visible segments found in {seq_dir}')

    seg_idx = args.segment_idx if args.segment_idx is not None else \
        max(range(len(segments)), key=lambda i: segments[i][1] - segments[i][0])
    start, end = segments[seg_idx]
    n_seg_frames = end - start + 1
    print(f'using segment {seg_idx} of {len(segments)}: raw frames [{start}, {end}] ({n_seg_frames} frames)')

    frame_dir = os.path.join(args.out_dir, f'{args.seq}_seg{seg_idx}_frames')
    stage_frames(seq_dir, start, end, frame_dir)

    prompt_local_idx = int(round((n_seg_frames - 1) * args.prompt_frac))
    prompt_raw_idx = start + prompt_local_idx
    bb = bbs[prompt_raw_idx][0]
    img_h, img_w = cv2.imread(os.path.join(seq_dir, f'{prompt_raw_idx:05d}_or.jpg'), 0).shape
    point = ((bb.x + bb.w / 2) * img_w, (bb.y + bb.h / 2) * img_h)
    print(f'prompting at local frame {prompt_local_idx} (raw {prompt_raw_idx}), point={point}')

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(args.model_id)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        state = predictor.init_state(frame_dir)
        predictor.add_new_points_or_box(state, frame_idx=prompt_local_idx, obj_id=1,
                                        points=[point], labels=[1])

        for frame_idx, _, masks in predictor.propagate_in_video(state, start_frame_idx=prompt_local_idx,
                                                                  reverse=False):
            results[frame_idx] = (masks[0] > 0).cpu().numpy().squeeze()
        for frame_idx, _, masks in predictor.propagate_in_video(state, start_frame_idx=prompt_local_idx,
                                                                  reverse=True):
            results[frame_idx] = (masks[0] > 0).cpu().numpy().squeeze()

    overlay_dir = os.path.join(args.out_dir, f'{args.seq}_seg{seg_idx}_overlays')
    os.makedirs(overlay_dir, exist_ok=True)
    diagnostics = []  # local_idx, area, centroid_x, centroid_y - eyeball for sudden jumps = drift
    for local_idx in sorted(results.keys()):
        mask = results[local_idx]
        frame = cv2.imread(os.path.join(frame_dir, f'{local_idx}.jpg'), 0)
        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        color = (255, 0, 0) if local_idx == prompt_local_idx else (0, 0, 255)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1)
        cv2.imwrite(os.path.join(overlay_dir, f'{local_idx:05d}.png'), overlay)

        area = int(mask.sum())
        if area:
            ys, xs = np.nonzero(mask)
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            cx, cy = float('nan'), float('nan')
        diagnostics.append((local_idx, area, cx, cy))

    csv_path = os.path.join(args.out_dir, f'{args.seq}_seg{seg_idx}_diagnostics.csv')
    with open(csv_path, 'w') as f:
        f.write('local_idx,area,centroid_x,centroid_y\n')
        for row in diagnostics:
            f.write(','.join(str(v) for v in row) + '\n')

    print(f'wrote {len(diagnostics)} overlays to {overlay_dir}')
    print(f'wrote diagnostics to {csv_path}')


if __name__ == '__main__':
    main()
