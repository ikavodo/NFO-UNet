"""Score every localizable NFO frame with an off-the-shelf YOLO person detector.

Purpose: pick better SAM2 checkpoint seed frames than a blind fractional index (0.25/0.5/0.75
of a segment) - the exact fractional frame might land mid-stride or partially occluded, while a
nearby frame with a clean, high-confidence YOLO detection is a better place to seed tracking
from. This script only produces the confidence scores; gen_nfo_pseudo_masks.py optionally
consumes them (--yolo-scores) to search a small window around each target fraction.

Requires a GPU and the ultralytics package (not available in this dev environment):
    pip install ultralytics

Usage:
    python3 -m gen_data.score_frames_yolo --seq seq1
"""
import argparse
import os

import cv2
import numpy as np

from gen_data.nfo_segment_utils import find_segments
from utils.bb_utils import parse_bbs

IN_DIR = 'data/nfo_processed'
PERSON_CLASS = 0  # COCO


def score_sequence(seq_dir, bbs, model, batch_size):
    localizable = sorted(idx for idx in bbs if bbs[idx] and bbs[idx][0].x >= 0)
    scores = {}
    for i in range(0, len(localizable), batch_size):
        chunk = localizable[i:i + batch_size]
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(seq_dir, f'{idx:05d}_or.jpg'), 0),
                             cv2.COLOR_GRAY2BGR) for idx in chunk]
        results = model(imgs, classes=[PERSON_CLASS], verbose=False)
        for idx, result in zip(chunk, results):
            confs = result.boxes.conf
            scores[idx] = float(confs.max()) if len(confs) else 0.0
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', required=True, help='e.g. seq1')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    seq_dir = os.path.join(IN_DIR, f'{args.seq}_gt')
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    segments = find_segments(bbs)
    print(f'{args.seq}: {len(segments)} segments, '
          f'{sum(e - s + 1 for s, e in segments)} localizable frames to score')

    from ultralytics import YOLO
    model = YOLO(args.model)

    scores = score_sequence(seq_dir, bbs, model, args.batch_size)

    out_path = f'{args.seq}_yolo_scores.csv'
    with open(out_path, 'w') as f:
        f.write('raw_idx,person_conf\n')
        for idx in sorted(scores):
            f.write(f'{idx},{scores[idx]}\n')

    confs = np.array(list(scores.values()))
    print(f'wrote {len(scores)} scores to {out_path}')
    print(f'conf stats: mean={confs.mean():.3f} median={np.median(confs):.3f} '
          f'frac_zero={(confs == 0).mean():.3f} frac_above_0.5={(confs > 0.5).mean():.3f}')


if __name__ == '__main__':
    main()
