import os

import cv2
import numpy as np

from tracking.core.track_sequence import track_windows_in_sequence

IN_DIR = 'data/nfo_final/nfo_final'
SEQ_SIZE = 7  # matches config/train_config.py's seq_size=7
NTH_FRAME = 2  # matches config/train_config.py's nth_frame=2
MARGIN = SEQ_SIZE // 2
SPAN = MARGIN * NTH_FRAME

# derived from NFO's own native-resolution (800x600) ground truth, not reused/rescaled
# from KTH - see conversation for the measurement:
# - MAX_DIST from measured GT centroid displacement at nth_frame=2 stride (p99 ~= 25px)
# - MERGE_RADIUS from measured person height (~195px mean) / 2
MAX_DIST = 25.0
MERGE_RADIUS = 100.0
EXPECTED_HEIGHT = 195.0  # measured mean NFO person height at native 800x600 resolution
BG_FRAMES = 30  # must suit the earliest window queried - see track_sequence's docstring

SEQS = ['seq1', 'seq2', 'seq3', 'seq4']


def parse_normalized_bbs(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        x, y, w, h = (float(v) for v in line.strip().split(','))
        boxes.append(None if x < 0 else (x, y, w, h))
    return boxes


def eval_sequence(seq, use_shape_scoring):
    seq_in = os.path.join(IN_DIR, seq)
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    norm_file = next(f for f in os.listdir(seq_in) if f != 'groundtruth.txt' and f.startswith('groundtruth'))
    boxes = parse_normalized_bbs(os.path.join(seq_in, norm_file))
    n = min(len(jpgs), len(boxes))

    valid_centers = [c for c in range(SPAN, n - SPAN) if boxes[c] is not None]
    frames_all = np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(n)], axis=0)
    h, w = frames_all.shape[1], frames_all.shape[2]

    expected_height = EXPECTED_HEIGHT if use_shape_scoring else None
    results = track_windows_in_sequence(frames_all, valid_centers, span=SPAN, nth_frame=NTH_FRAME,
                                        max_dist=MAX_DIST, merge_radius=MERGE_RADIUS,
                                        expected_height=expected_height, bg_frames=BG_FRAMES)

    residuals, n_no_track = [], 0
    for center in valid_centers:
        result = results[center]
        if result is None:
            n_no_track += 1
            continue
        gt = boxes[center]
        gt_cx, gt_cy = (gt[0] + gt[2] / 2) * w, (gt[1] + gt[3] / 2) * h
        dist_norm = np.hypot((result['x'] - gt_cx) / w, (result['y'] - gt_cy) / h)
        residuals.append(dist_norm)

    return residuals, n_no_track, len(valid_centers)


def run(use_shape_scoring):
    label = "WITH shape-aware scoring + sequence warm-start" if use_shape_scoring else \
        "WITHOUT shape-aware scoring, WITH sequence warm-start"
    print(f"=== {label} ===")
    all_residuals, total_no_track, total_valid = [], 0, 0
    for seq in SEQS:
        residuals, n_no_track, n_valid = eval_sequence(seq, use_shape_scoring)
        all_residuals.extend(residuals)
        total_no_track += n_no_track
        total_valid += n_valid
        msg = f"mean_resid={np.mean(residuals):.4f}" if residuals else "no tracks found"
        print(f"{seq}: valid_centers={n_valid} no_track={n_no_track} tracked={len(residuals)} {msg}")

    all_residuals = np.array(all_residuals)
    print()
    print(f"TOTAL valid_centers={total_valid} no_track={total_no_track} "
          f"({100 * total_no_track / total_valid:.1f}%) tracked={len(all_residuals)}")
    if len(all_residuals):
        print(f"residual (normalized [0,1] units, diagonal distance): "
              f"mean={all_residuals.mean():.4f} median={np.median(all_residuals):.4f} "
              f"p90={np.percentile(all_residuals, 90):.4f} p99={np.percentile(all_residuals, 99):.4f} "
              f"max={all_residuals.max():.4f}")
        print("for reference: the eval pipeline's max_dist_error threshold is 0.1 (10% of frame)")
    print()


def main():
    run(use_shape_scoring=False)
    run(use_shape_scoring=True)


if __name__ == '__main__':
    main()
