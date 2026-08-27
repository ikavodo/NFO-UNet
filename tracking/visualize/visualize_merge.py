import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracking.core.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval.eval_nfo import MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN, parse_normalized_bbs
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask

GT_COLOR = (255, 0, 0)      # red - ground truth
EST_COLOR = (0, 255, 0)     # green - tracker's estimate
MARKER_SIZE = 18


def find_center_valid_throughout(boxes, span, nth_frame):
    """First center index whose ENTIRE sampled window (not just the center frame) has a
    real, non-sentinel ground-truth box at every sampled frame."""
    for center in range(span, len(boxes) - span):
        sampled = range(center - span, center + span + 1, nth_frame)
        if all(boxes[i] is not None for i in sampled):
            return center
    raise RuntimeError("no window with valid GT throughout was found")


def load_window(seq, center):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    frame_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in frame_indices], axis=0)
    return frames


def mark(gray_img, gt_xy, est_xy=None):
    """Grayscale -> BGR-as-RGB with GT (red) and estimate (green, if given) marked."""
    color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    cv2.drawMarker(color, (int(gt_xy[0]), int(gt_xy[1])), GT_COLOR,
                    markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=3)
    if est_xy is not None:
        cv2.drawMarker(color, (int(est_xy[0]), int(est_xy[1])), EST_COLOR,
                        markerType=cv2.MARKER_CROSS, markerSize=MARKER_SIZE, thickness=3)
    return color


def build(seq, center, boxes, expected_height=None):
    frames = load_window(seq, center)
    masks = filter_by_shape(refine_mask(foreground_mask(frames)))
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=expected_height)

    T = frames.shape[0]
    frame_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    h, w = frames.shape[1], frames.shape[2]

    marked_orig, marked_mask = [], []
    for t in range(T):
        gt = boxes[frame_indices[t]]
        gt_xy = ((gt[0] + gt[2] / 2) * w, (gt[1] + gt[3] / 2) * h)
        est_xy = winner['history'][t] if (winner is not None and t in winner['history']) else None
        marked_orig.append(mark(frames[t], gt_xy, est_xy))
        marked_mask.append(mark(masks[t], gt_xy, est_xy))
    return marked_orig, marked_mask, winner


def render(seq, center, boxes, title, ax_rows, expected_height=None):
    marked_orig, marked_mask, winner = build(seq, center, boxes, expected_height=expected_height)
    for t in range(len(marked_orig)):
        ax_rows[0][t].imshow(marked_orig[t])
        ax_rows[0][t].axis('off')
        ax_rows[1][t].imshow(marked_mask[t])
        ax_rows[1][t].axis('off')
    ax_rows[0][0].set_title(title, loc='left', fontsize=10)
    vx = winner['vx'] if winner is not None else float('nan')
    score = winner['score'] if winner is not None else None
    print(f"{title}: winner vx={vx:.2f}px/frame, score={score} "
          f"(red=ground truth, green=tracker estimate)")


def load_boxes(seq):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    norm_file = next(f for f in os.listdir(seq_in) if f != 'groundtruth.txt' and f.startswith('groundtruth'))
    return parse_normalized_bbs(os.path.join(seq_in, norm_file))


EXPECTED_HEIGHT = 195.0  # matches tracking/eval_nfo.py


def main():
    fig, axes = plt.subplots(4, 7, figsize=(18, 9))

    boxes1 = load_boxes('seq1')
    center1 = find_center_valid_throughout(boxes1, SPAN, NTH_FRAME)
    render('seq1', center1, boxes1, f'seq1 (good case) center={center1} - orig / mask', axes[0:2, :],
          expected_height=EXPECTED_HEIGHT)

    boxes2 = load_boxes('seq2')
    center2 = find_center_valid_throughout(boxes2, SPAN, NTH_FRAME)
    render('seq2', center2, boxes2,
          f'seq2 (was bad case) center={center2} - WITH shape-aware scoring - orig / mask',
          axes[2:4, :], expected_height=EXPECTED_HEIGHT)

    plt.tight_layout()
    out_path = 'tracking/visualize/merge_visualization.png'
    plt.savefig(out_path, dpi=110)
    print(f"saved to {out_path}")


if __name__ == '__main__':
    main()
