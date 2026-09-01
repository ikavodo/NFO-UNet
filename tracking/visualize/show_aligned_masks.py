import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracking.core.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask
from tracking.eval.eval_nfo import (BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS,
                                    NTH_FRAME, SEQS, SPAN, load_frames, seq_dir)
from tracking.core.integrate_image import crop_at, anchor_for_frame, restrict_to_nearby

# CROP as a FRACTION of frame height, not 220px. 220/600 reproduces the original value exactly
# on nfo_final and gives 82px on nfo_processed's 224x224 frames - where a literal 220 would be
# the whole frame, so the script would still "run" while silently showing an uncropped image
# labelled as a person crop. Scale-free is the only way this survives a change of dataset.
CROP_FRAC = 220 / 600


# loader moved to tracking.eval.eval_nfo.load_frames so the dataset path lives in one place


def main(seq=SEQS[0], center=17):
    frames_all = load_frames(seq, up_to=center + SPAN + 1)
    CROP = int(round(CROP_FRAC * frames_all.shape[1]))
    masks_all = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))
    window_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = frames_all[window_indices]
    masks = masks_all[window_indices]
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT)
    T = frames.shape[0]
    center_t = T // 2

    unaligned, aligned = [], []
    for t in range(T):
        ax, ay = anchor_for_frame(winner, t)
        person_mask = restrict_to_nearby(masks[t], masks[t], detections[t], ax, ay, MERGE_RADIUS)
        unaligned.append(crop_at(person_mask, ax, ay, CROP))
        # aligned: shift the crop center by -vx*dt so every frame samples the same
        # real-world point as the center frame (this tracker's motion model is
        # horizontal-only, matching its constant-velocity Kalman assumption)
        dt = t - center_t
        aligned.append(crop_at(person_mask, ax - winner['vx'] * dt, ay, CROP))

    merged = np.zeros((CROP, CROP), dtype=np.uint8)
    for a in aligned:
        merged = np.maximum(merged, a)

    fig, axes = plt.subplots(3, T, figsize=(2.2 * T, 7))
    for t in range(T):
        axes[0][t].imshow(unaligned[t], cmap='gray', vmin=0, vmax=255)
        axes[0][t].axis('off')
        axes[1][t].imshow(aligned[t], cmap='gray', vmin=0, vmax=255)
        axes[1][t].axis('off')
    axes[0][0].set_title('unaligned (crop follows tracked point per-frame)', loc='left', fontsize=9)
    axes[1][0].set_title('aligned (motion-compensated to center frame)', loc='left', fontsize=9)
    for a in axes[2]:
        a.axis('off')
    axes[2][T // 2].imshow(merged, cmap='gray', vmin=0, vmax=255)
    axes[2][T // 2].set_title('merged (max over aligned masks)', fontsize=9)

    plt.tight_layout()
    out_path = 'tracking/visualize/aligned_masks.png'
    plt.savefig(out_path, dpi=110)
    print(f"vx={winner['vx']:.2f}px/frame, saved to {out_path}")


if __name__ == '__main__':
    main()
