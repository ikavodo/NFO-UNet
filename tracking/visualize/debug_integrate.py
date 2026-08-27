import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracking.core.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval.eval_nfo import BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN
from tracking.core.integrate_image import align_frames, anchor_for_frame, crop_at, fuse, restrict_to_nearby
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask

CROP = 220


def load_sequence_prefix(seq, up_to):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    return np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(up_to)], axis=0)


def main(seq='seq2', center=355):
    frames_all = load_sequence_prefix(seq, center + SPAN + 1)
    masks_all = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))
    window_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = frames_all[window_indices]
    masks = masks_all[window_indices]
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT)
    T = frames.shape[0]
    center_t = T // 2

    # per-frame: raw aligned crop, mask aligned crop, and an overlay (mask boundary in
    # red on top of the raw grayscale) so we can see exactly what the mask is pointing at
    raw_aligned, mask_aligned, overlays = [], [], []
    for t in range(T):
        ax, ay = anchor_for_frame(winner, t)
        person_mask = restrict_to_nearby(masks[t], masks[t], detections[t], ax, ay, MERGE_RADIUS)
        dt = t - center_t
        shifted_x = ax - winner['vx'] * dt
        raw_crop = crop_at(frames[t], shifted_x, ay, CROP)
        mask_crop = crop_at(person_mask, shifted_x, ay, CROP)
        raw_aligned.append(raw_crop)
        mask_aligned.append(mask_crop)

        overlay = cv2.cvtColor(raw_crop, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours((mask_crop > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
        overlays.append(overlay)

    raw_aligned = np.stack(raw_aligned)
    fused_raw_median = fuse(raw_aligned, method='median')
    fused_mask = np.max(np.stack(mask_aligned), axis=0)

    fig, axes = plt.subplots(4, T, figsize=(2.2 * T, 9))
    for t in range(T):
        axes[0][t].imshow(raw_aligned[t], cmap='gray', vmin=0, vmax=255)
        axes[0][t].axis('off')
        axes[1][t].imshow(mask_aligned[t], cmap='gray', vmin=0, vmax=255)
        axes[1][t].axis('off')
        axes[2][t].imshow(overlays[t])
        axes[2][t].axis('off')
    axes[0][0].set_title('raw aligned grayscale', loc='left', fontsize=9)
    axes[1][0].set_title('mask aligned (same as aligned_masks.png)', loc='left', fontsize=9)
    axes[2][0].set_title('overlay: mask boundary (red) on raw pixels', loc='left', fontsize=9)
    for a in axes[3]:
        a.axis('off')
    axes[3][T // 2 - 1].imshow(fused_raw_median, cmap='gray', vmin=0, vmax=255)
    axes[3][T // 2 - 1].set_title('fused raw (median)', fontsize=9)
    axes[3][T // 2 + 1].imshow(fused_mask, cmap='gray', vmin=0, vmax=255)
    axes[3][T // 2 + 1].set_title('fused mask (max)', fontsize=9)

    plt.tight_layout()
    out_path = 'tracking/visualize/debug_integrate.png'
    plt.savefig(out_path, dpi=110)
    print(f"saved to {out_path}")


if __name__ == '__main__':
    main()
