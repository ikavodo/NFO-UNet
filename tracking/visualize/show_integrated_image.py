import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tracking.core.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval.eval_nfo import BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN
from tracking.core.integrate_image import integrate
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask


def load_sequence_prefix(seq, up_to):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    return np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(up_to)], axis=0)


def main(seq='seq1', center=17):
    frames_all = load_sequence_prefix(seq, center + SPAN + 1)
    masks_all = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))
    window_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = frames_all[window_indices]
    masks = masks_all[window_indices]
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT)

    configs = [
        ('median', False), ('mean', False), ('gaussian', False),
        ('median', True), ('mean', True), ('gaussian', True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for (method, mask_bg), ax in zip(configs, axes.flat):
        img = integrate(frames, winner, detections=detections, merge_radius=MERGE_RADIUS,
                        frame_masks=masks, method=method, gaussian_sigma=1.5, mask_background=mask_bg)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"{method}, {'masked' if mask_bg else 'full-frame'}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    out_path = 'tracking/visualize/integrated_image_comparison.png'
    plt.savefig(out_path, dpi=110)
    print(f"saved to {out_path}")


if __name__ == '__main__':
    main()
