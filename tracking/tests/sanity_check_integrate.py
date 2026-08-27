import os

import cv2
import numpy as np

from tracking.core.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval.eval_nfo import BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN
from tracking.core.integrate_image import integrate
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask


def load_sequence_prefix(seq, up_to):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    return np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(up_to)], axis=0)


def main():
    seq, center = 'seq1', 17
    frames_all = load_sequence_prefix(seq, center + SPAN + 1)
    masks_all = filter_by_shape(refine_mask(foreground_mask(frames_all, bg_frames=BG_FRAMES)))
    window_indices = list(range(center - SPAN, center + SPAN + 1, NTH_FRAME))
    frames = frames_all[window_indices]
    masks = masks_all[window_indices]
    detections = detect_blobs(masks)
    tracks = track_blobs(detections, max_dist=MAX_DIST)
    winner = score_and_fit(tracks, expected_height=EXPECTED_HEIGHT)
    assert winner is not None, "expected a track on this known-good window"

    for method in ['median', 'mean']:
        for mask_bg in [False, True]:
            img = integrate(frames, winner, detections=detections, merge_radius=MERGE_RADIUS,
                            frame_masks=masks, method=method, mask_background=mask_bg)
            assert img.shape == (220, 220)
            assert img.dtype == frames.dtype
            print(f"method={method} mask_background={mask_bg}: "
                  f"min={img.min()} max={img.max()} mean={img.mean():.1f}")

    img_gauss = integrate(frames, winner, method='gaussian', gaussian_sigma=1.5)
    assert img_gauss.shape == (220, 220)
    print(f"method=gaussian sigma=1.5: min={img_gauss.min()} max={img_gauss.max()} "
          f"mean={img_gauss.mean():.1f}")
    print("OK")


if __name__ == '__main__':
    main()
