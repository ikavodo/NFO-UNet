import os

import cv2
import numpy as np

from tracking.core.blob_tracker import detect_blobs, score_and_fit, track_blobs
from tracking.eval.eval_nfo import BG_FRAMES, EXPECTED_HEIGHT, MAX_DIST, MERGE_RADIUS, NTH_FRAME, SPAN
from tracking.core.integrate_image import align_frames, integrate
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask


def load_sequence_prefix(seq, up_to):
    seq_in = f'data/nfo_final/nfo_final/{seq}'
    jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
    return np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(up_to)], axis=0)


def check_alignment_follows_the_person():
    """align_frames must hold the PERSON still across the stack, not a fixed world point.
    A world-fixed window keeps a static occluder sharp and median-removes the moving person,
    which is the opposite of what fuse() exists to do. Regression test for the -vx*dt
    double-correction fixed 2026-08-31 (the person drifted +8.6px per frame before it).

    The winner dict is built by hand so this tests align_frames alone, with no dependence on
    the tracker or on any dataset.
    """
    H, W, vx, bar_w, bar_h, T = 200, 700, 8.0, 40, 120, 7
    frames, history = [], {}
    for t in range(T):
        f = np.full((H, W), 30, np.uint8)
        x = int(60 + vx * t)
        f[40:40 + bar_h, x:x + bar_w] = 255                 # the person
        for sx in range(0, W, 45):                          # static occluder, drawn on top
            f[:, sx:sx + 22] = 110
        frames.append(f)
        history[t] = (x + bar_w / 2, 40 + bar_h / 2, bar_h, bar_w, float('nan'), float('nan'))
    winner = dict(vx=vx, frames=list(range(T)), history=history)

    aligned = align_frames(np.stack(frames), winner, crop_size=int(1.6 * bar_h))
    xs = [float(np.nonzero(c > 200)[1].mean()) for c in aligned if (c > 200).any()]
    assert len(xs) == T, f"person visible in only {len(xs)}/{T} aligned crops"
    step = float(np.mean(np.diff(xs)))
    assert abs(step) < 2.0, (f"person drifts {step:+.1f}px per frame inside the aligned crop "
                             f"- align_frames is following a world point, not the person")
    print(f"alignment ok: person drifts {step:+.2f}px/frame inside the aligned crop")


def main():
    check_alignment_follows_the_person()
    seq, center = 'seq1', 17
    if not os.path.isdir(f'data/nfo_final/nfo_final/{seq}'):
        print(f"skip fusion checks: data/nfo_final/nfo_final/{seq} not present")
        return
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
