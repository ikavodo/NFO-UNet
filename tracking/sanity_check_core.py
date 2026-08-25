import numpy as np

from tracking.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.blob_tracker import detect_blobs


def make_synthetic_window():
    """A dark background with a bright square moving right at 10px/frame."""
    frames = np.full((5, 60, 80), 30, dtype=np.uint8)
    for t in range(5):
        x = 10 + t * 10
        frames[t, 20:40, x:x + 15] = 220
    return frames


def main():
    frames = make_synthetic_window()
    masks = foreground_mask(frames)
    masks = refine_mask(masks)
    masks = filter_by_shape(masks)
    assert masks.shape == frames.shape, f"shape changed: {masks.shape} vs {frames.shape}"
    assert masks.dtype == np.uint8

    detections = detect_blobs(masks)
    assert len(detections) == 5
    found = sum(1 for d in detections if len(d) > 0)
    # MOG2's first frame typically has no learned background yet, so allow one miss
    assert found >= 4, f"expected blobs detected in most frames, got {found}/5"
    print(f"OK - detected blobs in {found}/5 synthetic frames")


if __name__ == '__main__':
    main()
