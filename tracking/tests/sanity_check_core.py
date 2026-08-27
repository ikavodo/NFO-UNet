import numpy as np

from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.blob_tracker import detect_blobs


SEQ_SIZE = 7  # matches config/train_config.py's seq_size=7


def make_synthetic_window():
    """A dark background with a bright square moving right at 10px/frame."""
    frames = np.full((SEQ_SIZE, 60, 80), 30, dtype=np.uint8)
    for t in range(SEQ_SIZE):
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
    assert len(detections) == SEQ_SIZE
    found = sum(1 for d in detections if len(d) > 0)
    # MOG2's first frame typically has no learned background yet, so allow one miss
    assert found >= SEQ_SIZE - 1, f"expected blobs detected in most frames, got {found}/{SEQ_SIZE}"
    print(f"OK - detected blobs in {found}/{SEQ_SIZE} synthetic frames")


if __name__ == '__main__':
    main()
