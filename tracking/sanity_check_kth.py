import numpy as np
import cv2

from gen_data.gen_kth_data.main import extract_bbs
from tracking.track_window import track_window
from utils.fs_utils import File

SEQ_DIR = 'data/kth_staged/person01_walking_d1_uncomp'
SEQ_SIZE = 5
PIXEL_TOLERANCE = 20  # native KTH resolution is 160x120 - generous tolerance for a first pass


def find_valid_window(bbs, seq_size):
    """First window where every frame (not just the center) has a real (non-sentinel)
    ground-truth detection, well inside a GT-valid stretch rather than right at its edge
    (e.g. the person just entering the frame)."""
    margin = seq_size // 2
    lookahead = 10  # require this many more valid frames after the window too
    for start in range(len(bbs) - seq_size - lookahead):
        window = bbs[start:start + seq_size + lookahead]
        if all(bb.x >= 0 for bb in window):
            return start
    raise RuntimeError(f"no stable valid-GT window found in {SEQ_DIR}")


def main():
    bbs = extract_bbs(File('groundtruth.txt', f'{SEQ_DIR}/groundtruth.txt'))
    start = find_valid_window(bbs, SEQ_SIZE)
    center_idx = start + SEQ_SIZE // 2

    frame_paths = [f'{SEQ_DIR}/{str(i).zfill(5)}.jpg' for i in range(start, start + SEQ_SIZE)]
    frames = np.stack([cv2.imread(p, 0) for p in frame_paths], axis=0)
    assert frames.shape[0] == SEQ_SIZE and frames.ndim == 3, f"unexpected frames shape: {frames.shape}"
    h, w = frames.shape[1], frames.shape[2]

    # raw KTH groundtruth.txt is normalized [0,1] - scale to this video's actual pixel size
    gt_bb = bbs[center_idx].scale((w, h))
    gt_cx, gt_cy = gt_bb.center()

    result = track_window(frames)
    assert result is not None, f"track_window found no track in window starting at frame {start}"

    dist = np.hypot(result['x'] - gt_cx, result['y'] - gt_cy)
    print(f"window start={start} center_frame={center_idx}")
    print(f"estimated=({result['x']:.1f}, {result['y']:.1f}) vx={result['vx']:.2f}px/frame "
          f"score={result['score']:.1f} resid_std={result['resid_std']:.2f}")
    print(f"ground truth=({gt_cx:.1f}, {gt_cy:.1f})")
    print(f"distance={dist:.1f}px (tolerance={PIXEL_TOLERANCE}px, frame size={w}x{h})")
    assert dist < PIXEL_TOLERANCE, f"estimated center {dist:.1f}px from ground truth, expected < {PIXEL_TOLERANCE}px"
    print("OK")


if __name__ == '__main__':
    main()
