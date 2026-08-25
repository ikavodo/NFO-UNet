import numpy as np
import cv2

from gen_data.gen_kth_data.main import extract_bbs
from tracking.track_window import track_window
from utils.fs_utils import File

SEQ_DIR = 'data/kth_staged/person01_walking_d1_uncomp'
SEQ_SIZE = 7
NTH_FRAME = 2  # matches config/train_config.py's nth_frame=2 (the paper's frame rate f=2)
MARGIN = SEQ_SIZE // 2
PIXEL_TOLERANCE = 20  # native KTH resolution is 160x120 - generous tolerance for a first pass


def find_valid_center(bbs):
    """First center frame index whose strided window (SEQ_SIZE frames, NTH_FRAME apart -
    matching AbstractDataSet.__getitem__'s own windowing convention) has a real
    (non-sentinel) ground-truth detection at every sampled frame, with some extra valid
    frames beyond the window too (avoids picking a center right at the edge of a valid
    stretch, e.g. the person just entering the frame)."""
    span = MARGIN * NTH_FRAME
    lookahead = 10
    for center in range(span, len(bbs) - span - lookahead):
        sampled = range(center - span, center + span + 1, NTH_FRAME)
        if all(bbs[i].x >= 0 for i in sampled) and all(bbs[center + span + j].x >= 0 for j in range(1, lookahead + 1)):
            return center
    raise RuntimeError(f"no stable valid-GT window found in {SEQ_DIR}")


def main():
    bbs = extract_bbs(File('groundtruth.txt', f'{SEQ_DIR}/groundtruth.txt'))
    center_idx = find_valid_center(bbs)
    span = MARGIN * NTH_FRAME
    frame_indices = list(range(center_idx - span, center_idx + span + 1, NTH_FRAME))

    frame_paths = [f'{SEQ_DIR}/{str(i).zfill(5)}.jpg' for i in frame_indices]
    frames = np.stack([cv2.imread(p, 0) for p in frame_paths], axis=0)
    assert frames.shape[0] == SEQ_SIZE and frames.ndim == 3, f"unexpected frames shape: {frames.shape}"
    h, w = frames.shape[1], frames.shape[2]

    # raw KTH groundtruth.txt is normalized [0,1] - scale to this video's actual pixel size
    gt_bb = bbs[center_idx].scale((w, h))
    gt_cx, gt_cy = gt_bb.center()

    result = track_window(frames)
    assert result is not None, f"track_window found no track in window centered at frame {center_idx}"

    dist = np.hypot(result['x'] - gt_cx, result['y'] - gt_cy)
    print(f"center_frame={center_idx} sampled_frames={frame_indices}")
    print(f"estimated=({result['x']:.1f}, {result['y']:.1f}) vx={result['vx']:.2f}px/frame "
          f"score={result['score']:.1f} resid_std={result['resid_std']:.2f}")
    print(f"ground truth=({gt_cx:.1f}, {gt_cy:.1f})")
    print(f"distance={dist:.1f}px (tolerance={PIXEL_TOLERANCE}px, frame size={w}x{h})")
    assert dist < PIXEL_TOLERANCE, f"estimated center {dist:.1f}px from ground truth, expected < {PIXEL_TOLERANCE}px"
    print("OK")


if __name__ == '__main__':
    main()
