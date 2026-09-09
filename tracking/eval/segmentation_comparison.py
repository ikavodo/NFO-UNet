"""SAM2 (confirmed-clear init, continuous propagation) vs our method, same clip, same moments.

    python -m tracking.eval.segmentation_comparison

Built for one specific poster requirement: a DIRECT, apples-to-apples comparison, not two
separately-styled figures asserted to be comparable. Both rows show the exact same 6 frames of
seq1_gt run 4 (f1216, 1234, 1252, 1271, 1289, 1308 - sam2_drift_check.py's own sample points,
picked there by evenly spacing across its 93-frame evaluated span). SAM2's row is cropped directly
from its own already-rendered montage (images/stream/seq1_gt_sam2_video_drift_frames_box_restart.png)
rather than rerun, so it is provably the same result already reported, not a fresh, possibly
different one.

93 = 31x3 exactly, so "our method" tiles the identical span with three independent 31-frame
windows (f1216-1246, f1247-1277, f1278-1308) instead of one continuous propagation - this IS the
actual difference being illustrated: SAM2 gets one continuous attempt and wobbles even from a good
start; our method re-anchors every 31 frames, each window independently verified against its own
motion-integrated reference, so nothing propagates across a window boundary to drift in the first
place. Two of the six shared frames fall in each window (see FRAME_TO_WINDOW below).

Reuses gt_sam_gate.py's render_row unchanged (via its new abs_frames param) for the actual masking
- this script only assembles pre-existing pieces into one comparison image, it does not recompute
anything gt_sam_gate.py doesn't already do.
"""
import os

import cv2
import numpy as np

from tracking.core.integrate_image import align_frames  # noqa: F401 (re-exported via gt_sam_gate)
from tracking.eval.gt_integrated_image import build_gt_winner
from tracking.eval.gt_sam_gate import render_row
from tracking.eval.lookbehind_discrimination import gt_runs, load_sequence

SEQ = 'seq1_gt'
RUN_INDEX = 4
SAM2_MONTAGE = 'images/stream/seq1_gt_sam2_video_drift_frames_box_restart.png'
FRAMES = (1216, 1234, 1252, 1271, 1289, 1308)          # sam2_drift_check.py's own sample points
WINDOWS = ((1216, 1246), (1247, 1277), (1278, 1308))   # 93 = 31x3, tiles the identical span
TILE = 280


def frames_for_window(window):
    return [f for f in FRAMES if window[0] <= f <= window[1]]


def main():
    frames, gt = load_sequence(f'data/nfo_processed/{SEQ}')
    T_all, H, W = frames.shape
    run = gt_runs(gt)[RUN_INDEX]
    assert run == (1154, 1308), f'seq1_gt run {RUN_INDEX} changed to {run} - update WINDOWS/FRAMES'

    our_tiles = []
    for window in WINDOWS:
        abs_idx = list(range(window[0], window[1] + 1))
        winner, _ = build_gt_winner(gt, window, W, H)
        row = render_row(winner, frames[abs_idx], abs_idx, frames, H, W, crop_mult=4.5,
                         samples=0, label='', abs_frames=frames_for_window(window))
        # render_row's first 260x260 tile is its own reference-image panel, which the SAM2 row
        # has no equivalent of - drop it so the two rows compare panel-for-panel, not a reference
        # tile against a raw frame.
        strip = row[:, 260:]
        our_tiles.append(cv2.resize(strip, (TILE * len(frames_for_window(window)), TILE)))
    our_row = np.hstack(our_tiles)

    sam2_row = cv2.imread(SAM2_MONTAGE)
    assert sam2_row is not None, f'missing {SAM2_MONTAGE} - run sam2_drift_check.py first'
    assert sam2_row.shape[1] == our_row.shape[1], \
        f'width mismatch: SAM2 row {sam2_row.shape[1]}px vs ours {our_row.shape[1]}px'

    def label_bar(width, text):
        bar = np.zeros((28, width, 3), np.uint8)
        cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return bar

    montage = np.vstack([
        label_bar(sam2_row.shape[1], 'SAM2, confirmed-clear init, ONE continuous propagation'),
        sam2_row,
        label_bar(our_row.shape[1], 'Our method, THREE independent 31-frame windows, same clip/frames'),
        our_row,
    ])
    out = 'images/stream/seq1_gt_segmentation_comparison.png'
    cv2.imwrite(out, montage)
    print(f'wrote {out}  ({montage.shape[1]}x{montage.shape[0]})')
    print(f'shared frames: {FRAMES}')
    print(f'windows: {WINDOWS}')


if __name__ == '__main__':
    main()
