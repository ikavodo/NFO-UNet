"""Shared helpers for locating and staging continuously-visible NFO segments, used by both the
single-segment diagnostic (prototype_sam2_video_segment.py) and the production multi-checkpoint
pseudo-mask generator (gen_nfo_pseudo_masks.py)."""
import os
import shutil


def find_segments(bbs):
    """Contiguous (start, end) inclusive raw-frame-index ranges with no '-1' sentinel row."""
    n_frames = max(bbs.keys()) + 1
    segments = []
    start = None
    for idx in range(n_frames):
        visible = idx in bbs and bbs[idx] and bbs[idx][0].x >= 0
        if visible and start is None:
            start = idx
        elif not visible and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, n_frames - 1))
    return segments


def stage_frames(seq_dir, start, end, frame_dir):
    # SAM2's video loader sorts by int(filename) - sequential local names avoid depending
    # on the original zero-padded, sequence-relative _or.jpg indexing
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)
    for local_idx, raw_idx in enumerate(range(start, end + 1)):
        src = os.path.abspath(os.path.join(seq_dir, f'{raw_idx:05d}_or.jpg'))
        os.symlink(src, os.path.join(frame_dir, f'{local_idx}.jpg'))
