"""Geometric good-visibility detection for NFO, replacing YOLO for checkpoint seeding.

The camera is static, so occluder positions (tree trunks, bushes) are fixed for an entire
sequence. This finds the small number of "clear corridor" x-ranges wide enough to fully contain
a person (not just their centroid) by analyzing a clean background image (median of the many
no-person frames every sequence already has), then cross-references each segment's GT
trajectory to find frames where the person is confirmed to be fully visible - no detector
needed at all, purely scene geometry + ground truth.

Validated against real backgrounds for seq1-4: median-background + column-mean darkness +
Otsu threshold + merge (gap<=10px) + width filter (>=85% of the sequence's own median GT box
width - tolerates a region just under a full body width) matches manual visual review of all
four scenes. seq1/seq3 have discrete bare-branch trunks with real gaps; seq2/seq4 have denser,
more continuous foliage but still yield usable corridors at this calibration.
"""
import os

import cv2
import numpy as np

from gen_data.nfo_segment_utils import find_segments

MERGE_GAP = 10  # px of occluder between two clear runs small enough to treat as one corridor
BG_SAMPLES = 40  # no-person frames to median-combine into the background image

# Manually curated clear-region x-ranges (raw px, 224-wide frames), one round of automated
# tuning per sequence proved too fragile (mean vs. min vs. frac-occluded each got some
# sequences right and others wrong, with no single knob setting working everywhere) - eyeballed
# and picked the best-looking automated result per sequence instead of chasing one unified
# formula further. seq1/seq4: column-min method, tighter margin - 2 checkpoints per segment.
# (Briefly widened seq1 to 4 regions to fix large unlabeled tails, but that traded coverage for
# more chances of an individual checkpoint's tracker getting stuck near an occluder with nothing
# to catch it pre-union - reverted once GT box+point prompting alone closed most of the coverage
# gap; see docs/nfo_pseudo_segmentation_approach.md for the tradeoff discussion.) seq2: original
# column-mean result plus a manually-confirmed third corridor at ~73% horizontal (a real gap
# next to a diagonal branch against the background wall, missed by every automated width
# threshold tried). seq3: original column-mean result, unchanged throughout.
CURATED_CLEAR_REGIONS = {
    'seq1': [(44, 85), (154, 203)],
    'seq2': [(2, 38), (56, 117), (153, 174)],
    'seq3': [(2, 30), (110, 166)],
    'seq4': [(18, 49), (151, 181)],
}


def compute_background_image(seq_dir, bbs, n_frames, n_samples=BG_SAMPLES):
    segments = find_segments(bbs)
    occupied = set()
    for s, e in segments:
        occupied.update(range(s, e + 1))
    bg_idxs = [i for i in range(n_frames) if i not in occupied]
    if not bg_idxs:
        raise RuntimeError('no no-person frames available to build a background image')
    sample = bg_idxs[::max(1, len(bg_idxs) // n_samples)][:n_samples]
    frames = np.stack([cv2.imread(os.path.join(seq_dir, f'{i:05d}_or.jpg'), 0) for i in sample])
    return np.median(frames, axis=0).astype(np.uint8)


MIN_WIDTH_FRAC = 0.85  # of median GT box width - tolerates a region just under a full body
                        # width, where the person is still likely mostly visible


def median_box_width_px(bbs, img_w):
    widths = sorted(bb.w * img_w for bbs_list in bbs.values() for bb in bbs_list if bb.x >= 0)
    return widths[len(widths) // 2]


def y_band_for_sequence(bbs, img_h, margin=20):
    """Walking-path height band, from the full range of GT box y-centers across the sequence."""
    ys = [(bb.y + bb.h / 2) * img_h for bbs_list in bbs.values() for bb in bbs_list if bb.x >= 0]
    return max(0, int(min(ys) - margin)), min(img_h, int(max(ys) + margin))


def find_clear_regions(bg, y_band, min_width, merge_gap=MERGE_GAP):
    """Returns merged, width-filtered [(x_start, x_end), ...] clear (unoccluded) x-ranges,
    using the column-wise mean intensity within the band + Otsu threshold."""
    y_lo, y_hi = y_band
    col_mean = bg[y_lo:y_hi, :].mean(axis=0)
    thresh, _ = cv2.threshold(col_mean.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clear_mask = col_mean > thresh

    runs = []
    start = None
    for x, is_clear in enumerate(clear_mask):
        if is_clear and start is None:
            start = x
        elif not is_clear and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, len(clear_mask) - 1))

    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if e - s + 1 >= min_width]


def default_clear_regions(seq_dir, bbs, n_frames, img_h, img_w):
    """Curated regions for seq1-4 (see CURATED_CLEAR_REGIONS docstring above); falls back to
    the automated column-mean/Otsu detection for any other sequence."""
    seq_name = os.path.basename(seq_dir).removesuffix('_gt')
    if seq_name in CURATED_CLEAR_REGIONS:
        return CURATED_CLEAR_REGIONS[seq_name]
    bg = compute_background_image(seq_dir, bbs, n_frames)
    min_width = MIN_WIDTH_FRAC * median_box_width_px(bbs, img_w)
    y_band = y_band_for_sequence(bbs, img_h)
    return find_clear_regions(bg, y_band, min_width=min_width)


def confirmed_clear_frames(bbs, start, end, clear_regions, img_w):
    """Raw frame indices in [start, end] where the GT box is fully contained in some clear
    region (full-body visibility, not just centroid-in-region)."""
    confirmed = []
    for idx in range(start, end + 1):
        if idx not in bbs or not bbs[idx] or bbs[idx][0].x < 0:
            continue
        bb = bbs[idx][0]
        x0, x1 = bb.x * img_w, (bb.x + bb.w) * img_w
        if any(rs <= x0 and x1 <= re for rs, re in clear_regions):
            confirmed.append(idx)
    return confirmed


def geometric_checkpoints(bbs, start, end, clear_regions, img_w):
    """One representative raw frame index per distinct, contiguous pass through a clear
    region (the middle of each such 'visibility burst') - these are the checkpoint seed
    candidates, replacing YOLO/naive-fraction selection entirely. Returns a sorted list,
    possibly empty (no confirmed-clear frame anywhere in the segment) or length 1+."""
    confirmed = confirmed_clear_frames(bbs, start, end, clear_regions, img_w)
    if not confirmed:
        return []
    bursts = []
    burst_start = confirmed[0]
    prev = confirmed[0]
    for idx in confirmed[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        bursts.append((burst_start, prev))
        burst_start = idx
        prev = idx
    bursts.append((burst_start, prev))
    return [(s + e) // 2 for s, e in bursts]
