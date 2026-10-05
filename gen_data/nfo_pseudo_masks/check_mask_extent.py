"""Acceptance test for the NFO pseudo-masks: do they reach the extent of their GT box?

    python -m gen_data.nfo_pseudo_masks.check_mask_extent
    python -m gen_data.nfo_pseudo_masks.check_mask_extent --mask-subdir old_sammask   # the backup, in data/nfo_processed/_archive/<seq>/

Written for the gt_to_native fix (6606a1a). Before it, every mask stopped short of the feet:
0/3496 reached their GT box bottom, median bottom gap 0.098-0.218 of box height per sequence.
gen_nfo_pseudo_masks.py's own *_pseudo_mask_diagnostics.csv cannot catch that - it measures
agreement BETWEEN checkpoints, which were all mis-prompted the same way and so agree on a short
mask - so this compares each mask directly against its own GT box instead.

Per sequence it reports:
  bottom gap   (box bottom - mask bottom) / box height. ~0 after the fix; ~0.1-0.2 before.
  top gap      (mask top - box top) / box height. The old bug mis-placed the top edge too.
  reach bottom fraction of masks ending within --tol-px of the box bottom
  pad leak     masks with pixels in the replicate-padded band (rows < 28 or > 195 at 224 for an
               800x600 source). scale_and_pad_img_to_square pads masks with BORDER_REPLICATE, so
               a mask touching the native frame edge gets smeared through the pad band.
"""
import argparse
import os

import cv2
import numpy as np

from utils.bb_utils import parse_bbs

ROOT = 'data/nfo_processed'
PAD_BAND = 28      # (800 - 600) / 2 rows * 224/800, for NFO's 800x600 native frames


def check_sequence(seq_dir, mask_dir, tol_px):
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    bottom, top, reach, leak, empty, n = [], [], 0, 0, 0, 0
    for f in sorted(os.listdir(mask_dir)):
        if not f.endswith('_sammask.png'):
            continue
        idx = int(f[:5])
        if idx not in bbs or not bbs[idx] or bbs[idx][0].x < 0:
            continue
        m = cv2.imread(os.path.join(mask_dir, f), 0) > 127
        if not m.any():
            empty += 1
            continue
        H = m.shape[0]
        bb = bbs[idx][0]
        rows = np.where(m.any(axis=1))[0]
        box_top, box_bot, box_h = bb.y * H, (bb.y + bb.h) * H, bb.h * H
        bottom.append((box_bot - (rows.max() + 1)) / box_h)
        top.append((rows.min() - box_top) / box_h)
        reach += (rows.max() + 1) >= box_bot - tol_px
        leak += bool(m[:PAD_BAND].any() or m[H - PAD_BAND:].any())
        n += 1
    return np.array(bottom), np.array(top), reach, leak, empty, n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=ROOT)
    p.add_argument('--seqs', nargs='*', default=None)
    p.add_argument('--mask-subdir', default='',
                   help="masks in data/nfo_processed/_archive/<seq>/<subdir> instead of <seq_dir> (e.g. old_sammask backup)")
    p.add_argument('--tol-px', type=float, default=2.0, help='"reaches bottom" tolerance, 224-px units')
    a = p.parse_args()

    seqs = a.seqs or sorted(d for d in os.listdir(a.root) if d.endswith('_gt'))
    print(f"{'seq':<9}{'n':>6}{'empty':>7}  {'bottom gap med / p10 / p90':<28}{'top gap med':>12}"
          f"{'reach bottom':>15}{'pad leak':>10}")
    for seq in seqs:
        seq_dir = os.path.join(a.root, seq)
        mask_dir = (os.path.join(os.path.dirname(seq_dir), '_archive', os.path.basename(seq_dir).removesuffix('_gt'),
                                 a.mask_subdir) if a.mask_subdir else seq_dir)
        b, t, reach, leak, empty, n = check_sequence(seq_dir, mask_dir, a.tol_px)
        if n == 0:
            print(f'{seq:<9} no masks with a GT box found in {mask_dir}')
            continue
        print(f'{seq:<9}{n:>6}{empty:>7}  {np.median(b):>6.3f} / {np.percentile(b, 10):>6.3f} / '
              f'{np.percentile(b, 90):>6.3f}   {np.median(t):>9.3f}{reach:>9}/{n:<5}{leak:>7}')
    print('\nPass: bottom-gap median ~0 and most masks reach the bottom. '
          'Before the fix: median 0.098-0.218, reach 0/n.')


if __name__ == '__main__':
    main()
