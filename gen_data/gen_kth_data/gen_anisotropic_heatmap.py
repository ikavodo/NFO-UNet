"""Generate anisotropic (covariance-shaped) heatmap targets for KTH from real per-frame
segmentation masks (gen_data/gen_sam_masks.py's *_sammask.png) - part of the anisotropic-heatmap
training experiment (see /home/akovi/.claude/plans/sparkling-munching-valiant.md).

Unlike the existing _gauss.jpg (a fixed isotropic kernel resized to the bounding box) or
_circle.jpg (a fixed-radius disk), this computes each frame's real mask mean/covariance via
image moments and renders a true covariance-parameterized 2D Gaussian
(utils/gauss_utils.py:generate_gauss_2d) - encoding gait-phase-relevant body shape/orientation,
not just position. Requires *_sammask.png to already exist (gen_sam_masks.py must be run first).

Usage:
    python3 -m gen_data.gen_kth_data.gen_anisotropic_heatmap --in-dir data/kth_processed
"""
import argparse
import os

import cv2
import numpy as np

from utils.gauss_utils import generate_gauss_2d

OUT_TAG = 'aniso'


def mask_to_gauss_params(mask: np.ndarray):
    """Returns ((mean_y, mean_x), 2x2 covariance in (y,x) order) from a binary mask's image
    moments, or None if the mask is empty or its covariance is degenerate (near-singular -
    e.g. a one-pixel-wide sliver has ~zero variance along one axis)."""
    M = cv2.moments(mask, binaryImage=True)
    if M['m00'] == 0:
        return None
    cy, cx = M['m01'] / M['m00'], M['m10'] / M['m00']
    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    cov = np.array([[mu02, mu11], [mu11, mu20]])
    if np.linalg.matrix_rank(cov) < 2 or np.linalg.det(cov) < 1e-3:
        return None
    return (cy, cx), cov


def run_sequence(seq_dir: str, skip_existing: bool):
    n_written = n_skipped_bad_mask = n_skipped_existing = 0
    mask_files = sorted(f for f in os.listdir(seq_dir) if f.endswith('_sammask.png'))
    for fname in mask_files:
        idx = int(fname.split('_')[0])
        out_path = os.path.join(seq_dir, f'{idx:05d}_{OUT_TAG}.jpg')
        if skip_existing and os.path.exists(out_path):
            n_skipped_existing += 1
            continue

        mask = cv2.imread(os.path.join(seq_dir, fname), 0)
        mask_bin = (mask > 127).astype(np.uint8)
        params = mask_to_gauss_params(mask_bin)
        if params is None:
            n_skipped_bad_mask += 1
            continue

        mu, cov = params
        hm = generate_gauss_2d(mask.shape, mu, cov).astype(np.uint8)
        cv2.imwrite(out_path, hm)
        n_written += 1

    return n_written, n_skipped_bad_mask, n_skipped_existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default='data/kth_processed')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    seq_dirs = sorted(d for d in os.listdir(args.in_dir) if d.endswith('_gt'))
    total_written = total_bad_mask = total_existing = 0
    for seq in seq_dirs:
        w, bm, e = run_sequence(os.path.join(args.in_dir, seq), args.skip_existing)
        total_written += w
        total_bad_mask += bm
        total_existing += e

    print(f'wrote {total_written} anisotropic heatmaps, skipped {total_bad_mask} frames with '
          f'empty/degenerate masks, {total_existing} already-existing')


if __name__ == '__main__':
    main()
