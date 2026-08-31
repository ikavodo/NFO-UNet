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
import multiprocessing as mp
import os
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

from utils.bb_utils import parse_bbs
from utils.gauss_utils import generate_gauss_2d

OUT_TAG = 'aniso'
IMG_SIZE = 224
# Coverage gate. Measured on this data: the two bounds are complementary and each catches the
# failure mode that exists in one domain only. On KTH (where this generator actually runs) the
# UPPER bound is load-bearing - it excludes 8.9% of frames, all of them over-segmentation
# blowouts where SAM grabbed background and the mask escaped the box, versus 4.4% excluded as
# low-coverage. On NFO the reverse holds: the coverage bound excludes 54-85% (most NFO
# pseudo-masks are fragments) while the upper bound excludes literally 0.0%, since those masks
# are clipped to the GT box by construction. So this is deliberately NOT a symmetric filter -
# do not read a single threshold pair as behaving the same way across both domains.
MIN_COVERAGE = 0.30   # (mask & box) / box_area - below this the mask is a fragment, and its
                      # 2nd moments describe a sliver rather than the person's gait shape
MAX_OUTSIDE = 0.25    # fraction of mask area falling OUTSIDE the box - above this the mask has
                      # blown out into background and its covariance is not the person's


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


def coverage_stats(mask_bin, bb, size=IMG_SIZE):
    """(coverage, outside_frac) for a mask against its GT box: how much of the box the mask
    covers, and how much of the mask escapes the box. Returns None if the box is degenerate."""
    x0, y0 = max(0, int(bb.x * size)), max(0, int(bb.y * size))
    x1, y1 = min(size, int((bb.x + bb.w) * size)), min(size, int((bb.y + bb.h) * size))
    if x1 <= x0 or y1 <= y0:
        return None
    total = mask_bin.sum()
    if total == 0:
        return None
    inbox = mask_bin[y0:y1, x0:x1].sum()
    return inbox / ((x1 - x0) * (y1 - y0)), 1 - inbox / total


def run_sequence(seq_dir: str, skip_existing: bool):
    n_written = n_bad_mask = n_existing = n_low_coverage = n_blown_out = 0
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    mask_files = sorted(f for f in os.listdir(seq_dir) if f.endswith('_sammask.png'))
    for fname in mask_files:
        idx = int(fname.split('_')[0])
        out_path = os.path.join(seq_dir, f'{idx:05d}_{OUT_TAG}.jpg')
        if skip_existing and os.path.exists(out_path):
            n_existing += 1
            continue

        mask = cv2.imread(os.path.join(seq_dir, fname), 0)
        mask_bin = (mask > 127).astype(np.uint8)
        params = mask_to_gauss_params(mask_bin)
        if params is None:
            n_bad_mask += 1
            continue

        # coverage gate - a fragment's or a blown-out mask's 2nd moments do not describe the
        # person's gait shape, so rendering a target from them teaches the wrong thing
        if idx in bbs and bbs[idx] and bbs[idx][0].x >= 0:
            cs = coverage_stats(mask_bin, bbs[idx][0])
            if cs is not None:
                coverage, outside = cs
                if coverage < MIN_COVERAGE:
                    n_low_coverage += 1
                    continue
                if outside > MAX_OUTSIDE:
                    n_blown_out += 1
                    continue

        mu, cov = params
        hm = generate_gauss_2d(mask.shape, mu, cov).astype(np.uint8)
        cv2.imwrite(out_path, hm)
        n_written += 1

    return n_written, n_bad_mask, n_existing, n_low_coverage, n_blown_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default='data/kth_processed')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    seq_dirs = [os.path.join(args.in_dir, d) for d in sorted(os.listdir(args.in_dir)) if d.endswith('_gt')]
    totals = np.zeros(5, dtype=np.int64)
    # sequences are fully independent (each writes only into its own directory) - parallelize
    # across processes the same way gen_kth_data/main.py already does, since this is a lot of
    # small per-frame numpy/cv2 work (moments + a 224x224 Gaussian render) across ~40k frames,
    # not something a GPU would meaningfully help with (tiny per-call arrays, transfer overhead
    # would dominate) - CPU parallelism across sequences is the actual bottleneck fix
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap_unordered(partial(run_sequence, skip_existing=args.skip_existing), seq_dirs),
            total=len(seq_dirs)))
    for r in results:
        totals += np.array(r, dtype=np.int64)
    written, bad_mask, existing, low_cov, blown = totals

    print(f'wrote {written} anisotropic heatmaps')
    print(f'skipped: {bad_mask} empty/degenerate mask, {low_cov} coverage < {MIN_COVERAGE} '
          f'(fragment), {blown} outside-box > {MAX_OUTSIDE} (blown out), {existing} already-existing')


if __name__ == '__main__':
    main()
