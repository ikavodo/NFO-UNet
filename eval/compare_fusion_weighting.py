"""Compare gait-consistency-weighted frame fusion against the existing fuse() methods.

Tests the proposal in docs/gait_integrated_image_segmentation.md: tracking/core/integrate_image.py's
'gaussian' fuse method weights frames purely by temporal distance |t - center|, and its own
docstring flags the resulting tension ("trades occlusion-robustness for pose fidelity - limb
articulation across a gait cycle changes shape frame to frame"). This replaces that weight with a
*gait-shape similarity* weight: how closely frame t's body configuration (2nd-moment orientation +
eccentricity) matches the center frame's, independent of temporal distance.

Setup (KTH, which has real per-frame masks so the comparison has ground truth):
  - take seq_size frames at nth_frame spacing around a center frame (matching kth_train's 7/2)
  - occlude each frame independently with generate_occlusion_branch (a different seed per frame,
    so occluders differ frame-to-frame - that difference is the entire reason fusion recovers
    anything)
  - align frames by GT bbox center (a stand-in for the classical tracker's Kalman motion
    estimate, which integrate_image.align_frames uses in the real pipeline)
  - fuse with each method, then score reconstruction against the CLEAN center frame, restricted
    to the center frame's own GT mask region (reconstructing the person is the goal; background
    is not)

Usage:
    python3 -m eval.compare_fusion_weighting --seq-dir data/kth_processed/person01_jogging_d1_uncomp_gt
"""
import argparse
import os

import cv2
import numpy as np

from utils.bb_utils import parse_bbs
from utils.occlusion_utils import augment_imgs_with_constant_occlusion, generate_occlusion_branch

IMG_SIZE = 224
OCC_COLOR = 60
TEMPORAL_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def shape_descriptor(mask: np.ndarray):
    """(orientation_deg, eccentricity) from a binary mask's 2nd central moments - the same
    descriptor whose gait-phase periodicity was measured directly on these masks, and which
    kth_train_anisotropic's predicted heatmap was shown to recover from raw pixels."""
    M = cv2.moments(mask, binaryImage=True)
    if M['m00'] == 0:
        return None
    mu20, mu02, mu11 = M['mu20'] / M['m00'], M['mu02'] / M['m00'], M['mu11'] / M['m00']
    eigvals, eigvecs = np.linalg.eigh(np.array([[mu02, mu11], [mu11, mu20]]))
    major, minor = eigvals[1], eigvals[0]
    vec = eigvecs[:, 1]
    return np.degrees(np.arctan2(vec[1], vec[0])), (np.sqrt(1 - minor / major) if major > 0 else 0.0)


def gait_weights(descriptors, center_i, sigma_angle=8.0, sigma_ecc=0.02):
    """Weight each frame by gait-shape similarity to the center frame. Angle difference is taken
    modulo 180 (an ellipse's major axis is undirected). Frames with no usable descriptor get the
    center frame's own weight floor rather than being dropped, so the weight vector always covers
    every frame."""
    ca, ce = descriptors[center_i]
    w = np.zeros(len(descriptors))
    for i, d in enumerate(descriptors):
        if d is None:
            continue
        a, e = d
        da = abs(((a - ca) + 90) % 180 - 90)
        w[i] = np.exp(-(da ** 2) / (2 * sigma_angle ** 2) - ((e - ce) ** 2) / (2 * sigma_ecc ** 2))
    return w / w.sum() if w.sum() > 0 else np.ones(len(descriptors)) / len(descriptors)


def temporal_gaussian_weights(n, center_i, sigma=2.0):
    """The weighting integrate_image.fuse(method='gaussian') already implements: a Gaussian in
    temporal distance from the center frame, with no awareness of pose at all."""
    dt = np.arange(n) - center_i
    w = np.exp(-(dt ** 2) / (2 * sigma ** 2))
    return w / w.sum()


def align_by_gt(frames, centers, center_i):
    """Translate every frame so its GT person center lands on the center frame's - a stand-in for
    integrate_image.align_frames' Kalman-motion-based alignment."""
    cx0, cy0 = centers[center_i]
    out = np.zeros_like(frames)
    for t in range(frames.shape[0]):
        dx, dy = cx0 - centers[t][0], cy0 - centers[t][1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out[t] = cv2.warpAffine(frames[t], M, (frames.shape[2], frames.shape[1]),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


def score(fused, clean_center, mask, recover_region=None):
    """Mean absolute reconstruction error against the clean center frame.

    `recover_region`: restrict scoring to person pixels that are actually OCCLUDED in the center
    frame - the only pixels fusion can possibly improve. Scoring over the whole mask instead is
    degenerate: the occluded center frame already has perfect pose, so "put ~all the weight on
    the center frame and don't really fuse" minimizes whole-mask error while removing no
    occluder at all (verified - temporal_gaussian(sigma=0.5), which puts ~79% of its weight on
    the center frame, won the whole-mask version outright).
    """
    m = mask.astype(bool) if recover_region is None else (mask.astype(bool) & recover_region)
    if not m.any():
        return None
    return float(np.abs(fused[m].astype(np.float64) - clean_center[m].astype(np.float64)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-dir', default='data/kth_processed/person01_jogging_d1_uncomp_gt')
    parser.add_argument('--seq-size', type=int, default=7)
    parser.add_argument('--nth-frame', type=int, default=2)
    parser.add_argument('--density', type=float, default=0.35)
    args = parser.parse_args()

    bbs = parse_bbs(os.path.join(args.seq_dir, 'groundtruth.txt'))
    margin = args.seq_size // 2
    span = margin * args.nth_frame

    def has_mask(i):
        return os.path.exists(os.path.join(args.seq_dir, f'{i:05d}_sammask.png'))

    usable = [i for i in sorted(bbs.keys())
              if has_mask(i) and all(has_mask(i + k * args.nth_frame) for k in range(-margin, margin + 1))
              and i in bbs and bbs[i] and bbs[i][0].x >= 0]
    if not usable:
        print('no frame has a full masked window available')
        return

    keys = ['no_fusion(center only)', 'mean', 'median', 'gait_weighted'] + \
           [f'temporal_gaussian(s={s})' for s in TEMPORAL_SIGMAS]
    results = {k: [] for k in keys}
    for center in usable:
        idxs = [center + k * args.nth_frame for k in range(-margin, margin + 1)]
        clean = np.stack([cv2.imread(os.path.join(args.seq_dir, f'{i:05d}_or.jpg'), 0) for i in idxs])
        masks = [(cv2.imread(os.path.join(args.seq_dir, f'{i:05d}_sammask.png'), 0) > 127).astype(np.uint8)
                 for i in idxs]

        # independent occluder per frame - different seeds, so the occluders move between frames
        occ_masks = [generate_occlusion_branch((IMG_SIZE, IMG_SIZE), density=args.density, seed=i)
                     for i in idxs]
        occluded = np.stack([
            augment_imgs_with_constant_occlusion(clean[t], occ_masks[t], OCC_COLOR)
            for t in range(len(idxs))])

        gt_centers = []
        for i in idxs:
            bb = bbs[i][0]
            cx, cy = bb.center()
            gt_centers.append((cx * IMG_SIZE, cy * IMG_SIZE))

        aligned = align_by_gt(occluded, gt_centers, margin)
        aligned_clean_center = clean[margin]
        center_mask = masks[margin]

        descriptors = [shape_descriptor(m) for m in masks]
        if descriptors[margin] is None:
            continue

        fusions = {
            'no_fusion(center only)': aligned[margin].astype(np.float64),
            'mean': aligned.mean(axis=0),
            'median': np.median(aligned, axis=0),
            'gait_weighted': np.tensordot(gait_weights(descriptors, margin),
                                          aligned.astype(np.float64), axes=(0, 0)),
        }
        # sweep the temporal baseline's own sigma so it gets the same tuning freedom
        # gait_weighted's two sigmas have - otherwise any gain could just be extra
        # hyperparameters rather than a genuinely better weighting signal
        for s in TEMPORAL_SIGMAS:
            fusions[f'temporal_gaussian(s={s})'] = np.tensordot(
                temporal_gaussian_weights(len(idxs), margin, sigma=s),
                aligned.astype(np.float64), axes=(0, 0))
        recover = occ_masks[margin]
        for k, fused in fusions.items():
            v = score(fused, aligned_clean_center, center_mask, recover_region=recover)
            if v is not None:
                results[k].append(v)

    n = len(results['mean'])
    print(f'{args.seq_dir}')
    print(f'n_windows={n}, seq_size={args.seq_size}, nth_frame={args.nth_frame}, '
          f'occlusion density={args.density}')
    print()
    print('reconstruction MAE vs. clean center frame, on person pixels OCCLUDED in the center')
    print('frame (the only pixels fusion can improve; lower = better):')
    for k in keys:
        arr = np.array(results[k])
        print(f'  {k:<28} mean={arr.mean():7.3f}  median={np.median(arr):7.3f}')


if __name__ == '__main__':
    main()
