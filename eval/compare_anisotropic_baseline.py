"""Compare the anisotropic-heatmap-trained UNet (kth_train_anisotropic) against the existing
circle-trained baseline (kth_train) on real KTH validation data:
1. Localization accuracy - apples-to-apples, same MaxEval peak-extraction for both models.
2. Gait-signal recovery (anisotropic model only) - does its predicted heatmap's own 2nd-moment
   shape (eccentricity/orientation) track the real GT mask's, the same diagnostic already run
   directly on GT masks with zero training involved.

See /home/akovi/.claude/plans/sparkling-munching-valiant.md, step 5.

Usage:
    python3 -m eval.compare_anisotropic_baseline --baseline-dir out/kth_train_20260826_123145 \
        --anisotropic-dir out/kth_train_anisotropic_20260828_145304
"""
import argparse
import os

import cv2
import numpy as np
import torch

import config.train_config as tc
from dataset.kth_dataset import KthDataSet
from eval.max_eval import MaxEval
from network.unet import UNet


def load_model(save_dir: str, n_channels: int) -> UNet:
    model = UNet(n_channels=n_channels, n_classes=1, bilinear=False)
    model.load_checkpoint(save_dir)
    model.eval()
    return model


def weighted_centroid(hm: np.ndarray, window: int = 40):
    """First-moment (mean-position) centroid within a local window around the heatmap's argmax
    - 'regresses' the isotropic (position-only) quantity out of a richer anisotropic prediction
    by marginalizing out its shape/covariance content within the locally-relevant region, unlike
    MaxEval's single-pixel argmax (a poor position estimator for a spread-out, low-peak
    anisotropic target on its own).

    Restricted to a local window rather than the whole image: a whole-image weighted centroid
    is highly sensitive to diffuse low-level "ambient" output values far from the true target,
    and a model trained with MSELoss against a smooth Gaussian target naturally produces a
    smoother, more diffuse output than one trained with LogisticLoss against a near-binary
    circle target - a whole-image centroid comparison inherits that loss-function-driven output-
    statistics difference as a confound, not a genuine position-accuracy difference (verified:
    an earlier whole-image version of this function reported the anisotropic model as far worse
    than even raw argmax, which the windowed version does not reproduce - see git history).
    """
    h, w = hm.shape
    max_idx = np.argmax(hm)
    ay, ax = max_idx // w, max_idx % w
    y0, y1 = max(0, ay - window), min(h, ay + window + 1)
    x0, x1 = max(0, ax - window), min(w, ax + window + 1)
    local = np.clip(hm[y0:y1, x0:x1], 0, None).astype(np.float64)
    total = local.sum()
    if total <= 0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cy = (ys * local).sum() / total
    cx = (xs * local).sum() / total
    return cx, cy


def moments_of(mask: np.ndarray):
    """Same eccentricity/orientation extraction used in this session's earlier GT-mask-only
    gait diagnostic - returns (angle_deg, eccentricity), or None if mask has no mass."""
    M = cv2.moments(mask, binaryImage=True)
    if M['m00'] == 0:
        return None
    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    cov = np.array([[mu02, mu11], [mu11, mu20]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    major, minor = eigvals[1], eigvals[0]
    vec = eigvecs[:, 1]
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    eccen = np.sqrt(1 - minor / major) if major > 0 else 0.0
    return angle, eccen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--anisotropic-dir', required=True)
    parser.add_argument('--n-samples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    tc.set_cfg('kth_train')
    c = tc.config
    baseline = load_model(args.baseline_dir, c.seq_size)
    aniso = load_model(args.anisotropic_dir, c.seq_size)

    ds = KthDataSet(root_dir=c.eval_data, seq_length=c.seq_size, nth_frame=c.nth_frame,
                    exclude_roots=[], transforms=[])
    max_eval = MaxEval(max_dist_error=25.0)

    rng = np.random.RandomState(args.seed)
    n = min(args.n_samples, len(ds))
    sample_idxs = rng.choice(len(ds), n, replace=False)

    dist_baseline_argmax, dist_aniso_argmax = [], []
    dist_baseline_centroid, dist_aniso_centroid = [], []
    ecc_err, angle_err = [], []
    n_gait_compared = 0

    with torch.no_grad():
        for i in sample_idxs:
            item = ds[int(i)]
            frame_bbs = item['bbs']
            if not frame_bbs or frame_bbs[0].x < 0:
                continue
            gt_cx, gt_cy = frame_bbs[0].center()
            gt_cx, gt_cy = gt_cx * 224, gt_cy * 224

            frames = item['frames'].unsqueeze(0)
            pred_b = baseline(frames)[0, 0].numpy()
            pred_a = aniso(frames)[0, 0].numpy()

            # argmax (MaxEval, matches the existing project convention) - a fine estimator for
            # the isotropic/circle target, a poor one for the anisotropic target (see
            # weighted_centroid's docstring)
            (centers_b, _) = max_eval.extract_centers(pred_b)
            (centers_a, _) = max_eval.extract_centers(pred_a)
            if centers_b:
                pb_x, pb_y = centers_b[0]
                dist_baseline_argmax.append(np.hypot(pb_x - gt_cx, pb_y - gt_cy))
            if centers_a:
                pa_x, pa_y = centers_a[0]
                dist_aniso_argmax.append(np.hypot(pa_x - gt_cx, pa_y - gt_cy))

            # weighted centroid - the fair comparison: regresses the position-only (isotropic)
            # quantity out of each model's full prediction, computed identically for both
            cb = weighted_centroid(pred_b)
            ca = weighted_centroid(pred_a)
            if cb:
                dist_baseline_centroid.append(np.hypot(cb[0] - gt_cx, cb[1] - gt_cy))
            if ca:
                dist_aniso_centroid.append(np.hypot(ca[0] - gt_cx, ca[1] - gt_cy))

            # gait-signal recovery: only where a real mask exists to compare against
            entry = ds.entries[ds.idx_mapping[int(i)]]
            if entry.aniso_file is None:
                continue
            mask_path = entry.aniso_file.path.replace('_aniso.jpg', '_sammask.png')
            if not os.path.exists(mask_path):
                continue
            gt_mask = (cv2.imread(mask_path, 0) > 127).astype(np.uint8)
            gt_moments = moments_of(gt_mask)
            pred_mask = (pred_a > np.percentile(pred_a, 99)).astype(np.uint8)
            pred_moments = moments_of(pred_mask)
            if gt_moments is None or pred_moments is None:
                continue
            angle_err.append(abs(((pred_moments[0] - gt_moments[0]) + 90) % 180 - 90))
            ecc_err.append(abs(pred_moments[1] - gt_moments[1]))
            n_gait_compared += 1

    print(f'n_samples with valid GT: baseline={len(dist_baseline_argmax)}, anisotropic={len(dist_aniso_argmax)}')
    print(f'localization error (px, distance to GT bbox center):')
    print(f'  [argmax]   baseline (circle): mean={np.mean(dist_baseline_argmax):.2f}  median={np.median(dist_baseline_argmax):.2f}')
    print(f'  [argmax]   anisotropic:       mean={np.mean(dist_aniso_argmax):.2f}  median={np.median(dist_aniso_argmax):.2f}')
    print(f'  [centroid] baseline (circle): mean={np.mean(dist_baseline_centroid):.2f}  median={np.median(dist_baseline_centroid):.2f}')
    print(f'  [centroid] anisotropic:       mean={np.mean(dist_aniso_centroid):.2f}  median={np.median(dist_aniso_centroid):.2f}')
    print(f'  (centroid is the fair comparison - see weighted_centroid docstring; argmax kept for reference)')
    print()
    print(f'gait-signal recovery (anisotropic model predicted heatmap vs. real GT mask), n={n_gait_compared}:')
    if n_gait_compared:
        print(f'  orientation error (deg): mean={np.mean(angle_err):.1f}  median={np.median(angle_err):.1f}')
        print(f'  eccentricity error:      mean={np.mean(ecc_err):.3f}  median={np.median(ecc_err):.3f}')


if __name__ == '__main__':
    main()
