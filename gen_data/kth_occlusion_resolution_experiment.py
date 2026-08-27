"""Quick experiment: composite synthetic occlusion onto a short, already-clean KTH segment that
has a real per-frame segmentation mask (gen_data/gen_sam_masks.py's *_sammask.png, computed by
single-image SAM2 on the *unoccluded* frame - reliable there since KTH scenes are clean and
unoccluded, so treated as ground truth here), then run the same multi-checkpoint SAM2 *video*
pipeline used for NFO (gen_nfo_pseudo_masks.py) on the occluded sequence, and score its output
directly against that real mask. Unlike NFO, where no real mask exists at all and quality could
only be judged by a proxy (box-recovery IoU) or eyeballing, this gives an actual ground-truth
number.

Resolution comparison: runs the whole thing at both the KTH pipeline's normal 224x224 and a
downsampled 112x112 (upsampled back to 224 only at scoring time) to test whether running SAM2 at
a coarser pixel grid hurts mask quality - the same question the native-vs-224 NFO comparison
(compare_resolution.py) was trying to answer, without needing native (pre-224) KTH source frames.

A single unbounded checkpoint at the segment midpoint is used (not the full geometric
multi-checkpoint machinery) - this is a short, single continuous span picked for a quick,
controlled check, not a full pipeline validation run.

Requires a GPU and the sam2 package (not available in this dev environment).

Usage:
    python3 -m gen_data.kth_occlusion_resolution_experiment
"""
import os
import shutil

import cv2
import numpy as np
import torch

from gen_data.gen_nfo_pseudo_masks import (combine_checkpoint_masks_union_gt_outlier,
                                           point_and_box_from_gt, propagate_one_checkpoint)
from utils.bb_utils import parse_bbs
from utils.occlusion_utils import augment_imgs_with_constant_occlusion, load_occlusion

SEQ_DIR = 'data/kth_processed/person01_jogging_d1_uncomp_gt'
OCCLUSION_PATH = 'data/kth_processed/occlusion/00000.jpg'
START, END = 14, 37  # contiguous span with real GT masks (_sammask.png) already on disk
OCC_COLOR = 60  # reasoned dark-gray default (not measured against real foliage tone), same
                # honesty-about-provenance convention as the other pipeline constants
TMP_DIR = 'tracking/kth_occlusion_experiment_tmp'
BASE_SIZE = 224  # KTH's usual processed resolution - the higher-pixel-density arm here


def stage_occluded_frames(frame_dir, size):
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir)
    occlusion = load_occlusion(OCCLUSION_PATH)
    if size != occlusion.shape[0]:
        occlusion = cv2.resize(occlusion.astype(np.uint8), (size, size)) > 0
    for local_idx, raw_idx in enumerate(range(START, END + 1)):
        img = cv2.imread(os.path.join(SEQ_DIR, f'{raw_idx:05d}_or.jpg'), 0)
        if size != img.shape[0]:
            img = cv2.resize(img, (size, size))
        occluded = augment_imgs_with_constant_occlusion(img, occlusion, OCC_COLOR)
        cv2.imwrite(os.path.join(frame_dir, f'{local_idx}.jpg'), occluded)


def run_arm(predictor, device, size, bbs):
    frame_dir = os.path.join(TMP_DIR, f'res{size}')
    stage_occluded_frames(frame_dir, size)

    n_seg_frames = END - START + 1
    cp_local_idx = n_seg_frames // 2
    point, box = point_and_box_from_gt(bbs, START + cp_local_idx, size, size)
    result = propagate_one_checkpoint(predictor, frame_dir, cp_local_idx, point, box, device)
    combined, _ = combine_checkpoint_masks_union_gt_outlier(
        [result], n_seg_frames, bbs, START, size, size,
        box_dilate_px=round(3 * size / BASE_SIZE), min_width_px=round(4 * size / BASE_SIZE))

    ious = []
    for local_idx, mask in combined.items():
        raw_idx = START + local_idx
        gt_path = os.path.join(SEQ_DIR, f'{raw_idx:05d}_sammask.png')
        if not os.path.exists(gt_path):
            continue
        gt = cv2.imread(gt_path, 0) > 127
        pred = mask
        if size != gt.shape[0]:
            pred = cv2.resize(mask.astype(np.uint8), gt.shape[::-1]) > 0
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        ious.append(inter / union if union else (1.0 if pred.sum() == 0 else 0.0))

    ious = np.array(ious)
    print(f'resolution {size}: n={len(ious)} mean_iou={ious.mean():.3f} '
          f'median={np.median(ious):.3f} frac_zero={(ious == 0).mean():.3f}')
    return ious


def main():
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2.1-hiera-large')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bbs = parse_bbs(os.path.join(SEQ_DIR, 'groundtruth.txt'))

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        run_arm(predictor, device, 224, bbs)
        run_arm(predictor, device, 112, bbs)


if __name__ == '__main__':
    main()
