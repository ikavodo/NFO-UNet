"""Generate per-frame person segmentation pseudo-labels for KTH using SAM2, prompted by the
existing heatmap ground truth (its peak pixel as a single positive point prompt).

Requires a GPU and the sam2 package, neither available in this dev environment - meant to be
run on a remote GPU box:
    git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .
(needs torch>=2.5.1, torchvision>=0.20.1 - see https://github.com/facebookresearch/sam2 for the
checkpoint download / INSTALL.md if the pip build has issues.)

Usage:
    python3 -m gen_data.gen_sam_masks --in-dir data/kth_processed --hm-filter circle
"""
import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.bb_utils import parse_bbs

OUT_TAG = 'sammask'  # written as {idx:05d}_sammask.png alongside the existing *_or.jpg etc.


def point_prompt_from_heatmap(hm_path: str):
    hm = cv2.imread(hm_path, 0)
    idx = np.argmax(hm)
    y, x = np.unravel_index(idx, hm.shape)
    return int(x), int(y)


def run_sequence(predictor, seq_dir: str, hm_tag: str, skip_existing: bool, batch_size: int):
    bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
    n_written, n_skipped_no_person, n_skipped_existing = 0, 0, 0

    todo = []  # (out_path, frame_rgb, point)
    for idx in sorted(bbs.keys()):
        # sentinel "no person in this frame" row - see prep_nfo_data.py / kth gen for the
        # same convention
        if any(bb.x < 0 for bb in bbs[idx]):
            n_skipped_no_person += 1
            continue

        out_path = os.path.join(seq_dir, f'{idx:05d}_{OUT_TAG}.png')
        if skip_existing and os.path.exists(out_path):
            n_skipped_existing += 1
            continue

        hm_path = os.path.join(seq_dir, f'{idx:05d}_{hm_tag}.jpg')
        or_path = os.path.join(seq_dir, f'{idx:05d}_or.jpg')
        if not (os.path.exists(hm_path) and os.path.exists(or_path)):
            continue

        point = point_prompt_from_heatmap(hm_path)
        frame_rgb = cv2.cvtColor(cv2.imread(or_path, 0), cv2.COLOR_GRAY2RGB)
        todo.append((out_path, frame_rgb, point))

    # batched: one backbone forward pass per chunk instead of per frame - this is the
    # expensive part for a model this size, the per-frame prompt decode is cheap either way
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        predictor.set_image_batch([c[1] for c in chunk])
        masks_batch, _, _ = predictor.predict_batch(
            point_coords_batch=[np.array([c[2]]) for c in chunk],
            point_labels_batch=[np.array([1]) for c in chunk],
            multimask_output=False)
        for (out_path, _, _), masks in zip(chunk, masks_batch):
            cv2.imwrite(out_path, (masks[0] * 255).astype(np.uint8))
            n_written += 1

    return n_written, n_skipped_no_person, n_skipped_existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default='data/kth_processed')
    parser.add_argument('--hm-filter', choices=['circle', 'gauss'], default='circle',
                        help='which heatmap tag to read the point prompt from - match training config')
    parser.add_argument('--model-id', default='facebook/sam2.1-hiera-large')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='frames per set_image_batch call - raise while GPU memory allows, '
                             'this is what actually speeds things up (amortizes the backbone pass)')
    args = parser.parse_args()

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(args.model_id)

    seq_dirs = sorted(d for d in os.listdir(args.in_dir) if d.endswith('_gt'))
    total_written = total_no_person = total_existing = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=(device == 'cuda')):
        for seq in tqdm(seq_dirs):
            w, np_, e = run_sequence(predictor, os.path.join(args.in_dir, seq), args.hm_filter,
                                     args.skip_existing, args.batch_size)
            total_written += w
            total_no_person += np_
            total_existing += e

    print(f'wrote {total_written} masks, skipped {total_no_person} no-person frames, '
          f'{total_existing} already-existing masks')


if __name__ == '__main__':
    main()
