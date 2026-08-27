import os

import cv2

from gen_data.gen_kth_data.kth_config import config as kth_gen_config
from gen_data.gen_kth_data.kth_utils import scale_and_pad_img_to_square
from utils.bb_utils import BoundingBox, save_bbs

IN_DIR = 'data/nfo_final/nfo_final'
OUT_DIR = 'data/nfo_processed'
# match KTH's own gen_kth_data resolution - the U-Net is trained at this size
IMG_SIZE = kth_gen_config.img_size


def parse_normalized_bbs(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    bb_dict = {}
    for i, line in enumerate(lines):
        x, y, w, h = (float(v) for v in line.strip().split(','))
        bb_dict[i] = None if x < 0 else BoundingBox(x, y, w, h)
    return bb_dict


def main():
    for seq in sorted(os.listdir(IN_DIR)):
        seq_in = os.path.join(IN_DIR, seq)
        if not os.path.isdir(seq_in):
            continue
        seq_out = os.path.join(OUT_DIR, f'{seq}_gt')
        os.makedirs(seq_out, exist_ok=True)

        jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))

        # each seq dir has 'groundtruth.txt' (raw pixel coords) plus one
        # inconsistently-named normalized variant (groundtruth_norm.txt /
        # groundtruth_normalized.txt / groundtruth_nromalized.txt) - use the latter
        norm_file = next(f for f in os.listdir(seq_in) if f != 'groundtruth.txt' and f.startswith('groundtruth'))
        bb_dict = parse_normalized_bbs(os.path.join(seq_in, norm_file))
        # labels align to the first len(bb_dict) frames; any trailing unlabeled frames are dropped
        if len(bb_dict) < len(jpgs):
            print(f'{seq}: only {len(bb_dict)}/{len(jpgs)} frames labeled, dropping trailing unlabeled frames')
        jpgs = jpgs[:len(bb_dict)]

        out_bbs = {}
        for i, fname in enumerate(jpgs):
            img = cv2.imread(os.path.join(seq_in, fname), 0)
            bb = bb_dict[i]
            # scale_and_pad_img_to_square needs a real BoundingBox even for frames with no
            # detection - its returned bb is simply discarded below in that case
            abs_bb = bb.scale((img.shape[1], img.shape[0])) if bb is not None else BoundingBox(0, 0, 0, 0)
            img, abs_bb = scale_and_pad_img_to_square(img, abs_bb, IMG_SIZE)
            out_path = os.path.join(seq_out, f'{str(i).zfill(5)}_or.jpg')
            if os.path.islink(out_path):
                # a previous version of this script wrote symlinks to the original source
                # images at these paths - cv2.imwrite would follow the symlink and
                # overwrite the source itself, so remove the symlink (not its target) first
                os.unlink(out_path)
            cv2.imwrite(out_path, img)
            out_bbs[i] = [] if bb is None else [abs_bb.scale((1 / IMG_SIZE, 1 / IMG_SIZE))]

        save_bbs(out_bbs, os.path.join(seq_out, 'groundtruth.txt'))


if __name__ == '__main__':
    main()
