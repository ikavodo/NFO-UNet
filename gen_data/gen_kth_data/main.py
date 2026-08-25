import argparse
import multiprocessing as mp
import os
from functools import partial, reduce
from typing import Tuple

import numpy as np
import cv2
from tqdm import tqdm

from gen_data.gen_kth_data import kth_config as conf
from gen_data.gen_kth_data.kth_utils import scale_and_pad_img_to_square
from utils.bb_utils import BoundingBox, save_bbs
from utils.fs_utils import File, search_files, ensure_dir, search_folders
from utils.gauss_utils import generate_gauss, generate_circle
from utils.img_utils import add_and_clip
from utils.occlusion_utils import generate_occlusion_morph, save_occlusion


def extract_bbs(file: File):
    with open(file.path) as f:
        lines = f.readlines()
    lines = [line.replace('\n', '').split(',') for line in lines]
    lines = [[float(v) for v in line] for line in lines]
    bbs = [BoundingBox(line[0], line[1], line[2], line[3]) for line in lines]
    return bbs


def generate_heatmap_gauss(shape: Tuple[int, int], gauss: np.ndarray, bb: BoundingBox):
    hm = np.zeros(shape, dtype=np.uint8)
    r_gauss = cv2.resize(gauss, (bb.w, bb.h))
    hm = add_and_clip(hm, r_gauss, (bb.y, bb.x))
    return hm


def extract_mean_bb(bbs, index):
    weights = c.bb_mean_weights
    weights_sum = sum(weights)
    weights = [v / weights_sum for v in weights]
    margin = len(weights) // 2
    if index - margin < 0 or index + margin >= len(bbs):
        return BoundingBox(-1, -1, 1, 1)

    try:
        relevant_bbs = bbs[index - margin:index + margin + 1]
        relevant_bbs = [bb.scale((weight, weight)) for weight, bb in zip(weights, relevant_bbs)]
        return reduce(lambda a, b: BoundingBox(a.x + b.x, a.y + b.y, a.w + b.w, a.h + b.h), relevant_bbs)
    except AttributeError:
        # this means that one of the BBs is None
        return BoundingBox(-1, -1, 1, 1)


def gen_seq(root: str, gauss: np.ndarray, skip_existing: bool = False):
    seq_path = os.path.join(c.out_dir, f'{os.path.basename(os.path.normpath(root))}_gt', '')
    if skip_existing and os.path.isfile(os.path.join(seq_path, 'groundtruth.txt')):
        return
    ensure_dir(seq_path)
    img_files = sorted(search_files(root, '.jpg'), key=lambda x: x.path)
    bbs = extract_bbs(search_files(root, 'groundtruth.txt')[0])
    assert len(img_files) == len(bbs)

    img_cache = []
    bb_cache = []

    for i, (img_file, bb) in enumerate(zip(img_files, bbs)):
        img = cv2.imread(img_file.path)[:, :, 0]
        bb = bb.scale((img.shape[1], img.shape[0]))
        img, bb = scale_and_pad_img_to_square(img, bb, c.img_size)
        bb = bb.round()
        img_cache.append(img)
        bb_cache.append(None if bb.x < 0 else bb)

    bb_dict = {}
    for i, img in enumerate(img_cache):
        bb_dict[i] = []
        cv2.imwrite(os.path.join(seq_path, f'{str(i).zfill(5)}_or.jpg'), img)
        mean_bb = extract_mean_bb(bb_cache, i)
        if mean_bb.x >= 0:
            mean_bb = mean_bb.round()
            hm_gauss = generate_heatmap_gauss(img.shape, gauss, mean_bb)
            hm_circle = generate_circle(img.shape, c.hm_circle_radius, mean_bb.center())
            cv2.imwrite(os.path.join(seq_path, f'{str(i).zfill(5)}_gauss.jpg'), hm_gauss)
            cv2.imwrite(os.path.join(seq_path, f'{str(i).zfill(5)}_circle.jpg'), hm_circle)
            mean_bb = mean_bb.scale((1 / c.img_size, 1 / c.img_size))
        bb_dict[i].append(mean_bb)
    save_bbs(bb_dict, os.path.join(seq_path, "groundtruth.txt"))


def gen_occlusion_sample(i: int, skip_existing: bool = False):
    out_path = os.path.join(c.out_dir, f'occlusion/{str(i).zfill(5)}.jpg')
    if skip_existing and os.path.isfile(out_path):
        return
    occlusion = generate_occlusion_morph((c.img_size, c.img_size), iterations=c.occ_iterations,
                                         init_occ_prob=c.occ_init_prob, occ_grow_prob=c.occ_growing_prob,
                                         occ_shrink_prob=c.occ_shrink_prob)
    save_occlusion(out_path, occlusion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip sequences/occlusion samples already written by a previous run')
    args = parser.parse_args()

    c = conf.config

    # do not change this... this one is configured such that it is exactly fitting the border of the image
    gauss = generate_gauss((198, 198), (99, 99), (30, 30)).astype(np.uint8)

    # sequences and occlusion samples are independent of each other -> parallelize both across processes
    roots = search_folders(c.in_dir, 'person', recursive=False)
    with mp.Pool(mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(partial(gen_seq, gauss=gauss, skip_existing=args.skip_existing), roots),
                  total=len(roots), desc='sequences'))
        list(tqdm(pool.imap_unordered(partial(gen_occlusion_sample, skip_existing=args.skip_existing),
                                       range(c.num_occ_samples)),
                  total=c.num_occ_samples, desc='occlusion'))
