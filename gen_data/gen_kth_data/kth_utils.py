from typing import Tuple, List

import numpy as np
import cv2

from utils.bb_utils import BoundingBox


def pad_img_and_bb_to_square(img: np.ndarray, bb: BoundingBox) -> Tuple[np.ndarray, BoundingBox]:
    shape = img.shape
    pad_value = abs(shape[0] - shape[1]) // 2
    pad = (0 if shape[0] <= shape[1] else pad_value, 0 if shape[1] <= shape[0] else pad_value)
    img = cv2.copyMakeBorder(img, pad[1], pad[1], pad[0], pad[0], cv2.BORDER_REPLICATE, value=0)
    bbs = bb.translate(pad)

    assert img.shape[0] == img.shape[1]
    return img, bbs


def scale_img_and_bb(img: np.ndarray, bb: BoundingBox, scale: Tuple[int, int]) -> Tuple[np.ndarray, BoundingBox]:
    before_shape = img.shape
    img = cv2.resize(img, (scale[0], scale[1]))
    scale = (img.shape[1] / before_shape[1], img.shape[0] / before_shape[0])
    bbs = bb.scale(scale)
    return img, bbs


def scale_and_pad_img_to_square(img: np.ndarray, bb: BoundingBox, size: int) -> Tuple[np.ndarray, BoundingBox]:
    img, bb = pad_img_and_bb_to_square(img, bb)
    img, bb = scale_img_and_bb(img, bb, (size, size))
    return img, bb


def parse_kth_bbs(file_path: str) -> List[BoundingBox]:
    bbs = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = [float(coord) for coord in line.replace(' ', '').replace('\n', '').split(',')]
        bbs.append(BoundingBox(x=data[0], y=data[1], w=data[2], h=data[3]))
    return bbs
