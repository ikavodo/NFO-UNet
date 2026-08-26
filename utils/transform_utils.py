import random
from typing import List, Callable, Tuple, Union

import numpy as np
import torch
import cv2
from torchvision.transforms import transforms

from utils.bb_utils import BoundingBox
from utils.fs_utils import File
from utils.occlusion_utils import augment_imgs_with_noisy_occlusion, augment_imgs_with_constant_occlusion, \
    load_occlusion

Transform = Callable[[np.ndarray, Union[np.ndarray, None], Union[List[BoundingBox], None]],
                     Union[Tuple[np.ndarray, np.ndarray, List[BoundingBox]],
                           Tuple[torch.Tensor, torch.Tensor, List[float]]]]


def chain(transforms: List[Transform]) -> Transform:
    def t(imgs, hm, bbs):
        for t in transforms:
            imgs, hm, bbs = t(imgs, hm, bbs)
        return imgs, hm, bbs

    return t


def rand_h_flip(img_width: float = 1.0) -> Transform:
    def t(imgs, hm, bbs):
        if np.random.uniform(0, 1) < 0.5:
            return np.flip(imgs, axis=1).copy(),\
                   np.flip(hm, axis=1).copy() if hm is not None else None,\
                   [bb.h_flip(img_width) for bb in bbs] if bbs else None
        return imgs, hm, bbs

    return t


def rand_v_flip(img_height: float = 1.0) -> Transform:
    def t(imgs, hm, bbs):
        if np.random.uniform(0, 1) < 0.5:
            return np.flip(imgs, axis=0).copy(),\
                   np.flip(hm, axis=0).copy() if hm is not None else None,\
                   [bb.v_flip(img_height) for bb in bbs] if bbs else None
        return imgs, hm, bbs

    return t


def rand_rot_90() -> Transform:
    def t(imgs, hm, bbs):
        if np.random.uniform(0, 1) < 0.5:
            return np.rot90(imgs, k=1).copy(),\
                   np.rot90(hm, k=1).copy() if hm is not None else None,\
                   [bb.rot90() for bb in bbs] if bbs else None
        return imgs, hm, bbs

    return t


def rand_noise(impact: int) -> Transform:
    def t(imgs, hm, bbs):
        noise = np.random.randint(-impact, impact + 1, imgs.shape)
        imgs = np.clip((imgs.astype(np.int32) + noise), 0, 255).astype(np.uint8)
        return imgs, hm, bbs

    return t


def reduce_colors(num_colors: int) -> Transform:
    def t(imgs, hm, bbs):
        colors = num_colors
        factor = 256 / colors
        imgs = ((imgs / factor).astype(np.uint8) * factor + (factor / 2)).astype(np.uint8)
        return imgs, hm, bbs
    return t


def thresh(threshold: int) -> Transform:
    def t(imgs, hm, bbs):
        imgs[imgs < threshold] = 0
        imgs[imgs >= threshold] = 255
        return imgs, hm, bbs
    return t


def rand_inverse() -> Transform:
    def t(imgs, hm, bbs):
        if np.random.uniform(0, 1) < 0.5:
            imgs = 255 - imgs
        return imgs, hm, bbs
    return t


def median_blur(kernel_size=3) -> Transform:
    def t(imgs, hm, bbs):
        imgs = cv2.medianBlur(imgs.astype(np.uint8), kernel_size)
        return imgs, hm, bbs

    return t


def rand_color_swap() -> Transform:
    def t(imgs, hm, bbs):
        colors = np.unique(imgs)
        swapped = np.copy(colors)
        np.random.shuffle(swapped)
        ret = np.zeros(imgs.shape, dtype=np.uint8)
        for x, y in zip(colors, swapped):
            ret = np.where(imgs == x, y, ret)

        return ret, hm, bbs
    return t


def rand_occlusion(occlusion_files: List[File], occlusion_color: int) -> Transform:
    def t(imgs, hm, bbs):
        occlusion = load_occlusion(random.choice(occlusion_files).path)
        imgs = augment_imgs_with_constant_occlusion(imgs, occlusion, occlusion_color)
        return imgs, hm, bbs
    return t


def to_tensor() -> Transform:
    tens = transforms.ToTensor()

    def t(imgs, hm, bbs):
        imgs, hm = tens(imgs), tens(hm) if hm is not None else None
        return imgs, hm, bbs

    return t


def resolve_transform(trans_list: List[Transform]) -> Transform:
    trans_list = list(filter(lambda x: type(x) != to_tensor, trans_list))
    trans_list.append(to_tensor())
    return chain(trans_list)


def rand_zoom_out(min_scale: float, max_scale: float) -> Transform:
    """Shrink the frame by a random factor and pad back to size with replicated borders.

    KTH persons are 71-144px tall at 224px (median 109); NFO persons are 45-64px (median 54)
    - the two scale distributions do not overlap at all, and that alone costs ~0.46 precision
    (see docs/training_failure_hypotheses.md). This puts KTH persons into NFO's range during
    training. min_scale=0.4 maps the KTH median to ~44px, max_scale=1.0 keeps the original.
    """
    def t(imgs, hm, bbs):
        s = np.random.uniform(min_scale, max_scale)
        h, w = imgs.shape[0], imgs.shape[1]
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        top, left = (h - nh) // 2, (w - nw) // 2

        def shrink(a):
            # cv2.resize caps at 4 channels, and these stacks are seq_size deep - go per channel
            a = np.stack([cv2.resize(a[..., c], (nw, nh), interpolation=cv2.INTER_AREA)
                          for c in range(a.shape[2])], axis=2)
            return cv2.copyMakeBorder(a, top, h - nh - top, left, w - nw - left, cv2.BORDER_REPLICATE)

        imgs = shrink(imgs)
        hm = shrink(hm) if hm is not None else None
        # same affine on the boxes, using the realised (rounded) scale, then re-centre.
        # bbs are normalised to [0, 1] here (cf. rand_h_flip's img_width=1.0 default), so the
        # translation has to be normalised too - scale factors are already unitless.
        bbs = [bb.scale((nw / w, nh / h)).translate((left / w, top / h)) for bb in bbs] if bbs else bbs
        return imgs, hm, bbs

    return t
