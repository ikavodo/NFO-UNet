from typing import Tuple

import numpy as np
import cv2

from utils.fs_utils import ensure_dir


_MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)


def __shrink_region(region: np.ndarray, shrink_prob: float):
    # ponytail: vectorized reimplementation of the original per-pixel loop (same semantics:
    # each True pixel independently triggers with shrink_prob, spreading True to its 3x3
    # neighborhood) using cv2.dilate instead of scalar Python loops - ~1000x faster at 224x224
    seeds = region & (np.random.random(region.shape) < shrink_prob)
    spread = cv2.dilate(seeds.astype(np.uint8), _MORPH_KERNEL).astype(bool)
    output = np.copy(region)
    output[spread] = True
    return output


def __grow_region(region: np.ndarray, grow_prob: float):
    seeds = (~region) & (np.random.random(region.shape) < grow_prob)
    spread = cv2.dilate(seeds.astype(np.uint8), _MORPH_KERNEL).astype(bool)
    output = np.copy(region)
    output[spread] = False
    return output


def generate_occlusion_morph(shape: Tuple[int, int], init_occ_prob: float = 0.05, iterations: int = 1,
                             occ_grow_prob: float = 1, occ_shrink_prob: float = 1) -> np.ndarray:
    occlusion = np.random.random(shape) > init_occ_prob

    for i in range(iterations):
        occlusion = __grow_region(occlusion, occ_grow_prob)
    for i in range(iterations):
        occlusion = __shrink_region(occlusion, occ_shrink_prob)

    return occlusion


def generate_occlusion_branch(shape: Tuple[int, int], n_frames: int, num_branches: int = 210,
                              thickness: int = 5, movement_x: int = 6, seed: int = 42) -> np.ndarray:
    """[T, H, W] bool occlusion mask stack of branch-like line occluders whose tips sway
    sinusoidally (freq t/3) while their bases stay fixed - a structurally accurate model of
    wind-blown foliage, and temporally coherent (the same branches move) rather than one
    static occlusion pattern repeated across every frame.

    Ported from master_thesis/src/occluders.py:occ_branch - keeps only the geometric/motion
    core (branch line segments + sway); drops that function's torch [T,C,H,W] video-tensor
    plumbing and its separately-drifting RGB light/shading canvas, neither of which applies
    here (this project's occlusion masks are plain booleans - color/shading is applied
    separately via augment_imgs_with_constant_occlusion/augment_imgs_with_noisy_occlusion).

    Re-seeds every frame like the original: branch base positions stay identical across
    frames (real branches don't relocate their trunk), only the sway offset changes what
    moves.
    """
    H, W = shape
    masks = np.zeros((n_frames, H, W), dtype=bool)
    for t in range(n_frames):
        current_movement = round(movement_x * np.sin(t / 3))
        np.random.seed(seed)
        branch_layer = np.zeros((H, W), dtype=np.uint8)
        for _ in range(num_branches):
            pt1 = (np.random.randint(0, W), np.random.randint(max(H - 50, 0), H))
            pt2 = (pt1[0] + np.random.randint(-50, 50) - current_movement,
                   pt1[1] + np.random.randint(-int(H * 1.3), H // 10))
            thick = np.random.randint(max(thickness - 1, 1), thickness + 2)
            cv2.line(branch_layer, pt1, pt2, 1, thick)
        masks[t] = branch_layer > 0
    return masks


def save_occlusion(file_path: str, occlusion: np.ndarray):
    ensure_dir(file_path)
    cv2.imwrite(file_path, occlusion * 255)


def load_occlusion(file_path: str):
    occ = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return occ < 127


def augment_imgs_with_noisy_occlusion(imgs: np.ndarray, occlusion: np.ndarray, color: np.ndarray) -> np.ndarray:
    num_channels = imgs.shape[2] if len(imgs.shape) >= 3 else 1
    color = np.reshape(np.stack([color for _ in range(num_channels)], axis=2).astype(np.uint8), imgs.shape)
    occlusion = np.reshape(np.stack([occlusion for _ in range(num_channels)], axis=2), imgs.shape)

    assert imgs.shape == occlusion.shape == color.shape
    return np.where(occlusion, color, imgs)


def augment_imgs_with_constant_occlusion(imgs: np.ndarray, occlusion: np.ndarray,
                                         occlusion_color: int) -> np.ndarray:
    num_channels = imgs.shape[2] if len(imgs.shape) >= 3 else 1
    occlusion_color = np.full(occlusion.shape, occlusion_color, dtype=np.uint8)
    occlusion_color = np.reshape(np.stack([occlusion_color for _ in range(num_channels)], axis=2), imgs.shape)
    occlusion = np.reshape(np.stack([occlusion for _ in range(num_channels)], axis=2), imgs.shape)

    assert imgs.shape == occlusion.shape == occlusion_color.shape
    return np.where(occlusion, occlusion_color, imgs)


def calculate_density_and_connectedness(occlusion: np.ndarray) -> Tuple[float, float]:
    (width, height) = occlusion.shape
    area = np.count_nonzero(occlusion)
    density = area / (width * height)

    circumference = calc_circumference(occlusion)
    connectedness = area / circumference
    connectedness *= (2 / np.sqrt(area / np.pi))

    return density, connectedness


def calc_circumference(occlusion: np.ndarray):
    counter = 0
    height, width = occlusion.shape
    for h in range(height):
        for w in range(width):
            if occlusion[h, w]:
                if h-1 < 0 or not occlusion[h-1, w] or h + 1 >= height:
                    counter += 1
                if w-1 < 0 or not occlusion[h, w-1] or w + 1 >= width:
                    counter += 1
            else:
                if h-1 >= 0 and occlusion[h-1, w]:
                    counter += 1
                if w-1 >= 0 and occlusion[h, w-1]:
                    counter += 1
    return counter