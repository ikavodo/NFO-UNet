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


def _sample_branch_specs(H: int, W: int, num_specs: int, thickness: int, y_start_band: int,
                         dx_range: int, dy_range: Tuple[int, int], seed: int):
    """Pre-samples num_specs candidate branch line segments (start point near the bottom
    edge, extending up/outward) - a superset that generate_occlusion_branch draws the
    first k of, so density search below only needs to vary k, not re-sample geometry."""
    rng = np.random.default_rng(seed)
    y_start_band = int(np.clip(y_start_band, 1, H))
    specs = []
    for _ in range(num_specs):
        x1 = int(rng.integers(0, W))
        y1 = int(rng.integers(H - y_start_band, H))
        x2 = int(np.clip(x1 + rng.integers(-dx_range, dx_range + 1), 0, W - 1))
        y2 = int(np.clip(y1 + rng.integers(dy_range[0], dy_range[1] + 1), 0, H - 1))
        thick = int(rng.integers(thickness, thickness + 2))
        specs.append(((x1, y1), (x2, y2), thick))
    return specs


def _render_branch_specs(H: int, W: int, specs, k: int) -> np.ndarray:
    canvas = np.zeros((H, W), dtype=np.uint8)
    for pt1, pt2, thick in specs[:k]:
        cv2.line(canvas, pt1, pt2, color=1, thickness=thick, lineType=cv2.LINE_AA)
    return canvas > 0


def _refine_mask_to_density(mask: np.ndarray, target: float, tol: float, max_steps: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = mask.astype(np.uint8)
    for _ in range(max_steps):
        d = m.mean()
        if abs(d - target) <= tol:
            break
        m_new = cv2.erode(m, kernel) if d > target else cv2.dilate(m, kernel)
        if np.array_equal(m, m_new):
            break
        m = m_new
    return m > 0


def generate_occlusion_branch(shape: Tuple[int, int], density: float = 0.3, tol: float = 0.02,
                              num_specs: int = 900, thickness: int = 1, y_start_band: int = 25,
                              dx_range: int = 25, dy_range: Tuple[int, int] = (-600, 30),
                              bbox: Tuple[int, int, int, int] = None, seed: int = 0) -> np.ndarray:
    """[H, W] bool occlusion mask of branch-like line segments, density-controlled to match
    `density` (fraction of the target region covered) within `tol` via binary search over how
    many pre-sampled segments to draw, plus a small erode/dilate refinement pass. Optionally
    restricted to a bounding box `(x1, y1, x2, y2)` - e.g. a person's own bbox - rather than
    filling the whole frame, so density is meaningful relative to the actual occlusion target
    instead of the frame as a whole.

    Ported from ~/PycharmProjects/MovingMNIST-OcclusionBench/occluders.py:branches_mask - a
    cleaner, density-controlled successor to master_thesis/src/occluders.py:occ_branch, which
    this function previously ported directly (indirect num_branches x thickness density
    control, no bbox restriction, global np.random.seed reuse). Drops that version's
    sinusoidal sway entirely: NFO's occluder geometry is treated as fixed per sequence in this
    project's own model (docs/nfo_pseudo_segmentation_approach.md's "Key constraint" - only
    the person moves), so animated sway was over-modeling motion this project doesn't assume
    exists. Single static mask now, matching generate_occlusion_morph's signature.
    """
    H, W = shape
    if bbox is None:
        x0, y0, x1b, y1b = 0, 0, W, H
    else:
        x0, y0, x1b, y1b = bbox
    bw, bh = x1b - x0, y1b - y0

    target = float(np.clip(density, 0.0, 1.0))
    full = np.zeros((H, W), dtype=bool)
    if target <= 0 or bw <= 0 or bh <= 0:
        return full

    specs = _sample_branch_specs(bh, bw, num_specs, thickness, min(y_start_band, bh),
                                 dx_range, dy_range, seed)

    def occ_for_k(k):
        m = _render_branch_specs(bh, bw, specs, k)
        return m.mean(), m

    occ_hi, m_hi = occ_for_k(num_specs)
    if occ_hi < target - tol:
        local = m_hi  # can't reach target even using every sampled branch
    else:
        lo, hi = 1, num_specs
        best = (num_specs, occ_hi, m_hi)
        while lo <= hi:
            mid = (lo + hi) // 2
            occ_mid, m_mid = occ_for_k(mid)
            best = (mid, occ_mid, m_mid)
            if abs(occ_mid - target) <= tol:
                break
            lo, hi = (mid + 1, hi) if occ_mid < target else (lo, mid - 1)
        _, occ_best, local = best
        if abs(occ_best - target) > tol:
            local = _refine_mask_to_density(local, target, tol)

    full[y0:y1b, x0:x1b] = local
    return full


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