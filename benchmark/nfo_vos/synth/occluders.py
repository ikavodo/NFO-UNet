"""occ_perlin / occ_branch occluder generators (with their private helpers), vendored verbatim from the author's
master_thesis repo, src/occluders.py:38-386 @686c773, so the synthetic training data needs no sibling checkout.
Used by synth/gpjatk_foliage.py.
"""
import warnings
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _interp_to_cv2(interpolation: Literal['cubic', 'lanczos', 'nearest']) -> int:
    """Maps an interpolation name string to the corresponding OpenCV resize flag."""
    match interpolation:
        case 'cubic':   return cv2.INTER_CUBIC
        case 'lanczos': return cv2.INTER_LANCZOS4
        case 'nearest': return cv2.INTER_NEAREST
        case _: raise ValueError(f"interpolation must be 'cubic', 'lanczos', or 'nearest', got '{interpolation}'")


def _build_light_canvas(H: int, W: int,
                        light_movement_x: int, light_movement_y: int,
                        grayscale_intercept: float, grayscale_slope: float,
                        seed: int):
    """
    Builds a smooth random light-intensity canvas on a padded grid.

    The canvas is larger than (H, W) by the maximum sinusoidal displacement on each side,
    so any per-frame crop always stays within bounds.

    Returns:
        light_canvas:  float32 numpy array [LIGHT_H, LIGHT_W] with values in
                       [grayscale_intercept, grayscale_intercept + grayscale_slope].
        light_pad_x:   horizontal padding used (pixels).
        light_pad_y:   vertical padding used (pixels).
    """
    light_pad_x = abs(light_movement_x)
    light_pad_y = abs(light_movement_y)
    LIGHT_H = H + 2 * light_pad_y
    LIGHT_W = W + 2 * light_pad_x

    np.random.seed(seed)
    light_size = (max(1, LIGHT_H // 20), max(1, LIGHT_W // 20))
    low_res_light = np.random.rand(*light_size).astype(np.float32)
    light_canvas = cv2.resize(low_res_light, (LIGHT_W, LIGHT_H), interpolation=cv2.INTER_CUBIC)
    light_canvas = grayscale_intercept + grayscale_slope * light_canvas

    return light_canvas, light_pad_x, light_pad_y


def _crop_canvas(canvas: np.ndarray,
                 pad_x: int, pad_y: int,
                 H: int, W: int,
                 t: int,
                 movement_x: int, movement_y: int,
                 freq: float) -> np.ndarray:
    """
    Crops a frame-sized [H, W] window from a padded canvas using a sinusoidal offset.

    The horizontal offset is movement_x * sin(t / freq) and the vertical offset is
    movement_y * cos(t / freq), both rounded to the nearest pixel.
    """
    dx = round(movement_x * np.sin(t / freq))
    dy = round(movement_y * np.cos(t / freq))
    x0 = pad_x + dx
    y0 = pad_y + dy
    return canvas[y0:y0 + H, x0:x0 + W]


def _frame_to_numpy(frame_tensor: torch.Tensor, C: int) -> np.ndarray:
    """
    Converts a [C, H, W] torch frame tensor to a uint8 numpy array.

    Returns:
        [H, W]    uint8 array when C == 1 (grayscale).
        [H, W, 3] uint8 array when C == 3 (RGB).
    """
    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()   # [H, W, C]
    if frame_np.max() <= 1.0:
        frame_np = (frame_np * 255).astype(np.uint8)
    else:
        frame_np = frame_np.astype(np.uint8)
    if C == 1:
        frame_np = frame_np.squeeze(-1)                       # [H, W]
    return frame_np


def _apply_occluder(frame_np: np.ndarray,
                    noise_intensity: np.ndarray,
                    mask: np.ndarray,
                    C: int,
                    overlay: bool) -> torch.Tensor:
    """
    Composites a grayscale noise layer onto a frame and returns a [C, H, W] float tensor.

    Args:
        frame_np:        uint8 numpy array — [H, W] for C=1, [H, W, 3] for C=3.
        noise_intensity: uint8 [H, W] grayscale values (zero outside the mask).
        mask:            bool  [H, W] — True where the occluder is active.
        C:               number of channels (1 or 3).
        overlay:         True  → occluder pixels show grayscale noise intensity.
                         False → occluder pixels are set to black.

    Returns:
        torch.Tensor of shape [C, H, W] with dtype float32 in [0.0, 1.0].
    """
    if C == 1:
        if overlay:
            processed_frame = np.where(mask, noise_intensity, frame_np)
        else:
            processed_frame = np.where(mask, 0, frame_np)
        processed_frame = processed_frame.astype(np.uint8)
        return torch.from_numpy(processed_frame).float().unsqueeze(0) / 255.0

    else:
        mask_3d = mask[:, :, np.newaxis]                          # [H, W, 1] for broadcasting
        if overlay:
            noise_rgb = np.stack([noise_intensity] * 3, axis=-1)  # [H, W, 3] grayscale branches
            processed_frame = np.where(mask_3d, noise_rgb, frame_np)
        else:
            processed_frame = np.where(mask_3d, 0, frame_np)
        processed_frame = processed_frame.astype(np.uint8)
        return torch.from_numpy(processed_frame).float().permute(2, 0, 1) / 255.0


def _render_noise_loop(tensor_video: torch.Tensor,
                       occluder_canvas: np.ndarray,
                       pad_x: int, pad_y: int,
                       light_canvas: np.ndarray,
                       light_pad_x: int, light_pad_y: int,
                       movement_x: int, movement_y: int,
                       light_movement_x: int, light_movement_y: int,
                       C: int, overlay: bool,
                       return_mask: bool = False):
    """
    Shared per-frame loop for all canvas-based noise functions.

    Crops both the occluder and the light map sinusoidally each frame, combines
    them into a grayscale intensity layer, and composites it onto the video frame.

    Args:
        tensor_video:    [T, C, H, W] input video tensor.
        occluder_canvas: [BIG_H, BIG_W] occluder canvas — bool (True = occluded) or
                         float32 [0, 1] for variable per-pixel intensity (e.g. branches).
                         Bool values are treated as 1.0, so existing callers are unaffected.
        pad_x/pad_y:     horizontal/vertical padding used when building occluder_canvas.
        light_canvas:    [LIGHT_H, LIGHT_W] float light-intensity canvas.
        light_pad_x/y:   padding used when building light_canvas.
        movement_x/y:    sinusoidal amplitude for the occluder crop (freq t/freq).
        light_movement_x/y: sinusoidal amplitude for the light crop (freq t/7).
        C:               number of channels (1 or 3).
        overlay:         True → occluded pixels show intensity; False → black.
        freq:            sinusoidal frequency divisor for the occluder crop (default 5.0).
        return_mask:     If True, also return [T, H, W] bool tensor of occluded pixels.

    Returns:
        torch.Tensor [T, C, H, W] float32, or tuple (video, masks [T,H,W] bool) if return_mask.
    """
    T, _, H, W = tensor_video.shape
    noisy_frames = []
    masks = [] if return_mask else None

    for t in range(T):
        # Occluder crop: sinusoidal at frequency t/freq
        occluder_t = _crop_canvas(occluder_canvas, pad_x, pad_y, H, W, t,
                                  movement_x, movement_y, 5.0)

        # Light map crop: sinusoidal at frequency t/7 (different speed → natural phase drift)
        light_t = _crop_canvas(light_canvas, light_pad_x, light_pad_y, H, W, t,
                               light_movement_x, light_movement_y, 7.0)    # float [H, W]

        # Combine: pixel intensity = occluder_factor * light * 255
        # For bool canvases True → 1.0, so behaviour is identical to before.
        mask = occluder_t.astype(bool)
        noise_intensity = np.zeros((H, W), dtype=np.uint8)
        noise_intensity[mask] = np.clip(
            occluder_t[mask].astype(np.float32) * light_t[mask] * 255.0, 0, 255
        ).astype(np.uint8)

        if return_mask:
            masks.append(torch.from_numpy(mask))

        frame_np = _frame_to_numpy(tensor_video[t], C)
        noisy_frames.append(_apply_occluder(frame_np, noise_intensity, mask, C, overlay))

    video = torch.stack(noisy_frames, dim=0)
    if return_mask:
        return video, torch.stack(masks, dim=0)   # [T,H,W] bool
    return video


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗ #
# ║                                                           PUBLIC NOISE FUNCTIONS                                                                                ║ #
# ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝ #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


####################
### Branch Noise ###
####################

def occ_branch(tensor_video: torch.Tensor,
                     num_branches: int = 210,
                     thickness: int = 5,
                     transparency: float = 1,
                     movement_x: int = 6,
                     grayscale_intercept = 0.4,
                     grayscale_slope = 0.6,
                     light_movement_x: int = 5,
                     light_movement_y: int = 3,
                     seed: int = 42,
                     overlay: bool = True,
                     return_mask: bool = False):
    """
    Adds branch-like noise (occluders) to a video tensor.
    Branches are always rendered in grayscale; vertical span scales with image height.
    Branches shift sinusoidally at freq t/3; light map drifts independently at t/7.

    Args:
        tensor_video:        [T, C, H, W] input video (C=1 or C=3).
        num_branches:        Number of branches drawn per frame.
        thickness:           Branch thickness in pixels (min 2).
        movement_x:          Tip sway amplitude (pixels, freq t/3); base stays fixed → tree-sway.
        grayscale_intercept: Light intensity floor (a in a + b·x).
        grayscale_slope:     Light intensity range (b in a + b·x).
        light_movement_x/y:  Light map scroll amplitude (pixels); see _build_light_canvas.
        seed:                Random seed.
        overlay:             True → branches show grayscale; False → black.

    Returns:
        [T, C, H, W] float32 tensor in [0.0, 1.0].
    """

    # Pre checks
    if thickness < 2:
        thickness = 2
        warnings.warn("🚨We recommend the thickness to be between 2 and 10 for optimal results 🚨\n")
    if transparency < 0 or transparency > 1:
        raise ValueError(f"Transparency argument should be between 0 and 1, is {transparency} instead")

    T, C, H, W = tensor_video.shape

    # Build the light canvas once (independent motion at t/7)
    light_canvas, light_pad_x, light_pad_y = _build_light_canvas(
        H, W, light_movement_x, light_movement_y, grayscale_intercept, grayscale_slope, seed)

    noisy_frames = []
    masks = [] if return_mask else None
    for t in range(T):
        # Sway offset: only the tip (pt2) shifts; the base (pt1) stays fixed → tree-sway effect
        current_movement = round(movement_x * np.sin(t / 3))

        np.random.seed(seed)
        branch_layer = np.zeros((H, W), dtype=np.float32)
        for _ in range(num_branches):
            pt1 = (np.random.randint(0, W), np.random.randint(H - 50, H))
            pt2 = (pt1[0] + np.random.randint(-50, 50) - current_movement,
                   pt1[1] + np.random.randint(-int(H * 1.3), H // 10))
            thick = np.random.randint(thickness - 1, thickness + 1)
            branch_intensity = float(np.random.uniform(100, 255) * transparency)
            cv2.line(branch_layer, pt1, pt2, branch_intensity, thick)

        # Crop light map for this frame
        light_t = _crop_canvas(light_canvas, light_pad_x, light_pad_y, H, W, t,
                                light_movement_x, light_movement_y, 7.0)
        branch_layer = np.clip(branch_layer * light_t, 0, 255).astype(np.uint8)
        mask = branch_layer > 0

        if return_mask:
            masks.append(torch.from_numpy(mask))

        frame_np = _frame_to_numpy(tensor_video[t], C)
        noisy_frames.append(_apply_occluder(frame_np, branch_layer, mask, C, overlay))

    video = torch.stack(noisy_frames, dim=0)
    if return_mask:
        return video, torch.stack(masks, dim=0)   # [T,H,W] bool
    return video


####################
### Perlin Noise ###
####################

def occ_perlin(tensor_video: torch.Tensor,
                      density: float = 0.5,
                      noise_h: int = 10,
                      noise_w: int = 10,
                      interpolation: Literal['cubic', 'lanczos', 'nearest'] = 'cubic',
                      movement_x: int = 8,
                      movement_y: int = 4,
                      grayscale_intercept: float = 0.4,
                      grayscale_slope: float = 0.6,
                      light_movement_x: int = 5,
                      light_movement_y: int = 3,
                      seed: int = 42,
                      overlay: bool = True,
                      return_mask: bool = False):
    """
    Adds Perlin-like foliage noise (occluders) to a video tensor.
    A low-res random grid is upscaled and thresholded on a padded canvas; each frame
    crops a sliding window for seamless motion (occluder t/5, light t/7).

    Args:
        tensor_video:        [T, C, H, W] input video (C=1 or C=3).
        density:             Fraction of pixels covered (0 = none, 1 = full).
        noise_h/noise_w:     Downscale factors for the low-res grid (larger = coarser clumps).
        interpolation:       Upscale method: 'cubic', 'lanczos', or 'nearest'.
        movement_x/y:        Occluder scroll amplitude (pixels, freq t/5).
        light_movement_x/y:  Light map scroll amplitude (pixels); see _build_light_canvas.
        grayscale_intercept: Light intensity floor (a in a + b·x).
        grayscale_slope:     Light intensity range (b in a + b·x).
        seed:                Random seed.
        overlay:             True → foliage shows grayscale; False → black.

    Returns:
        [T, C, H, W] float32 tensor in [0.0, 1.0].
    """
    if density < 0 or density > 1:
        raise ValueError(f"Density argument should be between 0 and 1, is {density} instead")

    interp_flag = _interp_to_cv2(interpolation)

    T, C, H, W = tensor_video.shape

    np.random.seed(seed)

    # --- Build a large noise canvas padded by the max displacement on every side ---
    # dx = movement_x * sin(t/5) ∈ [-movement_x, +movement_x], same for dy.
    # A canvas of (H + 2*pad_y, W + 2*pad_x) covers every possible crop position,
    # so per-frame motion is just an array slice — no warp, no seam.
    pad_x = abs(movement_x)
    pad_y = abs(movement_y)
    BIG_H = H + 2 * pad_y
    BIG_W = W + 2 * pad_x

    # Generate Perlin-like noise on the bigger canvas
    noise_size = (max(1, BIG_H // noise_h), max(1, BIG_W // noise_w))
    low_res_noise = np.random.rand(*noise_size).astype(np.float32)
    foliage_big = cv2.resize(low_res_noise, (BIG_W, BIG_H), interpolation=interp_flag)

    # Threshold to create binary occluder mask
    foliage_canvas = foliage_big > (1 - density)   # bool [BIG_H, BIG_W]

    light_canvas, light_pad_x, light_pad_y = _build_light_canvas(
        H, W, light_movement_x, light_movement_y, grayscale_intercept, grayscale_slope, seed)

    return _render_noise_loop(tensor_video, foliage_canvas, pad_x, pad_y,
                              light_canvas, light_pad_x, light_pad_y,
                              movement_x, movement_y, light_movement_x, light_movement_y,
                              C, overlay, return_mask=return_mask)


#######################
### Bernoulli Noise ###
#######################

