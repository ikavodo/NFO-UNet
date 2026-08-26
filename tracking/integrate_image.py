import numpy as np
import cv2


def crop_at(img, cx, cy, size):
    h, w = img.shape
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    x1, y1 = x0 + size, y0 + size
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    crop = img[y0:y1, x0:x1]
    return cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)


def anchor_for_frame(winner, t):
    if t in winner['history']:
        return winner['history'][t][0], winner['history'][t][1]
    frames_arr = np.array(winner['frames'])
    xs = np.array([winner['history'][f][0] for f in winner['frames']])
    A = np.vstack([frames_arr, np.ones(len(frames_arr))]).T
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    ys = np.array([winner['history'][f][1] for f in winner['frames']])
    return coef[0] * t + coef[1], ys.mean()


def restrict_to_nearby(mask, detections_at_frame, ax, ay, merge_radius):
    nearby = [d for d in detections_at_frame if np.hypot(d['x'] - ax, d['y'] - ay) <= merge_radius]
    keep = np.zeros_like(mask)
    for d in nearby:
        x1, y1, x2, y2 = d['bbox']
        keep[y1:y2, x1:x2] = 1
    return mask * keep


def align_frames(frames: np.ndarray, winner: dict, crop_size: int = 220) -> np.ndarray:
    """Crop every frame to crop_size x crop_size, centered on the winning track's anchor
    point at that frame and shifted by -vx*dt so every frame samples the same real-world
    point as the center frame (this tracker's motion model is horizontal-only, matching
    its constant-velocity Kalman assumption)."""
    T = frames.shape[0]
    center_t = T // 2
    aligned = np.zeros((T, crop_size, crop_size), dtype=frames.dtype)
    for t in range(T):
        ax, ay = anchor_for_frame(winner, t)
        dt = t - center_t
        aligned[t] = crop_at(frames[t], ax - winner['vx'] * dt, ay, crop_size)
    return aligned


def fuse(aligned: np.ndarray, method: str = 'median', gaussian_sigma: float = None) -> np.ndarray:
    """Fuse an aligned [T, H, W] stack into one [H, W] image.

    'median': robust to a minority of occluded frames at a given pixel (the occluder's
    intensity gets outvoted by the majority true value) - default, since occlusion
    robustness is the actual point, not just noise averaging.
    'mean': simple baseline: blends occluder and true value together, ghosting rather
    than reconstructing - kept for comparison, not the default.
    'gaussian': weights frames by a Gaussian in temporal distance from the center frame
    (gaussian_sigma, in frame-index units) - trades occlusion-robustness for pose
    fidelity (limb articulation across a gait cycle changes shape frame to frame).
    """
    if method == 'median':
        return np.median(aligned, axis=0).astype(aligned.dtype)
    if method == 'mean':
        return aligned.mean(axis=0).astype(aligned.dtype)
    if method == 'gaussian':
        if gaussian_sigma is None:
            raise ValueError("gaussian_sigma is required when method='gaussian'")
        T = aligned.shape[0]
        center_t = T // 2
        dt = np.arange(T) - center_t
        weights = np.exp(-(dt ** 2) / (2 * gaussian_sigma ** 2))
        weights /= weights.sum()
        return np.tensordot(weights, aligned.astype(np.float64), axes=(0, 0)).astype(aligned.dtype)
    raise ValueError(f"method must be 'median', 'mean', or 'gaussian', got {method!r}")


def integrate(frames: np.ndarray, winner: dict, detections=None, merge_radius: float = None,
             crop_size: int = 220, method: str = 'median', gaussian_sigma: float = None,
             mask_background: bool = False) -> np.ndarray:
    """End-to-end: align frames to the winning track's motion, optionally restrict each
    frame to only the person's own nearby detection(s) (mask_background=True), then fuse
    into one image.

    mask_background=False (default): fuses full-frame crops - recommended for feeding an
    off-the-shelf detector (YOLO etc.), since full-frame integration naturally blurs the
    background while keeping the aligned subject sharp, closer to natural image
    statistics than a cutout-on-blank-background. mask_background=True requires
    detections and merge_radius.
    """
    if mask_background:
        if detections is None or merge_radius is None:
            raise ValueError("mask_background=True requires detections and merge_radius")
        frames = frames.copy()
        for t in range(frames.shape[0]):
            ax, ay = anchor_for_frame(winner, t)
            frames[t] = restrict_to_nearby(frames[t], detections[t], ax, ay, merge_radius)

    aligned = align_frames(frames, winner, crop_size)
    return fuse(aligned, method=method, gaussian_sigma=gaussian_sigma)
