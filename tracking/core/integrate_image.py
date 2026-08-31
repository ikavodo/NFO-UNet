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


def restrict_to_nearby(img, frame_mask, detections_at_frame, ax, ay, merge_radius):
    """Zero out every pixel of img not belonging to a connected component of frame_mask
    near (ax, ay). Uses the actual per-pixel blob shape (via connectedComponents on
    frame_mask), not a bbox rectangle - a bbox-based version was tried first and produced
    a visible box-collage artifact once differently-sized/positioned per-frame bboxes were
    aligned and overlaid (hard rectangular seams, not a clean silhouette)."""
    binary = (frame_mask > 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.zeros_like(frame_mask, dtype=np.uint8)
    for d in detections_at_frame:
        if np.hypot(d['x'] - ax, d['y'] - ay) <= merge_radius:
            cx, cy = int(round(d['x'])), int(round(d['y']))
            if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
                lbl = labels[cy, cx]
                if lbl != 0:
                    keep[labels == lbl] = 1
    return img * keep


def align_frames(frames: np.ndarray, winner: dict, crop_size: int = 220) -> np.ndarray:
    """Crop every frame to crop_size x crop_size, FOLLOWING the winning track's fitted
    motion, so the person lands at the same place in every crop and static occluders sweep
    across it. That is the property fuse() needs: with the person stationary in aligned
    coordinates, an occluder covering a given pixel in a minority of frames gets outvoted by
    the median, while a world-fixed window would keep the (static) occluder sharp and
    median-remove the moving person instead - exactly backwards.

    The crop path is the constant-velocity motion model pinned at the center frame's anchor,
    with a single mean y (this tracker's motion model is horizontal-only, matching its Kalman
    assumption). Per-frame anchors are deliberately NOT used: under fragmented occlusion the
    blob centroid jumps between head, torso and legs, which would shift the crop by a large
    fraction of a body height between frames.

    Fixed 2026-08-31. The previous version cropped at `anchor_for_frame(winner, t) -
    vx * (t - center_t)`, which double-corrected: anchor_for_frame already returns the
    person's position AT frame t, so subtracting vx*dt cancelled the alignment and left a
    window fixed to p(center_t) in world coordinates. Measured on a synthetic bar
    translating at a known 8 px per strided frame behind a static striped occluder, the
    "aligned" person drifted +8.6 px per frame (52 px across a 7-frame window) against
    +0.78 px for this version, and the median recovered 20% more of the person's own pixels.
    Any result computed with integrate() before this date was computed on a de-aligned
    stack. A world-fixed window is a legitimate thing to want - it reconstructs the
    BACKGROUND - but nothing in this repo asked for it.
    """
    T = frames.shape[0]
    center_t = T // 2
    ax, _ = anchor_for_frame(winner, center_t)
    ay = float(np.mean([winner['history'][f][1] for f in winner['frames']]))
    aligned = np.zeros((T, crop_size, crop_size), dtype=frames.dtype)
    for t in range(T):
        aligned[t] = crop_at(frames[t], ax + winner['vx'] * (t - center_t), ay, crop_size)
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
             frame_masks: np.ndarray = None, crop_size: int = 220, method: str = 'median',
             gaussian_sigma: float = None, mask_background: bool = False) -> np.ndarray:
    """End-to-end: align frames to the winning track's motion, optionally restrict each
    frame to only the person's own nearby detection(s) (mask_background=True), then fuse
    into one image.

    mask_background=False (default): fuses full-frame crops - recommended for feeding an
    off-the-shelf detector (YOLO etc.), since full-frame integration naturally blurs the
    background while keeping the aligned subject sharp, closer to natural image
    statistics than a cutout-on-blank-background. mask_background=True requires
    detections, merge_radius, and frame_masks (the [T,H,W] filter_by_shape output for
    these same frames - used to find each blob's actual per-pixel shape via connected
    components, not just its bbox).
    """
    if mask_background:
        if detections is None or merge_radius is None or frame_masks is None:
            raise ValueError("mask_background=True requires detections, merge_radius, and frame_masks")
        frames = frames.copy()
        for t in range(frames.shape[0]):
            ax, ay = anchor_for_frame(winner, t)
            frames[t] = restrict_to_nearby(frames[t], frame_masks[t], detections[t], ax, ay, merge_radius)

    aligned = align_frames(frames, winner, crop_size)
    return fuse(aligned, method=method, gaussian_sigma=gaussian_sigma)
