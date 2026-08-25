import numpy as np
import cv2


def foreground_mask(frames: np.ndarray, bg_frames: int = None, var_threshold: float = 16.0,
                    warmup_frames: np.ndarray = None, adaptive_learning_rate: bool = False) -> np.ndarray:
    """MOG2 background subtraction over a [T, H, W] uint8 grayscale stack.
    Returns a [T, H, W] uint8 binary mask (0/255).

    warmup_frames: optional [Tw, H, W] stack of person-absent frames from the same
    (static-camera) sequence, run through the subtractor first and discarded, so it can
    learn e.g. wind-blown foliage as a legitimate multi-modal background state before ever
    seeing the actual window. Without this, every window starts from zero history, so
    anything with visual variability - vegetation motion included - looks exactly as "new"
    to the model as a real person does. cv2's MOG2 save()/read() roundtrip does not
    actually restore usable state (verified empirically - a reloaded model misbehaves like
    a brand-new one), so warm-up must be re-run per window rather than cached/cloned.
    bg_frames defaults to len(warmup_frames) when given (unless explicitly overridden), so
    the adaptation rate (learning_rate ~= 1/history) matches how much warm-up was fed in.

    adaptive_learning_rate: when True, use an explicit per-frame learning rate of
    1/min(frames_seen_so_far, bg_frames) instead of leaving it at MOG2's default (a fixed
    ~1/bg_frames from frame 0). Needed for track_sequence.track_windows_in_sequence, which
    processes a whole sequence with one fixed bg_frames covering both early windows (few
    real frames elapsed) and late ones (many elapsed) - a fixed history tuned for "fully
    warmed" leaves early windows under-adapted (verified empirically: bg_frames=100 with
    only ~38 real frames elapsed left a window barely learned at all, far worse than when
    history was set to match the actual elapsed count). This schedule starts fast (learn
    strongly from what little data exists) and settles to the configured rate once enough
    real frames have accumulated. Off by default - track_window's existing, already-
    validated single-window behavior is unaffected.
    """
    if bg_frames is None:
        bg_frames = len(warmup_frames) if warmup_frames is not None else 5
    subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_frames, varThreshold=var_threshold,
                                                     detectShadows=False)
    frames_seen = 0
    if warmup_frames is not None:
        for wf in warmup_frames:
            lr = 1.0 / min(frames_seen + 1, bg_frames) if adaptive_learning_rate else -1
            subtractor.apply(wf, learningRate=lr)
            frames_seen += 1
    masks = np.zeros_like(frames, dtype=np.uint8)
    for t in range(frames.shape[0]):
        lr = 1.0 / min(frames_seen + 1, bg_frames) if adaptive_learning_rate else -1
        masks[t] = subtractor.apply(frames[t], learningRate=lr)
        frames_seen += 1
    return masks


def refine_mask(masks: np.ndarray, close_kernel_size: int = 6, open_kernel_size: int = 4) -> np.ndarray:
    """Morphological close then open, per frame, on a [T, H, W] binary mask stack."""
    close_k = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    open_k = np.ones((open_kernel_size, open_kernel_size), np.uint8)
    refined = np.zeros_like(masks, dtype=np.uint8)
    for t in range(masks.shape[0]):
        m = (masks[t] > 0).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        refined[t] = m
    return refined


def filter_by_shape(masks: np.ndarray, min_area: float = 50, min_solidity: float = 0.1) -> np.ndarray:
    """Keep only contours passing area/solidity thresholds, per frame, on a [T, H, W] mask stack."""
    clean = np.zeros_like(masks, dtype=np.uint8)
    for t in range(masks.shape[0]):
        frame = (masks[t] > 0).astype(np.uint8)
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(frame)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity >= min_solidity:
                cv2.drawContours(out, [cnt], -1, 1, -1)
        clean[t] = out * 255
    return clean
