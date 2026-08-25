import numpy as np
import cv2


def foreground_mask(frames: np.ndarray, bg_frames: int = 5, var_threshold: float = 16.0) -> np.ndarray:
    """MOG2 background subtraction over a [T, H, W] uint8 grayscale stack.
    Returns a [T, H, W] uint8 binary mask (0/255)."""
    subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_frames, varThreshold=var_threshold,
                                                     detectShadows=False)
    masks = np.zeros_like(frames, dtype=np.uint8)
    for t in range(frames.shape[0]):
        masks[t] = subtractor.apply(frames[t])
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
