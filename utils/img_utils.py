import numpy as np


def add_and_clip(bg: np.ndarray, img: np.ndarray, pos) -> np.ndarray:
    bg = bg.astype(int)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            try:
                bg[pos[0] + x, pos[1] + y] += img[x, y]
            except IndexError:
                pass
    return np.clip(bg, 0, 255).astype(np.uint8)


def overlay(bg: np.ndarray, fg: np.ndarray, pos) -> np.ndarray:
    ret = np.copy(bg)
    for x in range(fg.shape[0]):
        for y in range(fg.shape[1]):
            try:
                ret[pos[0] + x, pos[1] + y] = fg[x, y]
            except IndexError:
                pass
    return ret
