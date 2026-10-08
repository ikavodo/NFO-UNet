import numpy as np

from benchmark.nfo_vos import diag_init_mask as D


def test_visible_mask_is_extent_and_foreground():
    bg = np.full((40, 40), 100, np.uint8)
    fr = bg.copy(); fr[10:30, 10:20] = 160            # person, different from background
    fr[10:30, 14:16] = 100                            # a branch hides a stripe: matches background
    ext = np.zeros((40, 40), bool); ext[5:35, 8:22] = True
    v = D.visible_in_extent(fr, bg, ext, mode='fixed')
    assert v[20, 12] and not v[20, 15] and not v[2, 2]   # person yes, occluded stripe no, outside no
    vo = D.visible_in_extent(fr, bg, ext, mode='otsu')
    assert vo[20, 12] and not vo[20, 15]
