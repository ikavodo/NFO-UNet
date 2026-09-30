"""gt_to_native must invert gen_data's scale_and_pad_img_to_square exactly.

NFO GT boxes are normalised to the 224x224 PADDED square (800x600 native, padded 100 rows top and
bottom with BORDER_REPLICATE, then scaled). The old code mapped them with y * native_height,
which compresses every box to 75% height - the cause of every pseudo-mask missing the feet.
"""
import numpy as np
import pytest

from gen_data.gen_kth_data.kth_utils import scale_and_pad_img_to_square
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
from utils.bb_utils import BoundingBox


@pytest.mark.parametrize('native_box', [(120, 150, 180, 460), (600, 40, 690, 590), (0, 0, 800, 600)])
def test_inverts_the_forward_pad_and_scale(native_box):
    W, H = 800, 600
    x0, y0, x1, y1 = native_box
    _, bb224 = scale_and_pad_img_to_square(np.zeros((H, W), np.uint8),
                                           BoundingBox(x0, y0, x1 - x0, y1 - y0), 224)
    norm = BoundingBox(bb224.x / 224, bb224.y / 224, bb224.w / 224, bb224.h / 224)
    assert gt_to_native(norm, W, H) == pytest.approx(native_box, abs=1e-6)


def test_square_native_frame_needs_no_padding():
    assert gt_to_native(BoundingBox(0.25, 0.5, 0.5, 0.25), 400, 400) == pytest.approx((100, 200, 300, 300))
