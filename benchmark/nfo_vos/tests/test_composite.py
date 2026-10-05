import numpy as np

from benchmark.nfo_vos import composite as C


def test_match_moments_maps_masked_src_stats_onto_whole_ref_stats():
    rng = np.random.default_rng(0)
    src = rng.normal(10, 2, (20, 20)); ref = rng.normal(100, 7, (20, 20))
    m = np.zeros((20, 20), bool); m[5:15, 5:15] = True
    out = C.match_moments(src, ref, mask=m)
    assert np.isclose(out[m].mean(), ref.mean()) and np.isclose(out[m].std(), ref.std())


def test_horizon_is_causal_strided_and_skips_frames_without_gt():
    has_gt = lambda k: k >= 100
    assert C.horizon_frames(112, 7, 2, has_gt) == [100, 102, 104, 106, 108, 110, 112]
    assert C.horizon_frames(103, 7, 2, has_gt) == [101, 103]


def test_shift_moves_content_by_the_given_offset():
    img = np.zeros((30, 40), np.uint8); img[10, 12] = 255
    out = C.shift(img, 5, -3)
    assert out[7, 17] == 255 and out.sum() == 255
