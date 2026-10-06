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


def test_alignment_offsets_ols_and_per_frame_with_ols_fallback():
    hist = {k: (2.0 * k + (0.6 if k == 4 else 0.0), 5.0) for k in range(0, 13, 2) if k != 8}
    ks, t = [0, 4, 8, 12], 12
    ols = C.alignment_offsets(hist, ks, t, 'ols')
    pos = C.alignment_offsets(hist, ks, t, 'pos')
    assert all(abs(ols[k][1]) < 1e-9 for k in ks)                       # horizontal-only model
    assert abs(ols[0][0] - (ols[12][0] + 24)) < 0.5                     # ~2 px/frame
    assert abs(pos[4][0] - (hist[12][0] - hist[4][0])) < 1e-9           # own position used
    assert pos[8] == ols[8]                                             # no detection -> OLS


def test_detect_and_track_is_shared_with_t4():
    from benchmark.nfo_vos import run_t4 as R
    import json
    t = json.load(open('results/benchmark/pilot/trials.json'))[6]
    out = R.detect_and_track(t['frames'][:5], t['warmup'], t['box_224'][3] - t['box_224'][1])
    assert out['frames'].shape[0] == 5 and len(out['dets']) == 5 and 'tracks' in out
