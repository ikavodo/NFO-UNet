import numpy as np, pytest
from benchmark.nfo_vos.ape import OUT

pytestmark = pytest.mark.skipif(not (OUT / 'plants.csv').exists(), reason='needs results/ape (pool + plants)')


def test_planted_clip_labels_and_shapes():
    from benchmark.nfo_vos.ape.data import index, clip_raw, N
    C, P = index(); r = C.iloc[0]; g = P[(r.video, r.ape_id)]; p = g[g['rank'] == 0].iloc[0]
    d = clip_raw(r, p, np.random.default_rng(0))
    assert d['X'].shape == (N, r.H, r.W) and d['X'].dtype == np.uint8 and d['bg'].shape == (r.H, r.W)
    assert not (d['V'] & ~d['S']).any()                                            # visible is part of the ape
    hid = [1 - v.sum() / s.sum() for v, s in zip(d['V'], d['S']) if s.any()]
    assert abs(np.median(hid) - p.h) < 0.15                                         # per-frame alpha ~ the searched median
    assert not np.isnan(d['bxj']).any() and not np.isnan(d['track']).any() and d['occ']['shift_xy']


def test_clear_clip_is_unoccluded():
    from benchmark.nfo_vos.ape.data import index, clip_raw
    C, _ = index(); d = clip_raw(C.iloc[0], None, np.random.default_rng(0))
    assert (d['V'] == d['S']).all()
