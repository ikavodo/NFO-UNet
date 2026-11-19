"""Learned feature-space integral image (docs/superpowers/plans/2026-11-19-layer-sam-feature-integral.md)."""
import cv2
import numpy as np
import pytest
import torch

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


# ---- Task 1: noise-matched, occluder-free target frame Xt ----

def test_target_equals_frame_where_unoccluded():
    from benchmark.nfo_vos.synth.gpjatk_foliage import make_layered_clip
    from benchmark.nfo_vos.adapter import dataset as D
    seq, cam, *_ = D.clip_specs('val')[0]
    *_, L = make_layered_clip(seq, cam, 0.45, 3, 'perlin', None, n=6, save=False, layers=True)
    free = L['a'] < 1e-6                       # static occluder alpha (no sway passed)
    assert np.allclose(L['Xt'][:, free], L['X'][:, free], atol=1e-2)   # a < 1e-6, not 0: up to 255e-6 gray levels
    assert np.abs(L['Xt'] - L['X'])[:, ~free].max() > 0


def test_make_clip_target_shares_noise():
    from benchmark.nfo_vos.adapter import dataset as D
    seq, cam, _, seed = D.clip_specs('val')[0]
    d = D.make_clip(seq, cam, 0.45, seed)
    occ_free = cv2.dilate((d['a'] >= 1e-6).astype(np.uint8), np.ones((33, 33), np.uint8)) == 0   # >= 16 px from occluder
    assert d['Xt'].shape == d['X'].shape
    assert np.median(np.abs(d['Xt'].astype(float) - d['X'].astype(float))[:, occ_free]) <= 1.0


def test_clip_raw_carries_target():
    from benchmark.nfo_vos.adapter.fusion_data import clip_raw, specs
    r = clip_raw(specs('val')[0], np.random.default_rng(0))
    assert r['Xt'].shape == r['X'].shape and r['Xt'].dtype == np.uint8
