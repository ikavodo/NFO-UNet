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


# ---- Task 2: key shift, aggregator, loss ----

def _inputs(n=2, k=3, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    chans = {'16': (256, h), '8': (64, 2 * h), '4': (32, 4 * h)}
    K = {s: torch.randn(n, k, c, w, w, generator=g) for s, (c, w) in chans.items()}
    Ft = {s: torch.randn(n, c, w, w, generator=g) for s, (c, w) in chans.items()}
    B = {s: torch.randn(n, c, w, w, generator=g) for s, (c, w) in chans.items()}
    valid = torch.ones(n, k, 1, h, h, dtype=torch.bool)
    age = torch.arange(k, 0, -1).repeat(n, 1)
    return K, valid, Ft, B, age


def test_shift_integer_equals_roll_interior():
    from benchmark.nfo_vos.adapter.feat_integral import shift_keys
    F = torch.randn(1, 1, 4, 8, 8)
    for mode in ('bilinear', 'roll'):
        Fs, v = shift_keys(F, torch.tensor([[32.]]), 16, mode)        # 2 cells right
        assert torch.allclose(Fs[..., 2:], F[..., :-2], atol=1e-5) and not v[..., :2].any() and v[..., 2:].all()


def test_shift_zero_is_identity():
    from benchmark.nfo_vos.adapter.feat_integral import shift_keys
    F = torch.randn(2, 3, 4, 8, 8)
    Fs, v = shift_keys(F, torch.zeros(2, 3), 16)
    assert torch.allclose(Fs, F, atol=1e-5) and v.all()


def test_weights_sum_to_one_and_init_uniform():
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    o = FeatIntegral(torch.zeros(1, 256, 1, 1))(*_inputs())
    assert torch.allclose(o['alpha16'].sum(1), torch.ones(1), atol=1e-5)
    assert torch.allclose(o['alpha16'], torch.full_like(o['alpha16'], 1 / 4), atol=1e-6)   # k = 3 valid + null
    assert o['a'].shape == (2, 32, 32) and o['C4'].shape == (2, 32, 32, 32)


def test_all_keys_invalid_gives_background():                          # Review Focus 1 and 2
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    K, v, Ft, B, age = _inputs()
    for fixed in (False, True):
        o = FeatIntegral(torch.zeros(1, 256, 1, 1))(K, torch.zeros_like(v), Ft, B, age, fixed=fixed)
        assert torch.allclose(o['C16'], B['16'], atol=1e-5) and o['a'].abs().max() < 1e-5
        assert all(torch.isfinite(o[c]).all() for c in ('C16', 'C8', 'C4', 'a'))


def test_fixed_mode_equals_recency_mass_where_gate_open():
    """Keys ≈ B except a 2×2 block (6% of cells, below the q90 threshold's 10%) that differs strongly: there the gate is
    open in every key, so a ≈ Σ w = 1 and Ĉ16 ≈ Σ_k w_k K_k (recency weights, τ = 3.5)."""
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    K, v, Ft, B, age = _inputs(n=1)
    g = torch.Generator().manual_seed(1)
    K['16'] = B['16'][:, None] + 0.01 * torch.randn(K['16'].shape, generator=g)
    K['16'][..., 3:5, 3:5] = 5 * torch.randn(1, 3, 256, 2, 2, generator=g)
    o = FeatIntegral(torch.zeros(1, 256, 1, 1))(K, v, Ft, B, age, fixed=True)
    w = torch.exp(-age[0].float() / 3.5); w = w / w.sum()
    ref = (w[:, None, None, None] * K['16'][0]).sum(0)
    assert o['alpha16'][0, :3, 3:5, 3:5].sum(0).min() > 0.99
    assert torch.allclose(o['C16'][0, :, 3:5, 3:5], ref[:, 3:5, 3:5], atol=5e-2)
    assert o['alpha16'][0, :3, 0, 0].sum() < 0.1                     # unchanged cell: mostly null (empty scene)


@GPU
def test_fp16_autocast_finite():                                       # Review Focus 4
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    K, v, Ft, B, age = _inputs()
    cu = lambda d: {s: x.cuda() for s, x in d.items()}
    m = FeatIntegral(torch.zeros(1, 256, 1, 1)).cuda()
    with torch.autocast('cuda', dtype=torch.float16):
        for fixed in (False, True):
            o = m(cu(K), v.cuda(), cu(Ft), cu(B), age.cuda(), fixed=fixed)
            assert all(torch.isfinite(o[c]).all() for c in ('C16', 'C8', 'C4', 'a', 'alpha16'))


def test_feat_loss_zero_on_empty_omega():                              # Review Focus 3 (Task 5's test, lives here)
    from benchmark.nfo_vos.adapter.feat_integral import feat_loss
    C = {s: torch.randn(2, c, h, h, requires_grad=True) for s, c, h in (('16', 256, 8), ('8', 64, 16), ('4', 32, 32))}
    Om = {s: torch.zeros(2, 1, x.shape[-2], x.shape[-1], dtype=torch.bool) for s, x in C.items()}
    l = feat_loss(C, {s: torch.randn_like(x) for s, x in C.items()}, Om, torch.zeros(1, 256, 1, 1))
    assert l.item() == 0 and l.requires_grad


# ---- Task 3: build(feat=True) ----

@GPU
def test_build_feat_shapes_and_masks():
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    from benchmark.nfo_vos.adapter.fusion_data import a256, build, clip_raw, specs
    raw = clip_raw(specs('val')[0], np.random.default_rng(0)); raw['modes'][:] = 0; raw['modes'][-1] = 2
    f, meta = build(image_predictor(), raw, all_targets=True, feat=True)
    n, nh = len(meta['ts']), raw['occ']['nh']
    assert f['K16'].shape == (n, nh, 256, 64, 64) and f['K4'].shape == (n, nh, 32, 256, 256)
    assert 'C16' not in f and f['T8'].shape == (n, 64, 128, 128)
    assert not f['Kvalid16'][-1].any() and f['Kvalid16'][0].any()     # mode 2 → all keys invalid (Review Focus 2)
    S16 = torch.as_tensor(np.stack([a256(raw['S'][t]) for t in meta['ts']]))[:, None, 1::4, 1::4] > 0.5
    assert f['Om16'].shape == (n, 1, 64, 64) and f['Om4'].shape == (n, 1, 256, 256)
    assert (f['Om16'].cpu() & ~S16).float().mean() < 0.01            # Ω ⊆ person (up to area-resampling at the rim)
    assert f['age'][0].tolist() == list(range(nh, 0, -1))
    assert f['O4'].shape == (n, 256, 256) and 0 <= f['O4'].min() and f['O4'].max() <= 1 and f['O4'][-1].max() == 0
    assert f['phase'].shape == (n, nh) and f['phase'].abs().max() <= 0.5


# ---- Task 5b: phase input, supervised evidence mass ----

def test_phase_input_no_effect_at_init():
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    m = FeatIntegral(torch.zeros(1, 256, 1, 1)); x = _inputs()
    a = m(*x)['alpha16']; b = m(*x, phase=torch.full((2, 3), 0.4))['alpha16']
    assert torch.allclose(a, b)


def test_old_checkpoint_loads_with_zero_phase_weights():
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    m = FeatIntegral(torch.zeros(1, 256, 1, 1)); sd = m.state_dict()
    sd['h.0.weight'] = sd['h.0.weight'][:, :-2].clone()                # the pre-5b layout (4d input channels)
    m2 = FeatIntegral(torch.zeros(1, 256, 1, 1)); m2.load_compat(sd)
    assert (m2.h[0].weight[:, -2:] == 0).all() and torch.equal(m2.h[0].weight[:, :-2], sd['h.0.weight'])


def test_mass_loss_zero_grad_at_target():
    from benchmark.nfo_vos.adapter.feat_integral import mass_loss
    y = torch.rand(2, 8, 8) * 0.9 + 0.05
    a = y.clone().requires_grad_(True)
    mass_loss(a, y).backward()
    assert a.grad.abs().max() < 1e-5


def test_score_head_layout():
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    h = FeatIntegral(torch.zeros(1, 256, 1, 1)).h
    assert [type(m).__name__ for m in h] == ['Conv2d', 'GELU', 'Conv2d', 'GELU', 'Conv2d'] and h[2].kernel_size == (3, 3)


# ---- Task 5b rev: decoupled evidence-mass head ----

def test_mass_head_init_and_invalid():
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    m = FeatIntegral(torch.zeros(1, 256, 1, 1)); K, v, Ft, B, age = _inputs()
    o = m(K, v, Ft, B, age)
    assert torch.allclose(o['a'], torch.full_like(o['a'], torch.sigmoid(torch.log(torch.tensor(3.))).item()), atol=1e-5)
    assert m(K, torch.zeros_like(v), Ft, B, age)['a'].abs().max() < 1e-6


def test_mass_loss_does_not_touch_mixing_weights():
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral, mass_loss
    m = FeatIntegral(torch.zeros(1, 256, 1, 1)); o = m(*_inputs())
    mass_loss(o['a'], torch.zeros_like(o['a'])).backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for n, p in m.named_parameters() if n.startswith(('h.', 'null', 'P.', 'e_age')))
    assert m.hm[-1].weight.grad.abs().max() > 0


# ---- Task 6: Fusion training on the feature integral ----

@GPU
def test_train_step_feat_reaches_fusion_and_aggregator():
    from benchmark.nfo_vos.adapter import fusion_adapter as FA
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    from benchmark.nfo_vos.adapter.fusion_data import clip_raw, specs
    pred = FA.image_predictor()
    net = FA.Fusion().cuda(); net.init_fallback(pred.model)
    agg = FeatIntegral(pred.model.no_mem_embed).cuda()
    raw = clip_raw(specs('val')[0], np.random.default_rng(0))
    loss = FA.train_step(pred, net, agg, raw, np.random.default_rng(0), lam=1.0, max_targets=4)
    loss.backward()
    assert np.isfinite(loss.item())
    assert any(net.out[l].weight.grad.abs().max() > 0 for l in net.out)
    for name in ('h', 'hm'):
        gr = getattr(agg, name)[-1].weight.grad
        assert gr is not None and gr.abs().max() > 0, name


@GPU
def test_nfo_select_item_feat_inputs():
    from benchmark.nfo_vos.adapter import fusion_nfo
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor, with_feat
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    from benchmark.nfo_vos.adapter.fusion_data import encode
    pred = image_predictor()
    it = fusion_nfo.items('select', pred, feat=True)[0]
    f = encode(pred, [it], feat=True)
    assert f['K16'].shape == (1, 6, 256, 64, 64) and 'C16' not in f
    g, _ = with_feat(FeatIntegral(pred.model.no_mem_embed).cuda(), f)
    assert g['C4'].shape == (1, 32, 256, 256) and g['a'].shape == (1, 256, 256)


# ---- Task 7: one-pass inference ----

def test_inference_dx_equals_pixel_path_dx():
    """The feature path takes its shifts from fusion_data.hist_warps; the pixel path computes them inline (inference.py).
    Same Theil–Sen fit over [j−12, j], so the same dx."""
    from benchmark.nfo_vos.adapter.fusion_data import hist_warps
    from utils.stats_utils import theil_at
    rng = np.random.default_rng(0)
    kc = np.stack([np.cumsum(rng.normal(4, 1, 40)) + 100, rng.normal(300, 2, 40)], 1)
    for j in (6, 13, 39):
        cs = list(range(max(0, j - 12), j + 1))
        fx = theil_at(cs, [kc[l, 0] for l in cs], [j] + list(range(j - 6, j)))
        inline = [fx[j] - fx[l] for l in range(j - 6, j)]
        _, Ms = hist_warps(kc, j, 6)
        assert np.allclose([M[0, 2] for M in Ms], inline)


@GPU
def test_inference_feat_matches_pixel_before_history():               # Review Focus 5
    from benchmark.nfo_vos.adapter import fusion_adapter as FA
    from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
    from benchmark.nfo_vos.adapter.fusion_data import clip_raw, specs
    from benchmark.nfo_vos.adapter.inference import masks_from_arrays
    pred = FA.image_predictor()
    raw = clip_raw(specs('val')[0], np.random.default_rng(0))
    Xn, bg = list(raw['X'][:10]), raw['bg']
    cx, cy, h = raw['bxj'][0]; g = np.array([cx - h / 6, cy - h / 2, cx + h / 6, cy + h / 2], np.float32)
    net = FA.Fusion().cuda(); net.init_fallback(pred.model); net.eval()
    with torch.no_grad():                                              # as dump_arms / timing call it
        m_pix, _ = masks_from_arrays(pred, net, Xn, bg, g)
        net.agg = FeatIntegral(pred.model.no_mem_embed).cuda().eval()
        m_feat, _ = masks_from_arrays(pred, net, Xn, bg, g)
    assert (m_pix[:6] == m_feat[:6]).all() and m_feat.shape == m_pix.shape
