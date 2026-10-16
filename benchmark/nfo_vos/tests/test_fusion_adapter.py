"""Tests for approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11). Plain asserts (the SAMURAI venv
has no pytest):      PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tests.test_fusion_adapter
"""
import numpy as np
import torch


def test_occlusion_sampler_ranges():
    from benchmark.nfo_vos.layers.fusion_data import sample_occlusion
    from benchmark.nfo_vos.synth import noise
    rng = np.random.default_rng(0)
    draws = [sample_occlusion(rng) for _ in range(300)]
    rho = np.array([d['rho'] for d in draws]); fl = np.array([d['flutter'] for d in draws])
    assert rho.min() == 0.0 and rho.max() <= 0.6 and ((rho == 0) | (rho >= 0.2)).all() and 0.05 < (rho == 0).mean() < 0.15
    assert 0.0 <= fl.min() and fl.max() <= 2 * noise.FLUTTER_AMP
    kinds = {str(d['clump']) for d in draws}
    assert kinds == {'12', '24', '(12, 24)'}, kinds


def test_clip_raw_and_build():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.layers.fusion_data import build, clip_raw, specs
    raw = clip_raw(specs('val')[0], np.random.default_rng(0))
    assert raw['X'].dtype == np.uint8 and raw['X'].ndim == 3 and {'nh', 'tau', 'z0'} <= set(raw['occ'])
    f, meta = build(R.image_predictor(), raw, max_targets=8, rng=np.random.default_rng(0))
    k = len(meta['ts'])
    assert 1 <= k <= 8 and min(meta['ts']) >= max(6, raw['occ']['nh'])
    assert f['X16'].shape == (k, 256, 64, 64) and f['C4'].shape == (k, 32, 256, 256) and f['B8'].shape == (k, 64, 128, 128)
    assert f['a'].shape == (k, 256, 256) and float(f['a'].min()) >= 0 and float(f['a'].max()) <= 1
    assert meta['trk'].shape == (len(raw['X']), 2) and np.isfinite(meta['trk']).all()


def test_rescale_clip_shrinks_person_and_keeps_labels():
    from benchmark.nfo_vos.layers import dataset as D
    from benchmark.nfo_vos.layers.fusion_data import rescale_clip, specs
    d = D.make_clip(*specs('val')[0])
    t = 20
    h0 = d['boxes'][t, 2]
    r = rescale_clip(d, 0.6)
    assert r['X'].shape == d['X'].shape and r['bg'].shape == d['bg'].shape and r['S'].dtype == bool
    assert abs(r['boxes'][t, 2] / h0 - 0.6) < 0.03
    ys, xs = np.nonzero(r['S'][t])
    assert abs((ys.max() - ys.min() + 1) - r['boxes'][t, 2]) <= 2           # boxes follow the rescaled silhouette
    assert (r['V'][t] <= r['S'][t]).all()
    assert rescale_clip(d, 1.0)['X'] is d['X']                                # s = 1: untouched


def _fake_feats(n, dev):
    g = torch.Generator(device='cpu').manual_seed(0)
    f = {}
    for key in ('X', 'C', 'B'):
        f[f'{key}16'] = torch.randn(n, 256, 64, 64, generator=g).to(dev)
        f[f'{key}8'] = torch.randn(n, 64, 128, 128, generator=g).to(dev)
        f[f'{key}4'] = torch.randn(n, 32, 256, 256, generator=g).to(dev)
    f['a'] = torch.rand(n, 256, 256, generator=g).to(dev)
    f['box'] = torch.tensor([[300., 200., 500., 800.]] * n, device=dev)
    return f


def test_residual_shapes_and_zero_init():
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion
    f = _fake_feats(2, 'cpu')
    net = Fusion(token=False)
    out = net(f)
    for l, c, s in (('16', 256, 64), ('8', 64, 128), ('4', 32, 256)):
        assert out[f'X{l}'].shape == (2, c, s, s)
        assert torch.equal(out[f'X{l}'], f[f'X{l}'])          # zero conv at init: features unchanged


def test_box_mode_equals_box_only_sam2_at_init():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion, decode
    sam = R.image_predictor().model.eval()
    f = _fake_feats(2, 'cuda')
    net = Fusion(token=False).cuda()
    with torch.no_grad():
        ref = decode(sam, f, token=None)
        got = decode(sam, net(f), token=None)
    assert torch.equal(ref, got), float((ref - got).abs().max())


def test_zero_convs_receive_gradient():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion, decode
    sam = R.image_predictor().model.eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    f = _fake_feats(2, 'cuda')
    net = Fusion(token=True).cuda()
    o = net(f, sam=sam)
    decode(sam, o, token=o['token']).float().mean().backward()
    assert all(net.out[l].weight.grad is not None and net.out[l].weight.grad.abs().sum() > 0 for l in ('16', '8', '4'))


def test_token_init_equals_point_at_evidence_centroid():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion, decode
    sam = R.image_predictor().model.eval()
    f = _fake_feats(2, 'cuda')
    f['a'] = torch.zeros(2, 256, 256, device='cuda'); f['a'][:, 100:140, 60:90] = 1     # evidence blob
    net = Fusion(token=True).cuda(); net.init_fallback(sam)
    with torch.no_grad():
        o = net(f, sam=sam)
        c = torch.tensor([[75 * 4.0, 120 * 4.0]] * 2, device='cuda')                       # blob centre, 1024 frame
        sp, _ = sam.sam_prompt_encoder(points=(c[:, None], torch.ones((2, 1), dtype=torch.int, device='cuda')),
                                       boxes=None, masks=None)
        w = 1 - np.exp(-1200 / 500)                                          # 40×30-cell blob, κ init 500
        want = w * sp + (1 - w) * net.fallback[None]
        assert torch.allclose(o['token'], want, atol=1e-4), float((o['token'] - want).abs().max())
        assert decode(sam, o, token=o['token']).shape == (2, 1, 256, 256)


def test_token_falls_back_without_evidence():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion
    sam = R.image_predictor().model.eval()
    f = _fake_feats(2, 'cuda'); f['a'] = torch.zeros(2, 256, 256, device='cuda')
    net = Fusion(token=True).cuda(); net.init_fallback(sam)
    with torch.no_grad():
        o = net(f, sam=sam)
    assert torch.allclose(o['token'], net.fallback[None].expand(2, -1, -1)), 'empty evidence must give the fallback token'
    assert torch.isfinite(o['token']).all()


def test_token_anchored_at_tracker_point_at_init():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion
    sam = R.image_predictor().model.eval()
    f = _fake_feats(2, 'cuda'); f['trk'] = torch.tensor([[300.0, 480.0], [700.0, 200.0]], device='cuda')
    f['box_trk'] = torch.cat([f['trk'] - 60, f['trk'] + 60], 1)
    net = Fusion(token=True).cuda(); net.init_fallback(sam)
    with torch.no_grad():
        o = net(f, sam=sam)
        pts = torch.cat([f['box_trk'].view(2, 2, 2), f['trk'][:, None]], 1)
        lab = torch.tensor([[2, 3, 1]], device='cuda').repeat(2, 1)
        sp, _ = sam.sam_prompt_encoder(points=(pts, lab), boxes=None, masks=None)
    assert torch.allclose(o['token'], sp, atol=1e-5), float((o['token'] - sp).abs().max())


if __name__ == '__main__':
    import sys
    names = [n for n in dir() if n.startswith('test_')]
    fails = 0
    for n in names:
        try:
            globals()[n](); print('PASS', n)
        except Exception as e:                               # noqa: BLE001 — report and continue
            fails += 1; print('FAIL', n, type(e).__name__, str(e)[:200])
    print(f'{len(names) - fails}/{len(names)} passed'); sys.exit(1 if fails else 0)
