"""Is the pixel teacher enc(C) a learnable function of the one-pass inputs on NFO, and is the residual distillation gap a
data problem or a representation problem? (findings §10.x item 18)

The current distilled student (feat_refine: learned α mix + residual refiner on [Ĉ, B]) reaches centred cosine 0.78 to
enc(C)16 on NFO, equally on its training crossings (0.779) and on held-out ones (0.785): no generalisation gap, so it
underfits. Here a high-capacity student sees *everything* the one-pass route has, unmixed: the 6 shifted key features
K_k, their validity, sub-cell phase, F_t and B (2066 ch) -> 3 conv layers (512) -> 256. NFO only (seq1 crossings s4–s7),
3000 steps; held-out s0–s3. Readout: train and held-out teacher cos over training.
  train ≈ held-out ≈ 0.8  -> information lost per frame before mixing: no feature-space student closes it (distillation capped)
  train ≫ held-out        -> data-limited: more unlabelled real footage is the lever
  both ≫ 0.8              -> the current student is too weak: a bigger student is the lever

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_capacity [big|small c w ...]
Real-time variants ('small c w'): a shared 1×1 256→c per key / F_t / B (weights shared across the 8 maps), stacked
unmixed with validity and phase, then 3×3 w, 3×3 w, 1×1 →256. Each run also times the student on the full 64×64 grid
(batch 1, autocast, 100 reps after warm-up).
"""
import sys
import json
import os

import numpy as np
import torch
import torch.nn as nn

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.feat_integral import ccos
from benchmark.nfo_vos.adapter.fusion_adapter import AMP, image_predictor
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_distil import OUT, TEST, TRAIN, collect
from benchmark.nfo_vos.tools.feat_probe_nfo import NETS


def inputs(b):
    K = torch.stack([x['K'] for x in b]).cuda().float(); v = torch.stack([x['v'] for x in b]).cuda().float()
    n, k, c, h, w = K.shape
    ph = torch.stack([x['dx'] / 16 - torch.round(x['dx'] / 16) for x in b]).cuda().float()
    Ft = torch.stack([x['Ft'] for x in b]).cuda().float(); B = torch.stack([x['B'] for x in b]).cuda().float()
    return torch.cat([(K * v[:, :, None]).reshape(n, k * c, h, w), v, ph[..., None, None].expand(n, k, h, w), Ft, B], 1)


class Small(nn.Module):
    """Shared per-map 1×1 compression (keys, F_t, B), unmixed stack + validity + phase, two narrow 3×3 convs."""
    def __init__(s, c, w, k=6):
        super().__init__(); s.c = c
        s.red = nn.Conv2d(256, c, 1)
        s.body = nn.Sequential(nn.Conv2d((k + 2) * c + 2 * k, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1),
                               nn.GELU(), nn.Conv2d(w, 256, 1))

    def forward(s, x):
        n, _, h, w = x.shape; k = 6
        K = x[:, :k * 256].reshape(n * k, 256, h, w); rest = x[:, k * 256:]
        v, ph, Ft, B = rest[:, :k], rest[:, k:2 * k], rest[:, 2 * k:2 * k + 256], rest[:, 2 * k + 256:]
        rk = s.red(K).reshape(n, k, s.c, h, w) * v[:, :, None]
        return s.body(torch.cat([rk.reshape(n, k * s.c, h, w), s.red(Ft), s.red(B), v, ph], 1))


def main():
    torch.manual_seed(0)
    cache = f'{OUT}/capacity_crops.pt'
    if os.path.exists(cache):
        tr, te = torch.load(cache, weights_only=False)
    else:
        pred = image_predictor()
        feat, pix = load_net(pred, NETS['learned']), load_net(pred, NETS['pixel'])
        trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
        tr = sum([collect(pred, feat, pix, trials[i]) for i in TRAIN], []); te = sum([collect(pred, feat, pix, trials[i]) for i in TEST], [])
        torch.save((tr, te), cache); del pred, feat, pix
    mu = torch.load('results/benchmark/rr/feat_integral/distil/refined_s4-7.pt')['agg.mu16'].cuda().float()
    T = lambda b: torch.stack([x['T'] for x in b]).cuda().float()

    def fit(f, data):
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
            return float(np.mean([ccos(f(data[i:i + 16]).float(), T(data[i:i + 16]), mu).mean().item() for i in range(0, len(data), 16)]))
    for nm, f in (('B', lambda b: torch.stack([x['B'] for x in b]).cuda().float()),
                  ('F_t', lambda b: torch.stack([x['Ft'] for x in b]).cuda().float()),
                  ('mean K', lambda b: (lambda K, v: (K * v[:, :, None]).sum(1) / v[:, :, None].sum(1).clamp(min=1))(
                      torch.stack([x['K'] for x in b]).cuda().float(), torch.stack([x['v'] for x in b]).cuda().float()))):
        print(f'reference {nm:6s}: train {fit(f, tr):.3f} held-out {fit(f, te):.3f}', flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'small':
        c, w = int(sys.argv[2]), int(sys.argv[3]); net = Small(c, w).cuda(); tag = f'small c{c} w{w}'
    else:
        cin = inputs(tr[:1]).shape[1]; tag = 'big'
        net = nn.Sequential(nn.Conv2d(cin, 512, 1), nn.GELU(), nn.Conv2d(512, 512, 3, padding=1), nn.GELU(),
                            nn.Conv2d(512, 512, 3, padding=1), nn.GELU(), nn.Conv2d(512, 256, 1)).cuda()
    x0 = torch.randn(1, *inputs(tr[:1]).shape[1:2], 64, 64, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
        for _ in range(20):
            net(x0)
        torch.cuda.synchronize(); ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; ev[0].record()
        for _ in range(100):
            net(x0)
        ev[1].record(); torch.cuda.synchronize()
    print(f'[{tag}] {ev[0].elapsed_time(ev[1]) / 100:.2f} ms/frame on the 64×64 grid', flush=True)
    if os.environ.get('TIME_ONLY'):
        return
    print(f'[{tag}] NFO train {len(tr)} / held-out {len(te)} frames; student {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M params', flush=True)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    f = lambda b: net(inputs(b))
    rng = np.random.default_rng(0)
    for it in range(1, 3001):
        b = [tr[i] for i in rng.choice(len(tr), 8, replace=False)]
        with torch.autocast('cuda', dtype=AMP):
            loss = 1 - ccos(f(b).float(), T(b), mu).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            net.eval(); print(f'[{tag}] it {it}: teacher cos train {fit(f, tr):.3f} held-out {fit(f, te):.3f}', flush=True); net.train()


if __name__ == '__main__':
    main()
