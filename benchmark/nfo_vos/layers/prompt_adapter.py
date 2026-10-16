"""Composite/background fusion adapter → gated dense prompt for a frozen, memory-free SAM2
(docs/layer_sam2/training/plan_fusion_prompt_adapter.md §2, §4, §4b; sparse prompt = box + point after O1, §10).

  per level ℓ ∈ {16, 8, 4}:  z = [P F_C̃, P F_B, P(F_C̃ − F_B), cos(F_C̃, F_B)]   (P: shared 1×1 conv → 32; Daudt et al. 2018)
  s16 block also gets the point map (Gaussian, σ 1 cell) and the box map; s4 block gets σ(l_C), σ(l_B); every block gets
  the composite's evidence mass a = Σ w G (pooled to its grid): C alone discards it, C̃ = a·C + (1 − a)·bg conflates it
  top-down s16 → s8 → s4 (3×3 conv, GroupNorm, GELU, width 64) → m (prompt logits), g (gate) at 256²
  dense = ↓g · mask_downscaling(m) + (1 − ↓g) · no_mask_embed   (prompt_encoder.py:57, 66; mask_decoder.py:205)
  dense = dense₀ + α·↓g·(mask_downscaling(m) − dense₀), α a learnable scalar initialised to 0 (ControlNet-style zero
  init; Zhang et al., ICCV 2023, arXiv 2302.05543): at step 0 exactly box + point SAM2. (A sigmoid gate at σ(−4) =
  0.018 already moved logits by up to 18.5: mask_downscaling(m) ≫ no_mask_embed.)
Loss: 20·focal + dice on the frozen decoder's 256² logits vs V (soft), + 0.1·BCE(m, S ∧ O).
Frozen SAM2.1 b+ (bf16 autocast, gradients flow through the decoder's activations only); adapter fp32.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.prompt_adapter [iters] [seed]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from benchmark.nfo_vos import run_t4 as R
from benchmark.nfo_vos.layers.prompt_data import Online, collate, encode

OUT = 'results/benchmark/rr/layers_prompt_adapter'


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class Adapter(nn.Module):
    def __init__(self, d=32, w=64):
        super().__init__()
        self.P = nn.ModuleDict({'16': nn.Conv2d(256, d, 1), '8': nn.Conv2d(64, d, 1), '4': nn.Conv2d(32, d, 1)})
        z = 3 * d + 1
        self.b16, self.b8, self.b4 = block(z + 3, w), block(z + w + 1, w), block(z + w + 3, w)   # +1: evidence mass a
        self.m, self.g = nn.Conv2d(w, 1, 1), nn.Conv2d(w, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def fuse(self, l, Fc, Fb):
        pc, pb = self.P[l](Fc), self.P[l](Fb)
        return torch.cat([pc, pb, pc - pb, F.cosine_similarity(Fc, Fb, dim=1)[:, None]], 1)

    def forward(self, b):
        n = b['pt'].shape[0]
        yy, xx = torch.meshgrid(torch.arange(64, device=b['pt'].device) + 0.5,
                                torch.arange(64, device=b['pt'].device) + 0.5, indexing='ij')
        p = b['pt'] / 16
        pmap = torch.exp(-((xx - p[:, 0, None, None]) ** 2 + (yy - p[:, 1, None, None]) ** 2) / 2)[:, None]
        bx = b['box'] / 16
        bmap = ((xx >= bx[:, 0, None, None]) & (xx <= bx[:, 2, None, None]) & (yy >= bx[:, 1, None, None])
                & (yy <= bx[:, 3, None, None])).float()[:, None]
        a = b['a'][:, None]
        x = self.b16(torch.cat([self.fuse('16', b['C16'], b['B16']), pmap, bmap, F.avg_pool2d(a, 4)], 1))
        x = self.b8(torch.cat([self.fuse('8', b['C8'], b['B8']), F.interpolate(x, scale_factor=2, mode='bilinear'),
                               F.avg_pool2d(a, 2)], 1))
        lo = torch.stack([torch.sigmoid(b['lC']), torch.sigmoid(b['lB'])], 1)
        x = self.b4(torch.cat([self.fuse('4', b['C4'], b['B4']), F.interpolate(x, scale_factor=2, mode='bilinear'), lo, a], 1))
        return self.m(x), self.alpha * torch.sigmoid(self.g(x))


def decode(sam, b, m=None, g=None, box=True):
    """Frozen SAM2 decoder on b['X*'] features with box + point (box=False: point only); dense = gated mix (m, g None →
    plain prompt)."""
    pe, dec = sam.sam_prompt_encoder, sam.sam_mask_decoder
    n = b['pt'].shape[0]
    if box:
        coords = torch.cat([b['box'].view(n, 2, 2), b['pt'][:, None]], 1)
        labels = torch.tensor([[2, 3, 1]], device=coords.device).repeat(n, 1)
    else:
        coords, labels = b['pt'][:, None], torch.ones((n, 1), dtype=torch.int, device=b['pt'].device)
    sparse, dense0 = pe(points=(coords, labels), boxes=None, masks=None)
    if m is not None:
        g64 = F.avg_pool2d(g, 4)
        dense = dense0 + g64 * (pe.mask_downscaling(m) - dense0)
    else:
        dense = dense0
    low, _, _, _ = dec(image_embeddings=b['X16'], image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=sparse,
                       dense_prompt_embeddings=dense, multimask_output=False, repeat_image=False,
                       high_res_features=[b['X4'], b['X8']])
    return low


def loss_fn(low, m, b):
    V = b['V'][:, None]
    p = torch.sigmoid(low)
    dice = 1 - (2 * (p * V).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + V.sum((1, 2, 3)) + 1)
    focal = sigmoid_focal_loss(low, V, reduction='none').mean((1, 2, 3))
    aux = F.binary_cross_entropy_with_logits(m, b['SO'][:, None])
    return (20 * focal + dice).mean() + 0.1 * aux


@torch.no_grad()
def evaluate(pred, net, val):
    """J / P / R vs native V per frame (low-res logits bilinearly upsampled, as SAM2Transforms.postprocess_masks).
    val: list of collated batches (the fixed O1 clip set, generated once)."""
    net.eval()
    rows = []
    for raw in val:
        b = encode(pred, raw)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m, g = net(b)
            outs = {'box+pt': decode(pred.model, b), 'adapter': decode(pred.model, b, m, g)}
        for i in range(len(raw['cid'])):
            Vt = torch.from_numpy(raw['Vn'][i]).cuda()
            r = {'mode': int(raw['mode'][i]), 'clip': raw['cid'][i], 'gate': float(g[i].mean())}
            for a, low in outs.items():
                pr = F.interpolate(low[i:i + 1].float(), Vt.shape, mode='bilinear', align_corners=False)[0, 0] > 0
                tp = (pr & Vt).sum().item()
                r[a] = (tp / max((pr | Vt).sum().item(), 1), tp / max(pr.sum().item(), 1), tp / max(Vt.sum().item(), 1))
            rows.append(r)
    net.train()
    return rows


def summarise(rows, rng=np.random.default_rng(0)):
    clips = sorted({r['clip'] for r in rows})
    per = {a: np.array([np.mean([r[a] for r in rows if r['clip'] == c], 0) for c in clips]) for a in ('box+pt', 'adapter')}
    dj = per['adapter'][:, 0] - per['box+pt'][:, 0]
    bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(5000)]
    s = {a: dict(J=v[:, 0].mean(), P=v[:, 1].mean(), R=v[:, 2].mean()) for a, v in per.items()}
    s['dJ'], s['ci'] = float(dj.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    s['by_mode'] = {m: {a: float(np.mean([r[a][0] for r in rows if r['mode'] == m])) for a in ('box+pt', 'adapter')}
                    | {'n': sum(r['mode'] == m for r in rows)} for m in range(4)}
    s['gate'] = float(np.mean([r['gate'] for r in rows]))
    return s


def main(iters=2000, seed=0, eval_every=500):
    torch.manual_seed(seed); np.random.seed(seed)
    pred = R.image_predictor()
    sam = pred.model.eval()
    for p_ in sam.parameters():
        p_.requires_grad_(False)
    net = Adapter().cuda()
    n_par = sum(p_.numel() for p_ in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    dl = iter(torch.utils.data.DataLoader(Online('train', seed), batch_size=16, num_workers=10, collate_fn=collate,
                                          prefetch_factor=2, persistent_workers=True))
    val = list(torch.utils.data.DataLoader(Online('val'), batch_size=8, num_workers=10, collate_fn=collate))
    out = f'{OUT}/seed{seed}'
    os.makedirs(out, exist_ok=True)
    print(f'adapter params {n_par}; val frames {sum(len(v["cid"]) for v in val)}', flush=True)
    log, t0 = [], time.time()
    for it in range(1, iters + 1):
        b = encode(pred, next(dl))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m, g = net(b)
            low = decode(sam, b, m, g)
            loss = loss_fn(low.float(), m.float(), b)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 100 == 0:
            print(f'it {it} loss {loss.item():.4f} gate {g.mean().item():.3f} alpha {net.alpha.item():+.3f} '
                  f'{(time.time() - t0) / it:.2f} s/it', flush=True)
        if it % eval_every == 0 or it == iters:
            s = summarise(evaluate(pred, net, val))
            s.update(it=it, sec=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            log.append(s)
            print(f"VAL it {it}: box+pt J {s['box+pt']['J']:.3f} | adapter J {s['adapter']['J']:.3f} "
                  f"P {s['adapter']['P']:.3f} R {s['adapter']['R']:.3f} | ΔJ {s['dJ']:+.3f} "
                  f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] gate {s['gate']:.3f} | by mode "
                  + ' '.join(f"{k}:{v['box+pt']:.2f}→{v['adapter']:.2f}(n{v['n']})" for k, v in s['by_mode'].items()),
                  flush=True)
            torch.save(net.state_dict(), f'{out}/adapter_it{it}.pt')
            json.dump(log, open(f'{out}/log.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:]))
