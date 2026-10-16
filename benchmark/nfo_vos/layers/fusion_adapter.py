"""Approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11): residual fusion of the frame, the
tracker-aligned composite and the empty scene on the three SAM2 feature levels the frozen decoder reads.

  per level ℓ ∈ {16, 8, 4}:  z^ℓ = [P F_t, P F_C, P F_B, P F_t − P F_C, P F_t − P F_B, a^ℓ]   (P shared 1×1 conv)
  top-down (FPN-style): x16 = block(z16); x8 = block(z8 ⊕ ↑x16); x4 = block(z4 ⊕ ↑x8);  R^ℓ = 1×1 conv(x^ℓ)
  F̂_t^ℓ = F_t^ℓ + R^ℓ, the final 1×1 conv of R^ℓ zero-initialised (ControlNet zero conv; Zhang et al., ICCV 2023,
  arXiv 2302.05543; replaces a scalar α, under which R received no gradient until α moved)
  F̂^16 is the decoder's image embedding; F̂^8, F̂^4 its high-res features (mask_decoder.py:205, 224-225)
  sparse prompt: token mode (default) w·[SAM2 point embedding at the a-weighted composite centroid + zero-init
  MLP(Σ a F_C^16 / Σ a)] + (1 − w)·learned fallback token, w = 1 − exp(−Σa / κ), κ learned (init 500 cells of 256²):
  a point prompt at step 0 when there is evidence, a learned prompt when there is none (PerSAM-style, arXiv 2305.03048); box mode (ablation) the tracker box, exactly box-only SAM2 at init.
Frozen SAM2.1 b+; loss 20·focal + Dice vs V on the decoder's 256² logits.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.fusion_adapter [token|box] [steps] [seed]
"""
import json
import os
import sys
import time

import numpy as np
import torch

AMP = torch.float16 if os.environ.get('AMP') == 'fp16' else torch.bfloat16   # fp16 on pre-Ampere GPUs
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from benchmark.nfo_vos import run_t4 as R
from benchmark.nfo_vos.layers.fusion_data import Clips, encode

OUT = 'results/benchmark/rr/layers_fusion_adapter'
CH, D_ = {'16': 256, '8': 64, '4': 32}, {'16': 64, '8': 32, '4': 32}
POOL = {'16': 4, '8': 2, '4': 1}


def _ident(x):            # DataLoader(batch_size=None) would otherwise convert numpy arrays to tensors
    return x


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class Fusion(nn.Module):
    def __init__(self, token=True, w=64):
        super().__init__()
        self.token = token
        self.P = nn.ModuleDict({l: nn.Conv2d(CH[l], D_[l], 1) for l in CH})
        zin = {l: 5 * D_[l] + 1 for l in CH}
        self.blk = nn.ModuleDict({'16': block(zin['16'], w), '8': block(zin['8'] + w, w), '4': block(zin['4'] + w, w)})
        self.out = nn.ModuleDict({l: nn.Conv2d(w, CH[l], 1) for l in CH})
        for l in CH:
            nn.init.zeros_(self.out[l].weight); nn.init.zeros_(self.out[l].bias)
        if token:
            self.tok = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
            nn.init.zeros_(self.tok[2].weight); nn.init.zeros_(self.tok[2].bias)
            self.fallback = nn.Parameter(torch.zeros(2, 256))      # set from SAM2's centre-point embedding (init_fallback)
            self.log_kappa = nn.Parameter(torch.tensor(float(np.log(500.0))))

    def z(self, f, l):
        pt, pc, pb = self.P[l](f[f'X{l}']), self.P[l](f[f'C{l}']), self.P[l](f[f'B{l}'])
        a = F.avg_pool2d(f['a'][:, None], POOL[l]) if POOL[l] > 1 else f['a'][:, None]
        return torch.cat([pt, pc, pb, pt - pc, pt - pb, a], 1)

    def forward(self, f, sam=None):
        up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x16 = self.blk['16'](self.z(f, '16'))
        x8 = self.blk['8'](torch.cat([self.z(f, '8'), up(x16)], 1))
        x4 = self.blk['4'](torch.cat([self.z(f, '4'), up(x8)], 1))
        o = dict(f)
        for l, x in (('16', x16), ('8', x8), ('4', x4)):
            o[f'X{l}'] = f[f'X{l}'] + self.out[l](x)
        if self.token and sam is not None:
            # token = SAM2's own positive-point embedding at the evidence-weighted centroid of the composite (a working
            # prompt at step 0) + a zero-initialised learned correction from the composite's prototype feature
            a = f['a']
            n = a.shape[0]
            g = (torch.arange(256, device=a.device, dtype=a.dtype) + 0.5) * 4          # 256² grid → 1024 input frame
            m = a.sum((1, 2))
            c = torch.stack([(a.sum(1) * g).sum(1), (a.sum(2) * g).sum(1)], 1) / m.clamp(min=1e-6)[:, None]
            c = torch.where((m > 1e-3)[:, None], c, torch.full_like(c, 512.0))
            with torch.no_grad():
                sp, _ = sam.sam_prompt_encoder(points=(c[:, None], torch.ones((n, 1), dtype=torch.int, device=a.device)),
                                               boxes=None, masks=None)
            a16 = F.avg_pool2d(a[:, None], 4)
            proto = (f['C16'] * a16).sum((2, 3)) / a16.sum((2, 3)).clamp(min=1e-3)
            delta = self.tok(proto)[:, None]
            w = (1 - torch.exp(-m / self.log_kappa.exp()))[:, None, None]
            o['token'] = w * (sp + torch.cat([delta, torch.zeros_like(sp[:, 1:])], 1)) + (1 - w) * self.fallback[None]
        return o


    @torch.no_grad()
    def init_fallback(self, sam):
        c = torch.tensor([[[512.0, 512.0]]], device=self.fallback.device)
        sp, _ = sam.sam_prompt_encoder(points=(c, torch.ones((1, 1), dtype=torch.int, device=c.device)), boxes=None,
                                       masks=None)
        self.fallback.copy_(sp[0])


def decode(sam, f, token=None, point=False):
    """Frozen SAM2 decoder on f['X*']. token None: tracker box (+ its centre point if point) as the sparse prompt."""
    pe, dec = sam.sam_prompt_encoder, sam.sam_mask_decoder
    n = f['X16'].shape[0]
    if token is None:
        coords, labels = f['box'].view(n, 2, 2), torch.tensor([[2, 3]], device=f['box'].device).repeat(n, 1)
        if point:
            coords = torch.cat([coords, f['pt'][:, None]], 1)
            labels = torch.cat([labels, torch.ones((n, 1), dtype=labels.dtype, device=labels.device)], 1)
        sparse, dense = pe(points=(coords, labels), boxes=None, masks=None)
    else:
        _, dense = pe(points=None, boxes=None, masks=None)
        sparse, dense = token, dense.expand(n, -1, -1, -1)
    low, _, _, _ = dec(image_embeddings=f['X16'], image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=sparse,
                       dense_prompt_embeddings=dense, multimask_output=False, repeat_image=False,
                       high_res_features=[f['X4'], f['X8']])
    return low


def loss_fn(low, V):
    V = V[:, None]
    p = torch.sigmoid(low)
    dice = 1 - (2 * (p * V).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + V.sum((1, 2, 3)) + 1)
    return (20 * sigmoid_focal_loss(low, V, reduction='none').mean((1, 2, 3)) + dice).mean()


@torch.no_grad()
def evaluate(pred, net, val):
    """J / P / R vs native V; arms: box-only SAM2, box + point SAM2 (the G2 control), adapter."""
    net.eval()
    rows = []
    for clip in val:
        f = encode(pred, [clip])
        with torch.autocast('cuda', dtype=AMP):
            o = net(f, sam=pred.model)
            outs = {'box': decode(pred.model, f), 'box+pt': decode(pred.model, f, point=True),
                    'adapter': decode(pred.model, o, token=o.get('token'))}
        for i, Vn in enumerate(clip['Vn']):
            Vt = torch.from_numpy(Vn).cuda()
            r = dict(mode=int(clip['mode'][i]), clip=clip['cid'], rho=clip['occ']['rho'], clump=str(clip['occ']['clump']))
            for a, low in outs.items():
                pr = F.interpolate(low[i:i + 1].float(), Vt.shape, mode='bilinear', align_corners=False)[0, 0] > 0
                tp = (pr & Vt).sum().item()
                r[a] = (tp / max((pr | Vt).sum().item(), 1), tp / max(pr.sum().item(), 1), tp / max(Vt.sum().item(), 1))
            rows.append(r)
    net.train()
    return rows


def summarise(rows, rng=None):
    rng = rng or np.random.default_rng(0)
    arms = ('box', 'box+pt', 'adapter')
    clips = sorted({r['clip'] for r in rows})
    per = {a: np.array([np.mean([r[a] for r in rows if r['clip'] == c], 0) for c in clips]) for a in arms}
    dj = per['adapter'][:, 0] - per['box+pt'][:, 0]
    bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(5000)]
    s = {a: dict(zip('JPR', per[a].mean(0))) for a in arms}
    s['dJ'], s['ci'] = float(dj.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    groups = {f'mode {m}': [r for r in rows if r['mode'] == m] for m in range(4)}
    groups |= {f'rho {lo}-{hi}': [r for r in rows if lo <= r['rho'] < hi] for lo, hi in ((0.0, 0.01), (0.2, 0.33), (0.33, 0.47), (0.47, 0.61))}
    groups |= {f'clump {c}': [r for r in rows if r['clump'] == c] for c in ('12', '24', '(12, 24)')}
    s['groups'] = {g: {a: float(np.mean([r[a][0] for r in rs])) for a in arms} | {'n': len(rs)} for g, rs in groups.items() if rs}
    return s


def main(mode='token', steps=1500, seed=0, eval_every=250, clips_per_step=2):
    torch.manual_seed(seed); np.random.seed(seed)
    pred = R.image_predictor()
    sam = pred.model.eval()
    for p_ in sam.parameters():
        p_.requires_grad_(False)
    net = Fusion(token=mode == 'token').cuda()
    if net.token:
        net.init_fallback(sam)
    n_par = sum(p_.numel() for p_ in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16)     # no-op under bf16
    dl = iter(torch.utils.data.DataLoader(Clips('train', seed), batch_size=None, num_workers=8, prefetch_factor=2, collate_fn=_ident,
                                          persistent_workers=True))
    val = list(torch.utils.data.DataLoader(Clips('val'), batch_size=None, num_workers=8, collate_fn=_ident))
    from benchmark.nfo_vos.layers import fusion_nfo
    nfo = fusion_nfo.items()
    fusion_nfo.feature_diff_montage(pred, nfo)
    s0 = fusion_nfo.evaluate(pred, net, nfo)
    print('NFO it 0 (seq2-4 handAv, n %d): ' % len(nfo) + ' | '.join(f"{a} J {v['J']:.3f} P {v['P']:.3f} R {v['R']:.3f}" for a, v in s0.items()), flush=True)
    out = f'{OUT}/{mode}_seed{seed}'
    os.makedirs(out, exist_ok=True)
    print(f'mode {mode}; params {n_par}; val clips {len(val)}, frames {sum(len(c["X"]) for c in val)}', flush=True)
    log, t0 = [], time.time()
    for it in range(1, steps + 1):
        f = encode(pred, [next(dl) for _ in range(clips_per_step)])
        with torch.autocast('cuda', dtype=AMP):
            o = net(f, sam=sam)
            loss = loss_fn(decode(sam, o, token=o.get('token')).float(), f['V'])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 50 == 0:
            al = ' '.join(f"|W{l}| {net.out[l].weight.abs().mean().item():.1e}" for l in CH)
            print(f'it {it} loss {loss.item():.4f} {al} {(time.time() - t0) / it:.2f} s/it', flush=True)
        if it % eval_every == 0 or it == steps:
            s = summarise(evaluate(pred, net, val))
            s['nfo'] = fusion_nfo.evaluate(pred, net, nfo)
            s.update(it=it, sec=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            log.append(s)
            print(f"VAL it {it}: box J {s['box']['J']:.3f} | box+pt J {s['box+pt']['J']:.3f} | adapter J "
                  f"{s['adapter']['J']:.3f} P {s['adapter']['P']:.3f} R {s['adapter']['R']:.3f} | ΔJ vs box+pt "
                  f"{s['dJ']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]", flush=True)
            print('  ' + ' | '.join(f"{g}: {v['box+pt']:.2f}→{v['adapter']:.2f} (n {v['n']})" for g, v in s['groups'].items()),
                  flush=True)
            print(f'NFO it {it}: ' + ' | '.join(f"{a} J {v['J']:.3f} P {v['P']:.3f} R {v['R']:.3f}" for a, v in s['nfo'].items()), flush=True)
            torch.save(net.state_dict(), f'{out}/fusion_it{it}.pt')
            json.dump(log, open(f'{out}/log.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'token', *(int(x) for x in a[1:]))
