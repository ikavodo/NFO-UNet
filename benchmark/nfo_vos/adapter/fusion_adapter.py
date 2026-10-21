"""Approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11): residual fusion of the frame, the
tracker-aligned composite and the empty scene on the three SAM2 feature levels the frozen decoder reads.

  per level ℓ ∈ {16, 8, 4}:  z^ℓ = [P F_t, P F_C, P F_B, P F_t − P F_C, P F_t − P F_B, a^ℓ]   (P shared 1×1 conv)
  top-down (FPN-style): x16 = block(z16); x8 = block(z8 ⊕ ↑x16); x4 = block(z4 ⊕ ↑x8);  R^ℓ = 1×1 conv(x^ℓ)
  F̂_t^ℓ = F_t^ℓ + R^ℓ, the final 1×1 conv of R^ℓ zero-initialised (ControlNet zero conv; Zhang et al., ICCV 2023,
  arXiv 2302.05543; replaces a scalar α, under which R received no gradient until α moved)
  F̂^16 is the decoder's image embedding; F̂^8, F̂^4 its high-res features (mask_decoder.py:205, 224-225)
  sparse prompt: token mode (default; v3.2) SAM2 prompt embedding of the kernel tracker's box (t0 size) + its point, the
  point token corrected by w·zero-init MLP(composite prototype); at step 0 = SAM2 with the GT-free tracker box + point. Without a track (f['trk'] absent): w·[point embedding at the a-weighted composite centroid + zero-init
  MLP(Σ a F_C^16 / Σ a)] + (1 − w)·learned fallback token, w = 1 − exp(−Σa / κ), κ learned (init 500 cells of 256²):
  a point prompt at step 0 when there is evidence, a learned prompt when there is none (PerSAM-style, arXiv 2305.03048); box mode (ablation) the tracker box, exactly box-only SAM2 at init.
Frozen SAM2.1 b+; loss 20·focal + Dice vs V on the decoder's 256² logits.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.adapter.fusion_adapter [token|box] [steps] [seed]
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

from benchmark.nfo_vos.adapter.fusion_data import Clips, build, encode

OUT = 'results/benchmark/rr/layers_fusion_adapter'
SAM2_CKPT = os.path.abspath(os.environ.get('NFO_SAM2_CKPT', '../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt'))
_predictor = None


def image_predictor():
    """The frozen SAM2.1 b+ image predictor (one per process) the adapter is attached to."""
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml', SAM2_CKPT, device='cuda'))
    return _predictor
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
            if 'trk' in f:                       # anchor = the kernel tracker's position at t (2026-10-16): the evidence
                c = f['trk']                     # centroid is ~80% diffuse background mass and lands ~0.6 h off target
            pts, lab = c[:, None], torch.ones((n, 1), dtype=torch.int, device=a.device)
            if 'trk' in f and 'box_trk' in f:    # v3.2: tracker box (t0 size) + tracker point; δ corrects the point token
                pts = torch.cat([f['box_trk'].view(n, 2, 2), pts], 1)
                lab = torch.cat([torch.tensor([[2, 3]], device=a.device).repeat(n, 1), lab], 1)
            with torch.no_grad():
                sp, _ = sam.sam_prompt_encoder(points=(pts, lab), boxes=None, masks=None)
            ip = pts.shape[1] - 1                # index of the point token
            a16 = F.avg_pool2d(a[:, None], 4)
            proto = (f['C16'] * a16).sum((2, 3)) / a16.sum((2, 3)).clamp(min=1e-3)
            delta = self.tok(proto)[:, None]
            w = (1 - torch.exp(-m / self.log_kappa.exp()))[:, None, None]
            if 'trk' in f:                       # location always known: no fallback; the composite only corrects (δ ∝ w)
                corr = torch.zeros_like(sp); corr[:, ip] = (w * delta)[:, 0]
                o['token'] = sp + corr
            else:
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


ALPHA = 0.67   # focal class weight: a missed person pixel costs 2.07× an extra background pixel (old default 0.25: 0.38×)
# LOSS env (2026-10-19, re-selection on valid labels): v34 = layer-aware recall-heavy (default); balanced = same W,
# focal α 0.5, Tversky FN 0.5 / FP 0.5; simple = balanced without the layer weights W
LOSS = os.environ.get('LOSS', 'v34')
LOSS_CFG = {'v34': dict(alpha=0.67, a_fp=0.3, b_fn=0.7, use_w=True), 'balanced': dict(alpha=0.5, a_fp=0.5, b_fn=0.5, use_w=True),
            'simple': dict(alpha=0.5, a_fp=0.5, b_fn=0.5, use_w=False)}[LOSS]


def loss_fn(low, V, W=None, a_fp=0.3, b_fn=0.7, gamma=0.75, alpha=ALPHA, use_w=True):
    """v3.4 (2026-10-16): layer-aware, recall-oriented. Per-pixel weights W (fusion_data.loss_weights: occluded person
    near visible fragments ignored, far occluded person 0.3, elsewhere 1) on focal (α = ALPHA) and on the FP sum of a
    Focal Tversky term (Abraham & Khan, ISBI 2019, arXiv 1810.07842; FN 0.7, FP 0.3, γ 0.75).
    History: v3.2 focal α 0.25 + Dice was precision-biased (FN/FP cost 0.38); v3.3 α 0.67 + Tversky still lost recall
    on seq1 (R 0.76 → 0.67 by step 200): a global price cannot be generous behind foliage and strict on background."""
    V = V[:, None]
    W = torch.ones_like(V) if (W is None or not use_w) else W[:, None]
    p = torch.sigmoid(low)
    tp, fp, fn = (p * V).sum((1, 2, 3)), (W * p * (1 - V)).sum((1, 2, 3)), ((1 - p) * V).sum((1, 2, 3))
    ft = (1 - (tp + 1) / (tp + a_fp * fp + b_fn * fn + 1)) ** gamma
    focal = (W * sigmoid_focal_loss(low, V, alpha=alpha, reduction='none')).sum((1, 2, 3)) / W.sum((1, 2, 3)).clamp(min=1)
    return (20 * focal + ft).mean()


@torch.no_grad()
def evaluate(pred, net, val):
    """Synthetic val (raw clips, built with the test-time path): J / P / R vs native V; arms box, box+pt, trk-box,
    adapter. Also the input statistics used for the parity check against NFO."""
    net.eval()
    rows, a_mass, a_on = [], [], []
    for raw in val:
        f, meta = build(pred, raw)
        fb = dict(f, box=f['box_trk'])          # tracker centre, t0 box size: the same GT-free control as on NFO
        with torch.autocast('cuda', dtype=AMP):
            o = net(f, sam=pred.model)
            outs = {'box': decode(pred.model, f), 'box+pt': decode(pred.model, f, point=True),
                    'trk-box': decode(pred.model, fb), 'adapter': decode(pred.model, o, token=o.get('token'))}
        for i, Vn in enumerate(meta['Vn']):
            Vt = torch.from_numpy(Vn).cuda()
            r = dict(mode=int(meta['mode'][i]), clip=meta['cid'], rho=meta['occ']['rho'], clump=str(meta['occ']['clump']))
            for a, low in outs.items():
                pr = F.interpolate(low[i:i + 1].float(), Vt.shape, mode='bilinear', align_corners=False)[0, 0] > 0
                tp = (pr & Vt).sum().item()
                r[a] = (tp / max((pr | Vt).sum().item(), 1), tp / max(pr.sum().item(), 1), tp / max(Vt.sum().item(), 1))
            rows.append(r)
        a_mass += meta['a_mass']; a_on += meta['a_on']
    net.train()
    return rows, dict(a_mass=float(np.median(a_mass)), a_on=float(np.nanmean(a_on)))


def summarise(rows, rng=None):
    rng = rng or np.random.default_rng(0)
    arms = ('box', 'box+pt', 'trk-box', 'adapter')
    clips = sorted({r['clip'] for r in rows})
    per = {a: np.array([np.mean([r[a] for r in rows if r['clip'] == c], 0) for c in clips]) for a in arms}
    s = {a: dict(zip('JPR', per[a].mean(0))) for a in arms}
    for ref in ('box+pt', 'trk-box'):
        dj = per['adapter'][:, 0] - per[ref][:, 0]
        bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(2000)]
        s[f'dJ_{ref}'] = [float(dj.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    groups = {f'mode {m}': [r for r in rows if r['mode'] == m] for m in (0, 2)}
    groups |= {f'rho {lo}-{hi}': [r for r in rows if lo <= r['rho'] < hi] for lo, hi in ((0.0, 0.01), (0.2, 0.4), (0.4, 0.61))}
    s['groups'] = {g: {a: float(np.mean([r[a][0] for r in rs])) for a in arms} | {'n': len(rs)} for g, rs in groups.items() if rs}
    return s


def fmt(d, arms):
    return ' | '.join(f"{a} {d[a]['J']:.3f}/{d[a]['P']:.2f}/{d[a]['R']:.2f}" for a in arms if a in d)


def main(mode='token', steps=1500, seed=0, eval_every=100, max_targets=32, n_val_fast=12, test=True):
    """One clip per step (≤ max_targets frames), built with the test-time path (fusion_data.build). Fast validation every
    `eval_every` steps: 12 synthetic val clips + the seq1 brush SELECTION set, with the parity statistics and the
    composite-removal sanity arm; best checkpoint by seq1 adapter J. At the end: full synthetic val (30 clips) and ONE
    evaluation of the selected checkpoint on the seq2–4 handAv TEST set.
    Printed J/P/R triples; 'GAP' = (adapter − trk-box) on synthetic vs on seq1: the domain gap in the quantity that matters."""
    from benchmark.nfo_vos.adapter import fusion_nfo
    torch.manual_seed(seed); np.random.seed(seed)
    pred = image_predictor()
    sam = pred.model.eval()
    for p_ in sam.parameters():
        p_.requires_grad_(False)
    net = Fusion(token=mode == 'token').cuda()
    if net.token:
        net.init_fallback(sam)
    n_par = sum(p_.numel() for p_ in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16)
    dl = iter(torch.utils.data.DataLoader(Clips('train', seed), batch_size=None, num_workers=6, prefetch_factor=2,
                                          collate_fn=_ident, persistent_workers=True))
    val_all = list(torch.utils.data.DataLoader(Clips('val'), batch_size=None, num_workers=6, collate_fn=_ident))
    val = val_all[:n_val_fast]
    sel = fusion_nfo.items('select', pred)
    out = f'{OUT}/{mode}_seed{seed}' + ('' if LOSS == 'v34' else f'_{LOSS}')
    os.makedirs(out, exist_ok=True)
    print(f'mode {mode}; params {n_par}; val clips {len(val)} fast / {len(val_all)} final; seq1 select frames {len(sel)}', flush=True)
    rng, log, best, t0 = np.random.default_rng(seed), [], (-1.0, None), time.time()

    def validate(it, rows_inp=None):
        rows, inp = evaluate(pred, net, val)
        s = summarise(rows); ss = fusion_nfo.evaluate(pred, net, sel)
        s.update(it=it, sec=time.time() - t0, sel=ss, inputs_syn=inp)
        gap_syn = s['adapter']['J'] - s['trk-box']['J']; gap_sel = ss['adapter']['J'] - ss['trk-box']['J']
        print(f"VAL it {it}: " + fmt(s, ('box+pt', 'trk-box', 'adapter')) + f" | ΔJ vs trk-box {s['dJ_trk-box'][0]:+.3f}", flush=True)
        print(f"SEL it {it}: " + fmt(ss, ('box+pt', 'trk-box', 'adapter', 'adapter-noC')), flush=True)
        print(f"GAP it {it}: adapter − trk-box  synthetic {gap_syn:+.3f}  seq1 {gap_sel:+.3f} | inputs Σa median syn "
              f"{inp['a_mass']:.0f} / seq1 {ss['inputs']['a_mass']:.0f}, a on person syn {inp['a_on']:.2f} / seq1 {ss['inputs']['a_on']:.2f}",
              flush=True)
        return s

    log.append(validate(0))
    for it in range(1, steps + 1):
        f, _ = build(pred, next(dl), max_targets=max_targets, rng=rng)
        with torch.autocast('cuda', dtype=AMP):
            o = net(f, sam=sam)
            loss = loss_fn(decode(sam, o, token=o.get('token')).float(), f['V'], f.get('W'), **LOSS_CFG)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 25 == 0:
            print(f'it {it} loss {loss.item():.4f} {(time.time() - t0) / it:.2f} s/it', flush=True)
        if it % eval_every == 0 or it == steps:
            s = validate(it)
            log.append(s)
            torch.save(net.state_dict(), f'{out}/fusion_it{it}.pt')
            if s['sel']['adapter']['J'] > best[0]:
                best = (s['sel']['adapter']['J'], it)
            json.dump(log, open(f'{out}/log.json', 'w'), indent=1, default=float)
    net.load_state_dict(torch.load(f'{out}/fusion_it{best[1]}.pt'))
    rows, inp = evaluate(pred, net, val_all if test else val)
    full = summarise(rows)
    print(f"FINAL synthetic val ({len(val_all)} clips, checkpoint it {best[1]}): " + fmt(full, ('box+pt', 'trk-box', 'adapter')), flush=True)
    if not test:                                     # smoke runs never touch the test set
        return
    test = fusion_nfo.evaluate(pred, net, fusion_nfo.items('test', pred))
    pf = test.pop('per_frame'); d = np.array(pf['adapter']) - np.array(pf['trk-box'])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    log.append(dict(final_val=full, test=test, test_dJ_trk=[float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    selected_it=best[1]))
    json.dump(log, open(f'{out}/log.json', 'w'), indent=1, default=float)
    print(f"TEST (seq2-4 handAv, it {best[1]} chosen on seq1): " + fmt(test, ('box', 'box+pt', 'trk-box', 'adapter', 'adapter-noC'))
          + f" | ΔJ adapter − trk-box {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]", flush=True)

if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'token', *(int(x) for x in a[1:]))
