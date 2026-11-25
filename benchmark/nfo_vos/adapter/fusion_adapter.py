"""Approach A (docs/layer_sam2/plan_fusion_prompt_adapter.md §11): residual fusion of the frame, the
tracker-aligned integral image and the empty scene on the three SAM2 feature levels the frozen decoder reads.

  per level ℓ ∈ {16, 8, 4}:  z^ℓ = [P F_t, P F_C, P F_B, P F_t − P F_C, P F_t − P F_B, a^ℓ]   (P shared 1×1 conv)
  top-down (FPN-style): x16 = block(z16); x8 = block(z8 ⊕ ↑x16); x4 = block(z4 ⊕ ↑x8);  R^ℓ = 1×1 conv(x^ℓ)
  F̂_t^ℓ = F_t^ℓ + R^ℓ, the final 1×1 conv of R^ℓ zero-initialised (ControlNet zero conv; Zhang et al., ICCV 2023,
  arXiv 2302.05543; replaces a scalar α, under which R received no gradient until α moved)
  F̂^16 is the decoder's image embedding; F̂^8, F̂^4 its high-res features (mask_decoder.py:205, 224-225)
  sparse prompt (v3.2): SAM2 prompt embedding of the kernel tracker's box (t0 size) + its point, the
  point token corrected by w·zero-init MLP(integral-image prototype); at step 0 = SAM2 with the GT-free tracker box + point. Without a track (f['trk'] absent): w·[point embedding at the a-weighted integral image centroid + zero-init
  MLP(Σ a F_C^16 / Σ a)] + (1 − w)·learned fallback token, w = 1 − exp(−Σa / κ), κ learned (init 500 cells of 256²):
  a point prompt at step 0 when there is evidence, a learned prompt when there is none (PerSAM-style, arXiv 2305.03048).
Frozen SAM2.1 b+; loss 20·focal + Dice vs V on the decoder's 256² logits.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.adapter.fusion_adapter [steps] [seed]
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

from benchmark.nfo_vos.adapter.fusion_data import Clips, a256, build

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
    def __init__(self, w=64):
        super().__init__()
        self.P = nn.ModuleDict({l: nn.Conv2d(CH[l], D_[l], 1) for l in CH})
        zin = {l: 5 * D_[l] + 1 for l in CH}
        self.blk = nn.ModuleDict({'16': block(zin['16'], w), '8': block(zin['8'] + w, w), '4': block(zin['4'] + w, w)})
        self.out = nn.ModuleDict({l: nn.Conv2d(w, CH[l], 1) for l in CH})
        for l in CH:
            nn.init.zeros_(self.out[l].weight); nn.init.zeros_(self.out[l].bias)
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
        o['_x4'], o['_x16'] = x4, x16          # trunk activations (AMODAL aux head)
        if sam is not None:
            # token = SAM2's own positive-point embedding at the evidence-weighted centroid of the integral image (a working
            # prompt at step 0) + a zero-initialised learned correction from the integral image's prototype feature
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
            if 'trk' in f:                       # location always known: no fallback; the integral image only corrects (δ ∝ w)
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


def decode(sam, f, token=None, point=False, aux=None):
    """Frozen SAM2 decoder on f['X*']. token None: tracker box (+ its centre point if point) as the sparse prompt.
    aux: a list that receives (IoU-head prediction, object-score logit) of the first item (gate signals)."""
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
    low, ious, _, obj = dec(image_embeddings=f['X16'], image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=sparse,
                            dense_prompt_embeddings=dense, multimask_output=False, repeat_image=False,
                            high_res_features=[f['X4'], f['X8']])
    if aux is not None:
        aux.append((float(ious[0, 0]), float(obj[0, 0])))
    return low


ALPHA = 0.67   # focal class weight: a missed person pixel costs 2.07× an extra background pixel (old default 0.25: 0.38×)


def loss_fn(low, V, W=None, a_fp=0.3, b_fn=0.7, gamma=0.75, alpha=ALPHA):
    """v3.4 (2026-10-16): layer-aware, recall-oriented. Per-pixel weights W (fusion_data.loss_weights: occluded person
    near visible fragments ignored, far occluded person 0.3, elsewhere 1) on focal (α = ALPHA) and on the FP sum of a
    Focal Tversky term (Abraham & Khan, ISBI 2019, arXiv 1810.07842; FN 0.7, FP 0.3, γ 0.75).
    History: v3.2 focal α 0.25 + Dice was precision-biased (FN/FP cost 0.38); v3.3 α 0.67 + Tversky still lost recall
    on seq1 (R 0.76 → 0.67 by step 200): a global price cannot be generous behind foliage and strict on background."""
    V = V[:, None]
    W = torch.ones_like(V) if W is None else W[:, None]
    p = torch.sigmoid(low)
    tp, fp, fn = (p * V).sum((1, 2, 3)), (W * p * (1 - V)).sum((1, 2, 3)), ((1 - p) * V).sum((1, 2, 3))
    ft = (1 - (tp + 1) / (tp + a_fp * fp + b_fn * fn + 1)) ** gamma
    focal = (W * sigmoid_focal_loss(low, V, alpha=alpha, reduction='none')).sum((1, 2, 3)) / W.sum((1, 2, 3)).clamp(min=1)
    return (20 * focal + ft).mean()


def with_feat(agg, f, fixed=False):
    """Fusion inputs from the learned feature integral (adapter/feat_integral.py): f from build(feat=True) or
    encode(feat=True); returns (f with C16/8/4 := Ĉ and a := the aggregator's evidence mass, the aggregator's output)."""
    S = ('16', '8', '4')
    o = agg({s: f[f'K{s}'] for s in S}, f['Kvalid16'], {s: f[f'X{s}'] for s in S}, {s: f[f'B{s}'] for s in S}, f['age'],
            fixed=fixed, phase=f.get('phase'))
    return dict(f, C16=o['C16'].float(), C8=o['C8'].float(), C4=o['C4'].float(), a=o['a'].float()), o


AMODAL = os.environ.get('AMODAL') == '1'   # training-only amodal aux head (findings §10.x 24), λ = 1


class AmHead(nn.Module):
    """Aux amodal logits from the Fusion trunk (x4 + upsampled x16), so the aux shapes the shared trunk but never the residual
    outputs out[l] that the modal decoder reads. Dropped at inference."""
    def __init__(self, w=64):
        super().__init__()
        self.h = nn.Sequential(nn.Conv2d(2 * w, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1))

    def forward(self, o):
        return self.h(torch.cat([o['_x4'], F.interpolate(o['_x16'], scale_factor=4, mode='bilinear', align_corners=False)], 1))


def am_loss(head, o, f):
    """Amodal loss restricted to what history can supply: weight 1 on Ω (hidden at t, visible in the aligned history; Om4) and on
    background outside S, 0 on V and on unrecoverable hidden cells. None if Ω is (almost) empty."""
    Wa = (1 - f['S'].float()) + f['Om4'][:, 0].float()
    if f['Om4'].sum() < 50:
        return None
    return loss_fn(head(o).float(), f['S'], Wa)


def train_step(pred, net, agg, raw, rng, lam=1.0, max_targets=32, am=None):
    """Loss of one training clip. agg None: today's pixel integral image. agg given (FEAT=1): the learned feature integral,
    loss v3.4 + lam · (clean-feature loss on hidden-recoverable cells + evidence-mass BCE) (plan Tasks 5b / 6)."""
    from benchmark.nfo_vos.adapter.feat_integral import feat_loss, mass_loss
    sam = pred.model
    f, _ = build(pred, raw, max_targets=max_targets, rng=rng, feat=agg is not None)
    with torch.autocast('cuda', dtype=AMP):
        aux = 0.0
        if agg is not None:
            f, oa = with_feat(agg, f)
            S = ('16', '8', '4')
            aux = 0.0 if lam == 0 else lam * (feat_loss(oa, {s: f[f'T{s}'] for s in S}, {s: f[f'Om{s}'] for s in S}, f['mu16'])
                         + mass_loss(oa['a'], f['O4']))
        o = net(f, sam=sam)
        loss = loss_fn(decode(sam, o, token=o.get('token')).float(), f['V'], f.get('W')) + aux
        return loss, (am_loss(am, o, f) if am is not None else None)


@torch.no_grad()
def evaluate(pred, net, val, agg=None, am=None):
    """Synthetic val (raw clips, built with the test-time path): J / P / R vs native V; arms box, box+pt, trk-box,
    adapter. Also the input statistics used for the parity check against NFO. agg: the feature-integral path."""
    net.eval()
    rows, a_mass, a_on, am_gap = [], [], [], []
    for raw in val:
        f, meta = build(pred, raw, feat=agg is not None)
        if agg is not None:
            with torch.autocast('cuda', dtype=AMP):
                f, _ = with_feat(agg, f)
            meta['a_mass'] = [float(x) for x in f['a'].sum((1, 2)).cpu()]
            A_ = f['a'].cpu().numpy()                 # learned mass on the visible person (build() leaves it empty in feat mode)
            meta['a_on'] = [float(A_[i][a256(V) > 0.5].mean()) if V.any() else np.nan for i, V in enumerate(meta['Vn'])]
        fb = dict(f, box=f['box_trk'])          # tracker centre, t0 box size: the same GT-free control as on NFO
        with torch.autocast('cuda', dtype=AMP):
            o = net(f, sam=pred.model)
            la = am_loss(am, o, f) if am is not None else None
            outs = {'box': decode(pred.model, f), 'box+pt': decode(pred.model, f, point=True),
                    'trk-box': decode(pred.model, fb), 'adapter': decode(pred.model, o, token=o.get('token'))}
            del o                                    # one adapter output alive at a time (16 GB)
            o = net(dict(f, C16=f['B16'], C8=f['B8'], C4=f['B4'], a=torch.zeros_like(f['a'])), sam=pred.model)
            outs['adapter-noC'] = decode(pred.model, o, token=o.get('token'))   # integral removed (C := B, a := 0)
            if la is not None:                       # kill readout: does the aux head need Ĉ on Ω? gap = L_am(noC) − L_am(Ĉ)
                am_gap.append(float(am_loss(am, o, f) - la))
            del o
            for sc in os.environ.get('SCALE', '').split(','):   # C' = B + s (C − B): is the change signal too weak?
                if sc:
                    o = net(dict(f, **{f'C{k}': f[f'B{k}'] + float(sc) * (f[f'C{k}'] - f[f'B{k}']) for k in ('16', '8', '4')}), sam=pred.model)
                    outs[f'scaled-{sc}'] = decode(pred.model, o, token=o.get('token')); del o
            if os.environ.get('ABL'):                 # which channel carries the integral: occupancy a or appearance C
                for nm, d in (('only-a', dict(C16=f['B16'], C8=f['B8'], C4=f['B4'])), ('only-C', dict(a=torch.zeros_like(f['a'])))):
                    o = net(dict(f, **d), sam=pred.model)
                    outs[nm] = decode(pred.model, o, token=o.get('token')); del o
            if agg is not None and os.environ.get('ORACLE'):   # upper bounds for better mixing: clean-frame features Xt
                for nm, ks in (('oracle-fine', ('8', '4')), ('oracle-all', ('16', '8', '4'))):
                    o = net(dict(f, **{f'C{k}': f[f'T{k}'].to(f[f'C{k}'].dtype) for k in ks}), sam=pred.model)
                    outs[nm] = decode(pred.model, o, token=o.get('token')); del o
        for i, Vn in enumerate(meta['Vn']):
            Vt = torch.from_numpy(Vn).cuda()
            St = torch.from_numpy(raw['S'][meta['ts'][i]]).cuda()
            r = dict(mode=int(meta['mode'][i]), clip=meta['cid'], rho=meta['occ']['rho'], clump=str(meta['occ']['clump']))
            r.update(_hist_cover(raw, meta, i))
            for a, low in outs.items():
                pr = F.interpolate(low[i:i + 1].float(), Vt.shape, mode='bilinear', align_corners=False)[0, 0] > 0
                tp = (pr & Vt).sum().item()
                r[a] = (tp / max((pr | Vt).sum().item(), 1), tp / max(pr.sum().item(), 1), tp / max(Vt.sum().item(), 1),
                        _box_iou(pr.cpu().numpy(), raw['S'][meta['ts'][i]]),   # 4th: mask bbox vs AMODAL person box
                        (pr & St & ~Vt).sum().item() / max((pr & ~Vt).sum().item(), 1))   # 5th: share of FP in the hidden person
            rows.append(r)
        a_mass += meta['a_mass']; a_on += meta['a_on']
    net.train()
    return rows, dict(a_mass=float(np.median(a_mass)), a_on=float(np.nanmean(a_on)), am_gap=am_gap)


def _box_iou(m, S):
    """IoU of the bounding boxes of mask m and amodal person mask S (the synthetic stand-in for NFO's GT boxes)."""
    if not m.any() or not S.any():
        return 0.0
    b = [(ys.min(), xs.min(), ys.max() + 1, xs.max() + 1) for ys, xs in (np.nonzero(m), np.nonzero(S))]
    ih = max(0, min(b[0][2], b[1][2]) - max(b[0][0], b[1][0])); iw = max(0, min(b[0][3], b[1][3]) - max(b[0][1], b[1][1]))
    ar = [(y1 - y0) * (x1 - x0) for y0, x0, y1, x1 in b]
    return float(ih * iw / (ar[0] + ar[1] - ih * iw))


def _hist_cover(raw, meta, i):
    """Per target t (pixel level, native res): vis = |V_t|/|S_t| (visible person share); rec = |S_t ∧ ¬V_t ∧ ∪_k V_k
    warped onto t| / |S_t| (share hidden at t but visible in the aligned history t−n_h..t−1; the Ω of _feat_inputs)."""
    import cv2
    from benchmark.nfo_vos.adapter.fusion_data import hist_warps
    t = meta['ts'][i]; S, V = raw['S'][t], raw['V'][t]; H, W = S.shape
    O = np.zeros((H, W), bool)
    for k, M in zip(*hist_warps(meta['trk'], t, meta['occ']['nh'])):
        O |= cv2.warpAffine(raw['V'][k].astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    n = max(int(S.sum()), 1)
    return dict(vis=float(V.sum() / n), rec=float((O & S & ~V).sum() / n))


def summarise(rows, rng=None):
    rng = rng or np.random.default_rng(0)
    arms = ('box', 'box+pt', 'trk-box', 'adapter', 'adapter-noC')
    clips = sorted({r['clip'] for r in rows})
    per = {a: np.array([np.mean([r[a] for r in rows if r['clip'] == c], 0) for c in clips]) for a in arms}
    s = {a: dict(zip('JPR', per[a].mean(0))) for a in arms}
    for ref in ('box+pt', 'trk-box', 'adapter-noC'):
        dj = per['adapter'][:, 0] - per[ref][:, 0]
        bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(2000)]
        s[f'dJ_{ref}'] = [float(dj.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    groups = {f'mode {m}': [r for r in rows if r['mode'] == m] for m in (0, 2)}
    groups |= {f'rho {lo}-{hi}': [r for r in rows if lo <= r['rho'] < hi] for lo, hi in ((0.0, 0.01), (0.2, 0.4), (0.4, 0.61))}
    s['groups'] = {g: {a: float(np.mean([r[a][0] for r in rs])) for a in arms} | {'n': len(rs)} for g, rs in groups.items() if rs}
    return s


def fmt(d, arms):
    return ' | '.join(f"{a} {d[a]['J']:.3f}/{d[a]['P']:.2f}/{d[a]['R']:.2f}" for a in arms if a in d)


def main(steps=1500, seed=0, eval_every=100, max_targets=32, n_val_fast=12, test=True):
    """FEAT=1 AGG=<stage-0 checkpoint> [LAM=1] [AGG_FREEZE=1: aggregator fixed, no aux loss] [REFINED=<feat_refine state>] [BIGS16=<feat_bigstudent state>] [RUN_DIR, STOP]: the learned feature integral (one encoder pass per frame) instead of the
    pixel integral image; the aggregator is fine-tuned jointly (own AdamW group); checkpoints {'fusion', 'agg'} in
    feat_seed<seed>/."""
    """One clip per step (≤ max_targets frames), built with the test-time path (fusion_data.build). Fast validation every
    `eval_every` steps: 12 synthetic val clips + the seq1 brush SELECTION set, with the parity statistics and the
    integral-image-removal sanity arm; best checkpoint by seq1 adapter J. At the end: full synthetic val (30 clips) and ONE
    evaluation of the selected checkpoint on the seq2–4 handAv TEST set.
    Printed J/P/R triples; 'GAP' = (adapter − trk-box) on synthetic vs on seq1: the domain gap in the quantity that matters."""
    from benchmark.nfo_vos.adapter import fusion_nfo
    torch.manual_seed(seed); np.random.seed(seed)
    pred = image_predictor()
    sam = pred.model.eval()
    for p_ in sam.parameters():
        p_.requires_grad_(False)
    net = Fusion().cuda()
    net.init_fallback(sam)
    agg, lam = None, float(os.environ.get('LAM', 1.0))
    if os.environ.get('FEAT') == '1':
        from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral
        agg = FeatIntegral(sam.no_mem_embed).cuda()
        sd0 = torch.load(os.environ.get('AGG') or 'results/benchmark/rr/feat_integral/stage0c_seed0.pt', map_location='cuda')
        agg.load_compat(sd0.get('agg', sd0))                       # an adapter checkpoint {'fusion', 'agg'} also works
        if os.environ.get('REFINED'):                              # distilled aggregator + s16 refiner (feat_refine)
            from benchmark.nfo_vos.adapter.feat_integral import Refined
            sdr = torch.load(os.environ['REFINED'], map_location='cuda')
            agg = Refined(agg, full='g8.0.weight' in sdr).cuda(); agg.load_state_dict(sdr)
        if os.environ.get('BIGS16'):                               # + big s16 student (tools/feat_bigstudent; needs REFINED)
            from benchmark.nfo_vos.adapter.feat_integral import BigS16
            agg = BigS16(agg).cuda(); agg.load_state_dict(torch.load(os.environ['BIGS16'], map_location='cuda'))
        if os.environ.get('AGG_FREEZE') == '1':                    # fixed history signal, as for the pixel integral
            agg.requires_grad_(False); agg.eval(); lam = 0.0
    n_par = sum(p_.numel() for p_ in net.parameters()) + (sum(p_.numel() for p_ in agg.parameters()) if agg else 0)
    am = AmHead().cuda() if AMODAL else None
    groups = [{'params': list(net.parameters()) + (list(am.parameters()) if am is not None else [])}] + ([{'params': list(agg.parameters())}] if agg and any(p_.requires_grad for p_ in agg.parameters()) else [])
    opt = torch.optim.AdamW(groups, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16)
    dl = iter(torch.utils.data.DataLoader(Clips('train', seed), batch_size=None, num_workers=6, prefetch_factor=2,
                                          collate_fn=_ident, persistent_workers=True))
    val_all = list(torch.utils.data.DataLoader(Clips('val'), batch_size=None, num_workers=6, collate_fn=_ident))
    val = val_all[:n_val_fast]
    sel = fusion_nfo.items('select', pred, feat=agg is not None)
    out = os.environ.get('RUN_DIR') or f'{OUT}/{"feat" if agg else "token"}_seed{seed}'
    stop = int(os.environ.get('STOP', steps))                     # end early at a pre-fixed step, schedule still over `steps`
    save = lambda p: torch.save(({'fusion': net.state_dict(), 'agg': agg.state_dict()} | ({'am': am.state_dict()} if am is not None else {})) if agg else net.state_dict(), p)
    os.makedirs(out, exist_ok=True)
    print(f'params {n_par}; val clips {len(val)} fast / {len(val_all)} final; seq1 select frames {len(sel)}', flush=True)
    rng, log, best, t0 = np.random.default_rng(seed), [], (-1.0, None), time.time()

    def validate(it, rows_inp=None):
        torch.cuda.empty_cache()                     # FEAT: the train graph's freed blocks fragment the 16 GB card (OOM 2026-11-19)
        rows, inp = evaluate(pred, net, val, agg, am)
        if inp['am_gap']:
            g = np.array(inp['am_gap']); print(f'AMGAP it {it}: L_am(noC) − L_am(C) on Ω {g.mean():+.4f} (pos {np.mean(g > 0):.2f}, {len(g)} clips)', flush=True)
        s = summarise(rows); ss = fusion_nfo.evaluate(pred, net, sel, agg)
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
        loss, la = train_step(pred, net, agg, next(dl), rng, lam, max_targets, am)
        if la is not None:     # λ_am = 1: same normalised loss_fn as the modal term (a step-1 grad-norm match is 0: out[l] is zero-init)
            if it % 25 == 0:
                print(f'it {it} modal {loss.item():.4f} amodal {la.item():.4f}', flush=True)
            loss = loss + la
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 25 == 0:
            print(f'it {it} loss {loss.item():.4f} {(time.time() - t0) / it:.2f} s/it', flush=True)
        if it % eval_every == 0 or it == steps:
            s = validate(it)
            log.append(s)
            save(f'{out}/fusion_it{it}.pt')
            if s['sel']['adapter']['J'] > best[0]:
                best = (s['sel']['adapter']['J'], it)
            json.dump(log, open(f'{out}/log.json', 'w'), indent=1, default=float)
        if it == stop:                                             # diagnostic runs: no selection, no test set
            return
    sd = torch.load(f'{out}/fusion_it{best[1]}.pt')
    if agg:
        net.load_state_dict(sd['fusion']); agg.load_state_dict(sd['agg'])
    else:
        net.load_state_dict(sd)
    rows, inp = evaluate(pred, net, val_all if test else val, agg)
    full = summarise(rows)
    print(f"FINAL synthetic val ({len(val_all)} clips, checkpoint it {best[1]}): " + fmt(full, ('box+pt', 'trk-box', 'adapter')), flush=True)
    if not test:                                     # smoke runs never touch the test set
        return
    test = fusion_nfo.evaluate(pred, net, fusion_nfo.items('test', pred, feat=agg is not None), agg)
    pf = test.pop('per_frame'); d = np.array(pf['adapter']) - np.array(pf['trk-box'])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    log.append(dict(final_val=full, test=test, test_dJ_trk=[float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    selected_it=best[1]))
    json.dump(log, open(f'{out}/log.json', 'w'), indent=1, default=float)
    print(f"TEST (seq2-4 handAv, it {best[1]} chosen on seq1): " + fmt(test, ('box', 'box+pt', 'trk-box', 'adapter', 'adapter-noC'))
          + f" | ΔJ adapter − trk-box {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]", flush=True)

if __name__ == '__main__':
    main(*(int(x) for x in sys.argv[1:]))
