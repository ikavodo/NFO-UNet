"""Full-input distillation (findings §10.x item 17): as feat_refine (item 15), but the student imitates the pixel route's
WHOLE adapter input: Ĉ at strides 16 / 8 / 4 (refiners g, g8, g4; α shared, computed at s16 and upsampled as in
FeatIntegral) and the evidence mass a (the aggregator's mass head, BCE to the pixel mass Σ w G). Same data and budget as
item 15 (seq1 s4–s7 unlabelled + 20 synthetic clips seed 11, 1500 steps, init it700), so the only change is the target.
Printed: held-out (s0–s3, every 3rd frame) teacher cosine per stride and mass L1, before / after.

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_refine_full
"""
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.feat_integral import Refined, ccos
from benchmark.nfo_vos.adapter.feat_stage0 import _agg_in
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips, build
from benchmark.nfo_vos.adapter.inference import masks_from_arrays
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_distil import OUT, TEST, TRAIN, W, frames
from benchmark.nfo_vos.tools.feat_probe_nfo import NETS

S = ('16', '8', '4')
SC = {'16': 1, '8': 2, '4': 4}


def crop(x, s, y0, x0):
    r = SC[s]
    return x[..., y0 * r:(y0 + W) * r, x0 * r:(x0 + W) * r].half().cpu()


def window(cx, cy):
    return min(max(int(cy) - W // 2, 0), 64 - W), min(max(int(cx) - W // 2, 0), 64 - W)


@torch.no_grad()
def collect(pred, feat, pix, t, every=1):
    Xn, bg, g = frames(t); d = {}

    def hk(j, f):
        if j % every:
            return
        cx, cy = (f['box_trk'][0].view(2, 2).mean(0) / 16).tolist()
        y0, x0 = window(cx, cy); K, v, dx = f['_agg']
        e = dict(v=crop(v[0, :, 0], '16', y0, x0).bool(), Ft=crop(f['X16'][0], '16', y0, x0),
                 ph=(dx[0] / 16 - torch.round(dx[0] / 16)).float().cpu(), w=(y0, x0))
        for s in S:
            e[f'K{s}'] = crop(K[s][0], s, y0, x0); e[f'B{s}'] = crop(f[f'B{s}'][0], s, y0, x0)
        d[j] = e
    masks_from_arrays(pred, feat, Xn, bg, g, probe=hk)

    def hp(j, f):
        if j in d:
            y0, x0 = d[j]['w']
            for s in S:
                d[j][f'T{s}'] = crop(f[f'C{s}'][0], s, y0, x0)
            d[j]['A'] = crop(f['a'][0], '4', y0, x0)
    masks_from_arrays(pred, pix, Xn, bg, g, probe=hp)
    return [x for x in d.values() if 'A' in x]


def crops_syn(pred, n=20):
    out = []
    dl = torch.utils.data.DataLoader(Clips('train', seed=11), batch_size=None, num_workers=6, collate_fn=_ident)
    for k, raw in enumerate(dl):
        if k == n:
            break
        with torch.no_grad():
            f, _ = build(pred, raw, max_targets=8, rng=np.random.default_rng(100 + k), feat=True, with_pixel=True)
        K, v, Ft, B, age = _agg_in(f)
        ap = f['a_pix']; ap = ap.reshape(ap.shape[0], *ap.shape[-2:]); assert ap.shape[-1] == 256, ap.shape
        cl = []
        for i in range(len(f['trk'])):
            y0, x0 = window(*(f['trk'][i] / 16).tolist())
            e = dict(v=crop(v[i, :, 0], '16', y0, x0).bool(), Ft=crop(Ft['16'][i], '16', y0, x0), ph=f['phase'][i].float().cpu(),
                     A=crop(ap[i], '4', y0, x0))
            for s in S:
                e[f'K{s}'] = crop(K[s][i], s, y0, x0); e[f'B{s}'] = crop(B[s][i], s, y0, x0); e[f'T{s}'] = crop(f[f'P{s}'][i], s, y0, x0)
            cl.append(e)
        out.append(cl); del f
    return out


def student(m, b):
    st = lambda key: torch.stack([x[key] for x in b]).cuda()
    v, Ft, ph = st('v'), st('Ft').float(), st('ph')
    K = {s: st(f'K{s}').float() for s in S}; B = {s: st(f'B{s}').float() for s in S}
    age = torch.arange(K['16'].shape[1], 0, -1, device='cuda')[None].expand(len(b), -1)
    with torch.autocast('cuda', dtype=AMP):
        lk, ln = m.agg.logits_learned(K['16'], Ft, B['16'], age, phase=ph)
    al = torch.softmax(torch.cat([lk.masked_fill(~v, float('-inf')), ln], 1), 1)
    C = {}
    for s in S:
        a_s = al if s == '16' else F.interpolate(al, size=K[s].shape[-2:], mode='bilinear', align_corners=False)
        C[s] = m.refine_s(s, (a_s[:, :-1, None] * K[s]).sum(1) + a_s[:, -1:] * B[s], B[s])
    mk = m.agg._mk.masked_fill(~v, float('-inf'))
    ml = torch.logsumexp(mk, 1, keepdim=True) + m.agg.mbias.float()
    a = torch.sigmoid(F.interpolate(ml.nan_to_num(neginf=-1e4), size=K['4'].shape[-2:], mode='bilinear', align_corners=False))[:, 0]
    return C, a


def losses(m, b):
    C, a = student(m, b); mu = m.agg.mu16
    cs = {s: ccos(C[s], torch.stack([x[f'T{s}'] for x in b]).cuda().float(), mu if s == '16' else None).mean() for s in S}
    A = torch.stack([x['A'] for x in b]).cuda().float().clamp(0, 1)
    return cs, F.binary_cross_entropy(a.clamp(1e-4, 1 - 1e-4), A), (a - A).abs().mean()


def report(m, data, tag):
    with torch.no_grad():
        r = [losses(m, data[i:i + 8]) for i in range(0, len(data), 8)]
    print(f'{tag}: teacher cos ' + ' '.join(f's{s} {np.mean([x[0][s].item() for x in r]):.3f}' for s in S)
          + f'  mass L1 {np.mean([x[2].item() for x in r]):.3f}', flush=True)


def main():
    torch.manual_seed(0)
    pred = image_predictor()
    feat, pix = load_net(pred, NETS['learned']), load_net(pred, NETS['pixel'])
    trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    tr = sum([collect(pred, feat, pix, trials[i]) for i in TRAIN], [])
    te = sum([collect(pred, feat, pix, trials[i], every=3) for i in TEST], [])
    syn = crops_syn(pred)
    print(f'NFO train {len(tr)} / held-out {len(te)} frames; synthetic {sum(map(len, syn))} targets', flush=True)
    m = Refined(copy.deepcopy(feat.agg), full=True).cuda().train()
    report(m, te, 'held-out before')
    opt = torch.optim.AdamW([{'params': [p for p in m.agg.parameters() if p.requires_grad], 'lr': 1e-4},
                             {'params': [p for n, p in m.named_parameters() if n.startswith('g')], 'lr': 3e-4}], weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for it in range(1, 1501):
        b = [tr[i] for i in rng.choice(len(tr), 8, replace=False)] if it % 2 else syn[rng.integers(len(syn))]
        cs, bce, _ = losses(m, b)
        loss = sum(1 - c for c in cs.values()) / 3 + bce
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            m.eval(); report(m, te, f'it {it} held-out'); m.train()
    m.eval(); report(m, tr, 'train after'); report(m, te, 'held-out after')
    torch.save(m.state_dict(), f'{OUT}/refinedfull_s4-7.pt')


if __name__ == '__main__':
    main()
