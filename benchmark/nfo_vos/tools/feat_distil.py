"""Distillation pilot for the synthetic → NFO domain gap of the learned feature integral (findings §10.x items 11–12).

The pixel integral enc(C) transfers from synthetic to NFO; Ĉ does not, and the only part of Ĉ learned on synthetic data is
the mixing weights α (feat_integral.py: ℓ_k = h(P F_k, P F_k − P F_t, P F_k − P B, age)). Teacher = the pixel route on real,
UNLABELLED NFO frames: fine-tune the aggregator (init: the it700 one) so that its s16 output matches enc(C) in centred
cosine, on seq1 crossings s4–s7 (tracker-centred 40×40-cell crops, no GT used). Inference stays one-pass.
Readout on the HELD-OUT crossings s0–s3: the synthetic → NFO transfer probe (feat_probe_transfer: linear reader fitted on
30 synthetic train clips, scored on NFO GT-box cells), gain of distilled Ĉ vs it700 Ĉ vs pixel. Pass: the Ĉ − pixel gap
(−0.020 on all 8 crossings) closes. Also printed: held-out teacher cosine before / after (generalisation of the fit).

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_distil
"""
import copy
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips
from benchmark.nfo_vos.adapter.feat_integral import ccos
from benchmark.nfo_vos.adapter.inference import masks_from_arrays
from benchmark.nfo_vos.adapter.tracker import gt_box
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_probe import cells as syn_cells, fit
from benchmark.nfo_vos.tools.feat_probe_nfo import NETS, labels

R = 'results/benchmark/rr/feat_integral'
OUT = f'{R}/distil'
TRAIN, TEST = range(4, 8), range(4)                              # seq1 crossing indices
W = 40                                                           # crop (cells) around the tracker box centre


def frames(t):
    w0, w1 = t['warmup']
    bg = np.median(np.stack([TR.native(i, t['seq']) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    return [TR.native(k, t['seq']) for k in t['frames']], bg, gt_box(t['seq'], t['frames'][0])


@torch.no_grad()
def collect(pred, feat, pix, t):
    """Per history frame: tracker-centred crops of K16 (k keys), valid, F_t16, B16, the shifts and the teacher enc(C)16."""
    Xn, bg, g = frames(t); d = {}

    def hk(j, f):
        cy, cx = (f['box_trk'][0].view(2, 2).mean(0) / 16).long().tolist()[::-1]
        y0, x0 = min(max(cy - W // 2, 0), 64 - W), min(max(cx - W // 2, 0), 64 - W)
        cr = lambda x: x[..., y0:y0 + W, x0:x0 + W].half().cpu()
        K, v, dx = f['_agg']; K = K['16']
        d[j] = dict(K=cr(K[0]), v=cr(v[0, :, 0]), Ft=cr(f['X16'][0]), B=cr(f['B16'][0]), dx=dx[0].cpu(), crop=(y0, x0))
    masks_from_arrays(pred, feat, Xn, bg, g, probe=hk)

    def hp(j, f):
        if j in d:
            y0, x0 = d[j]['crop']; d[j]['T'] = f['C16'][0][..., y0:y0 + W, x0:x0 + W].half().cpu()
    masks_from_arrays(pred, pix, Xn, bg, g, probe=hp)
    return [x for x in d.values() if 'T' in x]


def chat(agg, b):
    """Ĉ16 on a list of crops (the learned path of FeatIntegral.forward at stride 16, nbr = 0)."""
    K = torch.stack([x['K'] for x in b]).cuda().float(); v = torch.stack([x['v'] for x in b]).cuda().bool()
    Ft = torch.stack([x['Ft'] for x in b]).cuda().float(); B = torch.stack([x['B'] for x in b]).cuda().float()
    dx = torch.stack([x['dx'] for x in b]).cuda()
    age = torch.arange(K.shape[1], 0, -1, device='cuda')[None].expand(len(b), -1)
    with torch.autocast('cuda', dtype=AMP):
        lk, ln = agg.logits_learned(K, Ft, B, age, phase=dx / 16 - torch.round(dx / 16))
    al = torch.softmax(torch.cat([lk.masked_fill(~v, float('-inf')), ln], 1), 1)
    return (al[:, :-1, None] * K).sum(1) + al[:, -1:] * B


def tcos(agg, data, mu):
    with torch.no_grad():
        return float(np.mean([ccos(chat(agg, data[i:i + 16]), torch.stack([x['T'] for x in data[i:i + 16]]).cuda().float(), mu).mean().item()
                              for i in range(0, len(data), 16)]))


@torch.no_grad()
def nfo_cells(pred, net, t):
    """Ĉ16 cells of one crossing at feat_probe_nfo's GT-box label cells (same frames: history, labelled)."""
    Xn, bg, g = frames(t); rows = {}
    lab = {j: labels(b) for j, k in enumerate(t['frames']) if (b := gt_box(t['seq'], k)) is not None}

    def hk(j, f):
        if lab.get(j) is not None:
            rows[j] = f['C16'][0][:, torch.from_numpy(lab[j][0]).cuda()].T.half().cpu()
    masks_from_arrays(pred, net, Xn, bg, g, probe=hk)
    return torch.cat([rows[j] for j in sorted(rows)])


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    pred = image_predictor()
    feat, pix = load_net(pred, NETS['learned']), load_net(pred, NETS['pixel'])
    trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    tr = sum([collect(pred, feat, pix, trials[i]) for i in TRAIN], []); te = sum([collect(pred, feat, pix, trials[i]) for i in TEST], [])
    print(f'distil frames train {len(tr)} (s4–s7) / held-out {len(te)} (s0–s3)', flush=True)
    agg = copy.deepcopy(feat.agg).train(); mu = agg.mu16
    before = (tcos(agg, tr, mu), tcos(agg, te, mu))
    opt = torch.optim.AdamW([p for p in agg.parameters() if p.requires_grad], 1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for it in range(1, 401):
        b = [tr[i] for i in rng.choice(len(tr), 8, replace=False)]
        loss = 1 - ccos(chat(agg, b), torch.stack([x['T'] for x in b]).cuda().float(), mu).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f'it {it} loss {loss.item():.4f}', flush=True)
    agg.eval()
    print(f'teacher cos train {before[0]:.3f} → {tcos(agg, tr, mu):.3f}; held-out {before[1]:.3f} → {tcos(agg, te, mu):.3f}', flush=True)
    torch.save(agg.state_dict(), f'{OUT}/agg_distil_s4-7.pt')
    del tr, te
    # readout: transfer probe on held-out crossings
    feat.agg = agg
    nfo = torch.load(f'{R}/probe_nfo/cells_seq1.pt', weights_only=False)
    for i in TEST:
        c = nfo_cells(pred, feat, trials[i]); assert len(c) == len(nfo[i][1]), (i, len(c), len(nfo[i][1]))
        nfo[i][0]['distil'] = c
    syn, _ = torch.load(f'{R}/probe/cells_n60.pt', weights_only=False); syn = syn[:30]
    new = syn_cells(pred, {'learned': agg, 'nbr1': agg},
                    torch.utils.data.DataLoader(Clips('train', seed=7), batch_size=None, num_workers=6, collate_fn=_ident), 30)
    for a, b in zip(syn, new):                                    # same clips? (deterministic generation)
        assert len(a[1]) == len(b[1]) and torch.equal(a[0]['none'], b[0]['none']), 'synthetic clips differ from the cache'
        a[0]['distil'] = b[0]['learned']
    ytr = np.concatenate([y for _, y, _ in syn]); sc = {}
    for s in ('none', 'pixel', 'learned', 'distil'):
        mu_, sd, w, b_ = fit(torch.cat([c[s] for c, _, _ in syn]), ytr)
        per = []
        for i in TEST:
            c, y, oc = nfo[i]; Z = c['none'] if s == 'none' else torch.cat([c['none'], c[s]], 1)
            p = (((Z.cuda().float() - mu_) / sd) @ w + b_).cpu().numpy(); sel = oc | ~y
            per.append((roc_auc_score(y, p), roc_auc_score(y[sel], p[sel])))
        sc[s] = np.array(per)
    res = {}
    for s in ('pixel', 'learned', 'distil'):
        for j, nm in enumerate(('all', 'occl')):
            d = sc[s][:, j] - sc['none'][:, j]; res[f'{s}_{nm}'] = d.tolist()
            print(f'transfer gain {s:8s} {nm}: {d.mean():+.4f}  per held-out crossing ' + ' '.join(f'{x:+.3f}' for x in d), flush=True)
    json.dump(res, open(f'{OUT}/transfer_heldout.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
