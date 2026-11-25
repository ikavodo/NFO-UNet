"""Distillation + refiner: the fix the fixed-rule test points to (findings §10.x items 13–14: re-weighting the keys alone
closes ≈ 25% of the synthetic → NFO transfer gap; a fixed α transfers worse, so the limit is that Ĉ is a convex mix of
features, while the pixel route re-encodes the averaged image).

Student: the it700 aggregator (α) plus a small residual refiner at s16, Ĉ' = Ĉ + g([Ĉ − μ, B − μ]) (1×1 768→128, 3×3,
1×1 → 256, last layer zero-init: Ĉ' = Ĉ at step 0; ≈ 0.3 M parameters, one-pass, history and empty scene only, no F_t,
so a probe gain cannot come from re-reading the current frame). Teacher: the pixel route's enc(C)16 on the same frame.
Data, no GT: seq1 crossings s4–s7 (real, unlabelled) and 20 synthetic train clips (seed 11, disjoint from the probe's
seed-7 clips), tracker-centred 40×40-cell crops; loss 1 − centred cosine, NFO and synthetic batches alternated.
Readout = feat_distil's: transfer gain over [F_t, B] on held-out seq1 s0–s3, reader fitted on 30 synthetic clips.

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_refine
"""
import copy
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.feat_integral import Refined, ccos
from benchmark.nfo_vos.adapter.feat_stage0 import _agg_in
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips, build
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_distil import OUT, R, TEST, TRAIN, W, collect, nfo_cells
from benchmark.nfo_vos.tools.feat_probe import cells as syn_cells, fit
from benchmark.nfo_vos.tools.feat_probe_nfo import NETS


def crops_syn(pred, n=20):
    """Synthetic crops in collect()'s format (+ 'ph'), grouped per clip (the key count varies between clips)."""
    out = []
    dl = torch.utils.data.DataLoader(Clips('train', seed=11), batch_size=None, num_workers=6, collate_fn=_ident)
    for k, raw in enumerate(dl):
        if k == n:
            break
        with torch.no_grad():
            f, _ = build(pred, raw, max_targets=8, rng=np.random.default_rng(100 + k), feat=True, with_pixel=True)
        K, v, Ft, B, age = _agg_in(f); cl = []
        for i in range(len(f['trk'])):
            cx, cy = (f['trk'][i] / 16).long().tolist()
            y0, x0 = min(max(cy - W // 2, 0), 64 - W), min(max(cx - W // 2, 0), 64 - W)
            cr = lambda x: x[..., y0:y0 + W, x0:x0 + W].half().cpu()
            cl.append(dict(K=cr(K['16'][i]), v=cr(v[i, :, 0]).bool(), Ft=cr(Ft['16'][i]), B=cr(B['16'][i]),
                           ph=f['phase'][i].float().cpu(), T=cr(f['P16'][i])))
        out.append(cl); del f
    return out


def student(m, b):
    K = torch.stack([x['K'] for x in b]).cuda().float(); v = torch.stack([x['v'] for x in b]).cuda().bool()
    Ft = torch.stack([x['Ft'] for x in b]).cuda().float(); B = torch.stack([x['B'] for x in b]).cuda().float()
    ph = torch.stack([x['ph'] if 'ph' in x else x['dx'] / 16 - torch.round(x['dx'] / 16) for x in b]).cuda()
    age = torch.arange(K.shape[1], 0, -1, device='cuda')[None].expand(len(b), -1)
    with torch.autocast('cuda', dtype=AMP):
        lk, ln = m.agg.logits_learned(K, Ft, B, age, phase=ph)
    al = torch.softmax(torch.cat([lk.masked_fill(~v, float('-inf')), ln], 1), 1)
    return m.refine((al[:, :-1, None] * K).sum(1) + al[:, -1:] * B, B)


def tcos(m, data):
    mu = m.agg.mu16
    with torch.no_grad():
        return float(np.mean([ccos(student(m, data[i:i + 16]), torch.stack([x['T'] for x in data[i:i + 16]]).cuda().float(), mu).mean().item()
                              for i in range(0, len(data), 16)]))


def main():
    torch.manual_seed(0)
    pred = image_predictor()
    feat, pix = load_net(pred, NETS['learned']), load_net(pred, NETS['pixel'])
    trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    tr = sum([collect(pred, feat, pix, trials[i]) for i in TRAIN], []); te = sum([collect(pred, feat, pix, trials[i]) for i in TEST], [])
    syn = crops_syn(pred)
    print(f'NFO train {len(tr)} / held-out {len(te)} frames; synthetic {sum(map(len, syn))} targets in {len(syn)} clips', flush=True)
    m = Refined(copy.deepcopy(feat.agg)).cuda().train(); mu = m.agg.mu16
    print(f'teacher cos before: NFO train {tcos(m, tr):.3f} held-out {tcos(m, te):.3f}', flush=True)
    opt = torch.optim.AdamW([{'params': [p for p in m.agg.parameters() if p.requires_grad], 'lr': 1e-4},
                             {'params': m.g.parameters(), 'lr': 3e-4}], weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for it in range(1, 1501):
        b = [tr[i] for i in rng.choice(len(tr), 8, replace=False)] if it % 2 else syn[rng.integers(len(syn))]
        loss = 1 - ccos(student(m, b), torch.stack([x['T'] for x in b]).cuda().float(), mu).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0:
            print(f'it {it} loss {loss.item():.4f}  held-out NFO teacher cos {tcos(m, te):.3f}', flush=True)
    m.eval()
    print(f'teacher cos after: NFO train {tcos(m, tr):.3f} held-out {tcos(m, te):.3f}', flush=True)
    torch.save(m.state_dict(), f'{OUT}/refined_s4-7.pt')
    del tr, te, syn
    feat.agg = m
    nfo = torch.load(f'{R}/probe_nfo/cells_seq1.pt', weights_only=False)
    with torch.no_grad():
        for i in TEST:
            c = nfo_cells(pred, feat, trials[i]); assert len(c) == len(nfo[i][1])
            nfo[i][0]['refined'] = c
        sy, _ = torch.load(f'{R}/probe/cells_n60.pt', weights_only=False); sy = sy[:30]
        new = syn_cells(pred, {'learned': m, 'nbr1': m},
                        torch.utils.data.DataLoader(Clips('train', seed=7), batch_size=None, num_workers=6, collate_fn=_ident), 30)
    for a, b in zip(sy, new):
        assert len(a[1]) == len(b[1]) and torch.equal(a[0]['none'], b[0]['none']), 'synthetic clips differ from the cache'
        a[0]['refined'] = b[0]['learned']
    ytr = np.concatenate([y for _, y, _ in sy]); sc = {}
    for s in ('none', 'refined'):
        mu_, sd, w, b_ = fit(torch.cat([c[s] for c, _, _ in sy]), ytr)
        per = []
        for i in TEST:
            c, y, oc = nfo[i]; Z = c['none'] if s == 'none' else torch.cat([c['none'], c[s]], 1)
            p = (((Z.cuda().float() - mu_) / sd) @ w + b_).cpu().numpy(); sel = oc | ~y
            per.append((roc_auc_score(y, p), roc_auc_score(y[sel], p[sel])))
        sc[s] = np.array(per)
    res = json.load(open(f'{OUT}/transfer_heldout.json'))
    for j, nm in enumerate(('all', 'occl')):
        d = sc['refined'][:, j] - sc['none'][:, j]; res[f'refined_{nm}'] = d.tolist()
        print(f'transfer gain refined  {nm}: {d.mean():+.4f}  per held-out crossing ' + ' '.join(f'{x:+.3f}' for x in d)
              + '  | ' + ' '.join(f'{s} {np.mean(res[f"{s}_{nm}"]):+.4f}' for s in ('pixel', 'learned', 'distil', 'fixed')), flush=True)
    json.dump(res, open(f'{OUT}/transfer_heldout.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
