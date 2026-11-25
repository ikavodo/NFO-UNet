"""Linear probe (Alain & Bengio, ICLR-W 2017, arXiv 1610.01644): how much person information does each history channel carry,
independent of any trained adapter? Frozen s16 features; a fresh logistic regression per input set, trained on synthetic
TRAIN clips, tested on the 30 val clips. Cells: s16 cells inside the amodal person box expanded by 50% per side. Label:
person (area share of S_t ≥ 0.5) vs background (share 0); mixed cells dropped. Hidden-person cells: person and visible
share < 0.1, the cells only history can explain.

Input sets (256 ch each, standardised on train): none = [F_t, B]; + learned Ĉ (it700 aggregator) / + nbr1 Ĉ / + pixel
enc(C) / + oracle T (features of the occluder-free frame). Reported: AUC person-vs-background on all cells, and on
hidden-person-vs-background; information gain = AUC(set) − AUC(none), with a 2000× val-clip bootstrap.
An affine rescaling of any input leaves the probe unchanged, so this measures content, not scale.

    PYTHONPATH=. python -m benchmark.nfo_vos.tools.feat_probe [n_train_clips] [linear|mlp]
Cells are cached in OUT/cells_n<n>.pt, so the MLP reader (1 hidden layer, 256) runs on identical cells: it separates
'information absent' from 'present but not linearly readable' if a linear gain comes out small.
"""
import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos.adapter.feat_stage0 import _agg_in
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips, a256, build
from benchmark.nfo_vos.eval.dump_arms import load_net

D = 'results/benchmark/rr/layers_fusion_adapter'
AGGS = {'learned': f'{D}/feat_seed0/feat_seed0/fusion_it700.pt', 'nbr1': f'{D}/feat_nbr/fusion_it700_nbr1.pt'}
SETS = ('none', 'learned', 'nbr1', 'pixel', 'oracle')
OUT = 'results/benchmark/rr/feat_integral/probe'


def s16(m):
    """Native mask -> (64, 64) area share."""
    return a256(m).reshape(64, 4, 64, 4).mean((1, 3))


@torch.no_grad()
def cells(pred, aggs, loader, n):
    """Per clip: dict(set -> (cells, 256·k) float16 CPU), labels y, hidden flag h."""
    out = []
    for k, raw in enumerate(loader):
        if k == n:
            break
        f, meta = build(pred, raw, max_targets=16, rng=np.random.default_rng(k), feat=True, with_pixel=True)
        with torch.autocast('cuda', dtype=AMP):
            Ch = {nm: a(*_agg_in(f), phase=f['phase'])['C16'].float() for nm, a in aggs.items()}
        src = {'learned': Ch['learned'], 'nbr1': Ch['nbr1'], 'pixel': f['P16'], 'oracle': f['T16']}
        rows = {s: [] for s in SETS}; ys, hs = [], []
        for i, t in enumerate(meta['ts']):
            S, V = s16(raw['S'][t]), s16(raw['V'][t])
            yy, xx = np.nonzero(S > 0.5)
            if len(yy) < 4:
                continue
            h, w = yy.max() - yy.min() + 1, xx.max() - xx.min() + 1
            reg = np.zeros_like(S, bool)
            reg[max(0, yy.min() - h // 2):yy.max() + h // 2 + 1, max(0, xx.min() - w // 2):xx.max() + w // 2 + 1] = True
            keep = reg & ((S >= 0.5) | (S == 0))
            m = torch.from_numpy(keep).cuda()
            base = torch.cat([f['X16'][i], f['B16'][i]], 0)[:, m].T
            rows['none'].append(base.half().cpu())
            for s in SETS[1:]:
                rows[s].append(torch.cat([base, src[s][i][:, m].T.float()], 1).half().cpu())
            ys.append(S[keep] >= 0.5); hs.append((S[keep] >= 0.5) & (V[keep] < 0.1))
        if ys:
            out.append(({s: torch.cat(v) for s, v in rows.items()}, np.concatenate(ys), np.concatenate(hs)))
        print(f'clip {k}: {sum(len(y) for _, y, _ in out[-1:])} cells', flush=True)
    return out


def fit(X, y, l2=1e-3, it=200):
    """Standardised L2 logistic regression, full batch LBFGS on the GPU. Returns (mu, sd, w, b)."""
    X = X.cuda().float(); y = torch.as_tensor(y, dtype=torch.float32, device='cuda')
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4); Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], device='cuda', requires_grad=True); b = torch.zeros(1, device='cuda', requires_grad=True)
    pw = (1 - y).sum() / y.sum().clamp(min=1)                   # class balance
    opt = torch.optim.LBFGS([w, b], max_iter=it, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y, pos_weight=pw) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mu, sd, w.detach(), b.detach()


def fit_mlp(X, y, wd=1e-3, steps=600, bs=8192):
    """Standardised 1-hidden-layer MLP (256, ReLU), class-balanced BCE, AdamW minibatches. Returns a scoring function."""
    X = X.cuda().float(); y = torch.as_tensor(y, dtype=torch.float32, device='cuda')
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd); pw = (1 - y).sum() / y.sum().clamp(min=1)
    g = torch.Generator(device='cuda').manual_seed(0)
    for _ in range(steps):
        i = torch.randint(0, len(X), (bs,), device='cuda', generator=g)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net((X[i] - mu) / sd)[:, 0], y[i], pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Z: net((Z.cuda().float() - mu) / sd)[:, 0].detach()


def main(n_train=60, reader='linear'):
    import os
    os.makedirs(OUT, exist_ok=True)
    cache = f'{OUT}/cells_n{n_train}.pt'
    if os.path.exists(cache):
        tr, va = torch.load(cache, weights_only=False)
    else:
        pred = image_predictor()
        aggs = {nm: load_net(pred, c).agg.eval() for nm, c in AGGS.items()}
        tr = cells(pred, aggs, torch.utils.data.DataLoader(Clips('train', seed=7), batch_size=None, num_workers=6, collate_fn=_ident), int(n_train))
        va = cells(pred, aggs, torch.utils.data.DataLoader(Clips('val', n_val=30), batch_size=None, num_workers=6, collate_fn=_ident), 30)
        torch.save((tr, va), cache)
    ytr = np.concatenate([y for _, y, _ in tr])
    print(f'train cells {len(ytr)} (person {ytr.mean():.2f}); val cells {sum(len(y) for _, y, _ in va)}', flush=True)
    score = {}                                                  # set -> per val clip (all-AUC, hidden-AUC)
    for s in SETS:
        if reader == 'mlp':
            sc = fit_mlp(torch.cat([c[s] for c, _, _ in tr]), ytr)
        else:
            mu, sd, w, b = fit(torch.cat([c[s] for c, _, _ in tr]), ytr)
            sc = lambda Z: ((Z.cuda().float() - mu) / sd) @ w + b
        per = []
        for c, y, h in va:
            p = sc(c[s]).cpu().numpy()
            sel = h | ~y                                        # hidden person vs background
            per.append((roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
                        roc_auc_score(y[sel], p[sel]) if h.any() and (~y).any() else np.nan))
        score[s] = np.array(per)
    rng = np.random.default_rng(0)
    res = {}
    for s in SETS:
        line = f'{s:8s}'
        for j, nm in enumerate(('all', 'hidden')):
            a = score[s][:, j]; d = a - score['none'][:, j]; ok = ~np.isnan(d)
            bs = [np.nanmean(d[ok][rng.integers(0, ok.sum(), ok.sum())]) for _ in range(2000)]
            res[f'{s}_{nm}'] = [float(np.nanmean(a)), float(np.nanmean(d)), *map(float, np.percentile(bs, [2.5, 97.5]))]
            line += f'  {nm} AUC {np.nanmean(a):.3f} (gain {np.nanmean(d):+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], {ok.sum()} clips)'
        print(line, flush=True)
    for j, nm in enumerate(('all', 'hidden')):              # the decisive comparison: learned Ĉ vs pixel, paired per clip
        d = score['learned'][:, j] - score['pixel'][:, j]; ok = ~np.isnan(d)
        bs = [d[ok][rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(2000)]
        print(f'learned − pixel {nm}: {d[ok].mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]', flush=True)
    np.save(f'{OUT}/probe_n{n_train}_{reader}_perclip.npy', np.stack([score[s] for s in SETS]))
    json.dump(res, open(f'{OUT}/probe_n{n_train}_{reader}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
