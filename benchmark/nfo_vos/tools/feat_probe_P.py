"""Layer-wise probe (2026-11-24): is Ĉ's hidden-person information lost in the Fusion's SHARED s16 input projection P
(256 -> 64, applied to X, C and B alike; fusion_adapter.Fusion.z)? Same cached cells and readout as tools/feat_probe.py, but
the probe sees the Fusion's own z-input [P X, P C, P B, P X − P C, P X − P B] instead of the raw features.
P from the feat adapter (it700) and the pixel adapter (pf, it200), crossed with Ĉ and the pixel integral.

    PYTHONPATH=. python -m benchmark.nfo_vos.tools.feat_probe_P
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos.tools.feat_probe import OUT, fit

D = 'results/benchmark/rr/layers_fusion_adapter'
PS = {'P_feat': f'{D}/feat_seed0/feat_seed0/fusion_it700.pt', 'P_pix': f'{D}/token_seed0/fusion_it200.pt'}


def projP(ck):
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    sd = sd.get('fusion', sd)
    w, b = sd['P.16.weight'][:, :, 0, 0].float().cuda(), sd['P.16.bias'].float().cuda()
    return lambda x: x.cuda().float() @ w.T + b


def zin(c, s, P):
    X, B = c['none'][:, :256], c['none'][:, 256:512]
    pt, pb = P(X), P(B)
    if s == 'none':
        return torch.cat([pt, pb, pt - pb], 1)
    pc = P(c[s][:, 512:768])
    return torch.cat([pt, pc, pb, pt - pc, pt - pb], 1)


def main():
    tr, va = torch.load(f'{OUT}/cells_n60.pt', weights_only=False)
    ytr = np.concatenate([y for _, y, _ in tr])
    rng = np.random.default_rng(0)
    for pn, ck in PS.items():
        P = projP(ck)
        sc = {}
        for s in ('none', 'learned', 'pixel'):
            mu, sd, w, b = fit(torch.cat([zin(c, s, P) for c, _, _ in tr]), ytr)
            per = []
            for c, y, h in va:
                p = (((zin(c, s, P) - mu) / sd) @ w + b).cpu().numpy(); sel = h | ~y
                per.append(roc_auc_score(y[sel], p[sel]) if h.any() and (~y).any() else np.nan)
            sc[s] = np.array(per)
        out = f'{pn}: hidden AUC none {np.nanmean(sc["none"]):.3f}'
        for s in ('learned', 'pixel'):
            d = sc[s] - sc['none']; ok = ~np.isnan(d)
            bs = [d[ok][rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(2000)]
            out += f' | {s} {np.nanmean(sc[s]):.3f} gain {d[ok].mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]'
        d = sc['learned'] - sc['pixel']; ok = ~np.isnan(d)
        bs = [d[ok][rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(2000)]
        print(out + f' | learned − pixel {d[ok].mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]', flush=True)


if __name__ == '__main__':
    main()
