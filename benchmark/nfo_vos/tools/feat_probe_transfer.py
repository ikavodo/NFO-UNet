"""Synthetic → NFO transfer probe: a linear reader fitted (and standardised) on SYNTHETIC cells, scored on NFO seq1 cells,
i.e. the adapter's own situation (trained on synthetic, applied to NFO) without the adapter. If Ĉ's information is there on
NFO (feat_probe_nfo) but a synthetic-trained reader loses it, Ĉ's statistics shift between the domains more than the
pixel integral's. Train: feat_probe's cached 60 synthetic train clips (label: person mask share); test: feat_probe_nfo's
cached 8 seq1 crossings (label: GT-box core). Sets none = [F_t, B], + pixel enc(C), + learned Ĉ (it700). Gain = AUC(set) −
AUC(none) per crossing; paired learned − pixel with a crossing bootstrap. Primary readout: all cells.

    PYTHONPATH=. [NORM=clip] python -m benchmark.nfo_vos.tools.feat_probe_transfer
NORM=clip: every set standardised per channel within each clip / crossing before the reader (AdaBN-style domain
normalisation, Li et al. ICLR-W 2017, arXiv 1603.04779): does the Ĉ − pixel transfer gap close under a mean/scale fix?
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos.tools.feat_probe import fit

R = 'results/benchmark/rr/feat_integral'
SETS = ('none', 'pixel', 'learned')


def main():
    tr, _ = torch.load(f'{R}/probe/cells_n60.pt', weights_only=False)
    nfo = torch.load(f'{R}/probe_nfo/cells_seq1.pt', weights_only=False)
    ytr = np.concatenate([y for _, y, _ in tr])
    norm = os.environ.get('NORM', '')
    if norm == 'clip':
        for c, _, _ in tr + nfo:
            for s in SETS:
                Z = c[s].float(); c[s] = ((Z - Z.mean(0)) / Z.std(0).clamp(min=1e-4)).half()
    sc = {}
    for s in SETS:                                  # pixel / learned cells are C only (NFO) or [F_t, B, C] (synthetic)
        mu, sd, w, b = fit(torch.cat([c[s] for c, _, _ in tr]), ytr)
        per = []
        for c, y, oc in nfo:
            Z = c['none'] if s == 'none' else torch.cat([c['none'], c[s]], 1)
            p = (((Z.cuda().float() - mu) / sd) @ w + b).cpu().numpy(); sel = oc | ~y
            per.append((roc_auc_score(y, p), roc_auc_score(y[sel], p[sel])))
        sc[s] = np.array(per)
        print(f'{s:8s} transfer AUC all {sc[s][:, 0].mean():.3f} occl {sc[s][:, 1].mean():.3f}', flush=True)
    rng, res = np.random.default_rng(0), {}
    for a, b_ in (('pixel', 'none'), ('learned', 'none'), ('learned', 'pixel')):
        for j, nm in enumerate(('all', 'occl')):
            d = sc[a][:, j] - sc[b_][:, j]
            ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
            res[f'{a}-{b_}_{nm}'] = [float(d.mean()), *map(float, ci), d.tolist()]
            print(f'{a} − {b_} {nm}: {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  per crossing ' + ' '.join(f'{x:+.3f}' for x in d), flush=True)
    json.dump(res, open(f'{R}/probe_nfo/transfer_seq1{"_" + norm if norm else ""}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
