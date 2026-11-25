"""Fixed-rule feature integral (FeatIntegral fixed=True: α_k = w_k G_k, the pixel integral's recency × robust change gate,
in feature space, nothing learned) through the same synthetic → NFO transfer readout as feat_distil (held-out seq1
crossings s0–s3, reader fitted on 30 synthetic train clips). Question: does a parity-safe α transfer like the pixel
integral (the learned α is the cause of the gap) or like the learned Ĉ (feature-space mixing itself shifts)?

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_fixed_transfer
"""
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import Clips
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_distil import OUT, R, TEST, nfo_cells
from benchmark.nfo_vos.tools.feat_probe import cells as syn_cells, fit
from benchmark.nfo_vos.tools.feat_probe_nfo import NETS


class Fixed(torch.nn.Module):
    def __init__(self, a):
        super().__init__(); self.a = a

    def forward(self, *x, phase=None, **k):
        return self.a(*x, fixed=True)


def main():
    pred = image_predictor()
    feat = load_net(pred, NETS['learned']); fx = Fixed(feat.agg); feat.agg = fx
    trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    nfo = torch.load(f'{R}/probe_nfo/cells_seq1.pt', weights_only=False)
    for i in TEST:
        c = nfo_cells(pred, feat, trials[i]); assert len(c) == len(nfo[i][1])
        nfo[i][0]['fixed'] = c
    syn, _ = torch.load(f'{R}/probe/cells_n60.pt', weights_only=False); syn = syn[:30]
    new = syn_cells(pred, {'learned': fx, 'nbr1': fx},
                    torch.utils.data.DataLoader(Clips('train', seed=7), batch_size=None, num_workers=6, collate_fn=_ident), 30)
    for a, b in zip(syn, new):
        assert len(a[1]) == len(b[1]) and torch.equal(a[0]['none'], b[0]['none']), 'synthetic clips differ from the cache'
        a[0]['fixed'] = b[0]['learned']
    ytr = np.concatenate([y for _, y, _ in syn]); sc = {}
    for s in ('none', 'fixed'):
        mu_, sd, w, b_ = fit(torch.cat([c[s] for c, _, _ in syn]), ytr)
        per = []
        for i in TEST:
            c, y, oc = nfo[i]; Z = c['none'] if s == 'none' else torch.cat([c['none'], c[s]], 1)
            p = (((Z.cuda().float() - mu_) / sd) @ w + b_).cpu().numpy(); sel = oc | ~y
            per.append((roc_auc_score(y, p), roc_auc_score(y[sel], p[sel])))
        sc[s] = np.array(per)
    res = json.load(open(f'{OUT}/transfer_heldout.json'))
    for j, nm in enumerate(('all', 'occl')):
        d = sc['fixed'][:, j] - sc['none'][:, j]; res[f'fixed_{nm}'] = d.tolist()
        print(f'transfer gain fixed    {nm}: {d.mean():+.4f}  per held-out crossing ' + ' '.join(f'{x:+.3f}' for x in d), flush=True)
        for s in ('pixel', 'learned', 'distil'):
            print(f'   ({s} {nm}: {np.mean(res[f"{s}_{nm}"]):+.4f})', flush=True)
    json.dump(res, open(f'{OUT}/transfer_heldout.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
