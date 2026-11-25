"""Why is Ĉ (learned feature integral) less useful than enc(pixel integral) although closer to the clean features T?
Hypothesis (2026-11-24): the softmax average with the null key ∅ = B shrinks Ĉ toward the empty scene, so the change signal
the adapter reads (C − B) is weak. On 30 synthetic val clips, s16 cells of Ω (hidden, seen in history) and of the visible
person: r = ‖C − B‖ / ‖T − B‖ (1 = full change signal, 0 = C equals the empty scene) and cos(C − B, T − B), per source
C ∈ {learned Ĉ, pixel enc(C), F_t}.

    PYTHONPATH=. python -m benchmark.nfo_vos.tools.feat_stats FEAT_CKPT
"""
import sys

import numpy as np
import torch

from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips, build
from benchmark.nfo_vos.adapter.feat_stage0 import _agg_in
from benchmark.nfo_vos.eval.dump_arms import load_net


@torch.no_grad()
def main(ckpt):
    pred = image_predictor()
    agg = load_net(pred, ckpt).agg.eval()
    val = torch.utils.data.DataLoader(Clips('val', n_val=30), batch_size=None, num_workers=6, collate_fn=_ident)
    acc = {}
    for k, raw in enumerate(val):
        f, _ = build(pred, raw, max_targets=16, rng=np.random.default_rng(k), feat=True, with_pixel=True)
        with torch.autocast('cuda', dtype=AMP):
            ol = agg(*_agg_in(f), phase=f['phase'])
        B, T = f['B16'].float(), f['T16'].float()
        dT = T - B
        for reg, m in (('Omega', f['Om16'][:, 0]), ('visible', f['Vis16'][:, 0])):
            if not m.any():
                continue
            for nm, C in (('learned', ol['C16']), ('pixel', f['P16']), ('F_t', f['X16'])):
                dC = C.float() - B
                r = (dC.norm(dim=1) / dT.norm(dim=1).clamp(min=1e-6))[m]
                c = torch.nn.functional.cosine_similarity(dC, dT, dim=1, eps=1e-6)[m]
                acc.setdefault((reg, nm), []).append((float(r.median()), float(c.mean())))
        print(f'clip {k}', flush=True)
    for (reg, nm), v in acc.items():
        v = np.array(v)
        print(f'{reg:8s} {nm:8s}  median ‖C−B‖/‖T−B‖ {v[:, 0].mean():.3f}   cos(C−B, T−B) {v[:, 1].mean():.3f}   ({len(v)} clips)')


if __name__ == '__main__':
    main(sys.argv[1])
