"""Does the adapter USE its integral image on synthetic val (where the integral is known to be good)?
adapter vs adapter-noC (C := B, a := 0) per arm checkpoint, 30 val clips, paired clip bootstrap, all frames and history-
frames only (mode 0). Compare feature integral (FEAT ckpt with 'agg') and pixel integral (plain ckpt).

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.syn_noC ckpt1.pt [ckpt2.pt ...]
"""
import json
import os
import sys

import numpy as np
import torch

from benchmark.nfo_vos.adapter.fusion_adapter import _ident, evaluate, image_predictor, summarise
from benchmark.nfo_vos.adapter.fusion_data import Clips
from benchmark.nfo_vos.eval.dump_arms import load_net

OUT = 'results/benchmark/rr/feat_integral'


def boot(rows, j=0, arm='adapter', ref='adapter-noC', rng=np.random.default_rng(0)):
    """adapter − noC of metric j (0 J, 3 amodal box IoU): per-clip mean over the given frames, then a clip bootstrap (95%)."""
    cl = sorted({r['clip'] for r in rows})
    d = np.array([np.mean([r[arm][j] - r[ref][j] for r in rows if r['clip'] == c]) for c in cl])
    if len(d) == 0:
        return 'n 0'
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return f"{d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] ({len(rows)} fr, {len(d)} clips)"


def main(*ckpts):
    pred = image_predictor()
    val = list(torch.utils.data.DataLoader(Clips('val', n_val=30), batch_size=None, num_workers=6, collate_fn=_ident))
    os.makedirs(OUT, exist_ok=True)
    for c in ckpts:
        net = load_net(pred, c)
        rows, _ = evaluate(pred, net, val, getattr(net, 'agg', None))
        name = c.split('layers_fusion_adapter/')[-1].replace('/', '_').replace('.pt', '')
        json.dump(rows, open(f'{OUT}/syn_noC_rows_{name}.json', 'w'))
        m0 = [r for r in rows if r['mode'] == 0]                      # frames with an integral image (history)
        hard = [r for r in m0 if r['vis'] < 0.5 and r['rec'] >= 0.5 * (1 - r['vis'])]
        print(f"== {name}: mode-0 frames {len(m0)}; median vis {np.median([r['vis'] for r in m0]):.2f}, "
              f"median rec {np.median([r['rec'] for r in m0]):.3f}; HARD (vis < 0.5, ≥ half of the hidden part seen "
              f"in history) {len(hard)}/{len(m0)} = {len(hard) / max(len(m0), 1):.1%}", flush=True)
        groups = [('mode 0 all', m0), ('HARD', hard)]
        groups += [(f'rho {lo}-{hi}', [r for r in m0 if lo <= r['rho'] < hi]) for lo, hi in ((0, .01), (.2, .4), (.4, .61))]
        groups += [(f'rec {lo}-{hi}', [r for r in m0 if lo <= r['rec'] < hi]) for lo, hi in ((0, .1), (.1, .3), (.3, 1.01))]
        groups += [(f'vis {lo}-{hi}', [r for r in m0 if lo <= r['vis'] < hi]) for lo, hi in ((0, .3), (.3, .6), (.6, 1.01))]
        for g, rs in groups:
            print(f"  {g:12s} adapter − noC J {boot(rs)} | box IoU {boot(rs, 3)}", flush=True)
            for o in ('oracle-fine', 'oracle-all'):
                if o in rs[0]:
                    print(f"  {'':12s} {o} − adapter J {boot(rs, 0, o, 'adapter')} | box IoU {boot(rs, 3, o, 'adapter')}", flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
