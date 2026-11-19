"""Does the adapter USE its integral image on synthetic val (where the integral is known to be good)?
adapter vs adapter-noC (C := B, a := 0) per arm checkpoint, 30 val clips, paired clip bootstrap, all frames and history-
frames only (mode 0). Compare feature integral (FEAT ckpt with 'agg') and pixel integral (plain ckpt).

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.syn_noC ckpt1.pt [ckpt2.pt ...]
"""
import sys

import numpy as np
import torch

from benchmark.nfo_vos.adapter.fusion_adapter import _ident, evaluate, image_predictor, summarise
from benchmark.nfo_vos.adapter.fusion_data import Clips
from benchmark.nfo_vos.eval.dump_arms import load_net


def main(*ckpts):
    pred = image_predictor()
    val = list(torch.utils.data.DataLoader(Clips('val', n_val=30), batch_size=None, num_workers=6, collate_fn=_ident))
    for c in ckpts:
        net = load_net(pred, c)
        rows, _ = evaluate(pred, net, val, getattr(net, 'agg', None))
        for nm, rs in (('all frames', rows), ('mode 0', [r for r in rows if r['mode'] == 0])):
            s = summarise(rs)
            print(f"{c.split('layers_fusion_adapter/')[-1]:40s} {nm:10s} J adapter {s['adapter']['J']:.3f} noC "
                  f"{s['adapter-noC']['J']:.3f} trk-box {s['trk-box']['J']:.3f} | adapter − noC "
                  f"{s['dJ_adapter-noC'][0]:+.3f} [{s['dJ_adapter-noC'][1]:+.3f}, {s['dJ_adapter-noC'][2]:+.3f}]", flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
