"""Per-stage runtime of the deployed adapter path (inference.masks_from_arrays): where do the ms/frame go?

Stages are timed by wrapping the callables the path uses, each with torch.cuda.synchronize() before and after (so the
breakdown removes CPU/GPU overlap; the total with syncs is printed next to the untimed total). Stages: encoder
(sam.forward_image + _prepare_backbone_features, frames + empty scene + the pixel route's C), tracker (kernel_track,
CPU), key shift (feat route), aggregator (feat route), adapter (Fusion), mask decoder; 'rest' = total − stages
(transforms, gates, change map upsampling, CPU copies, mask resize). Median per-window ms/frame over windows 2..n.

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.profile_stages feat|pixel [n_windows]
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter import feat_integral as FI
from benchmark.nfo_vos.adapter import inference as INF
from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
from benchmark.nfo_vos.eval.dump_arms import ARM_CKPT, CKPT, load_net

ACC = defaultdict(float)


def timed(name, fn):
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize(); ACC[name] += time.perf_counter() - t
        return r
    return w


def main(arm='feat', n=12):
    trials = [t for t in json.load(open(TR.OUT)) if t['admissible']][:int(n)]
    pred = image_predictor(); sam = pred.model
    net = load_net(pred, ARM_CKPT['adapter-feat'] if arm == 'feat' else CKPT)
    plain = [INF.fast_masks(pred, net, t)[1] for t in trials]           # untimed reference, same windows
    sam.forward_image = timed('encoder', sam.forward_image)
    sam._prepare_backbone_features = timed('encoder', sam._prepare_backbone_features)
    INF.kernel_track = timed('tracker', INF.kernel_track)
    INF.decode = timed('decoder', INF.decode)
    FI.shift_keys = timed('key shift', FI.shift_keys)
    net.forward = timed('adapter', net.forward)
    if getattr(net, 'agg', None) is not None:
        net.agg.forward = timed('aggregator', net.agg.forward)
    rows = []
    for t in trials:
        ACC.clear(); nf = len(t['frames'])
        tot = INF.fast_masks(pred, net, t)[1]
        rows.append({'total': tot, **{k: v / nf for k, v in ACC.items()}})
    keys = ['encoder', 'tracker', 'key shift', 'aggregator', 'adapter', 'decoder']
    print(f'{arm}: untimed total {1000 * np.median(plain[1:]):.1f} ms/frame; with per-stage syncs '
          f'{1000 * np.median([r["total"] for r in rows[1:]]):.1f} ms/frame (median over {len(rows) - 1} windows)')
    for k in keys:
        v = [r.get(k, 0) for r in rows[1:]]
        if max(v) > 0:
            print(f'  {k:10s} {1000 * np.median(v):6.1f} ms')
    rest = [r['total'] - sum(r.get(k, 0) for k in keys) for r in rows[1:]]
    print(f'  {"rest":10s} {1000 * np.median(rest):6.1f} ms')


if __name__ == '__main__':
    main(*sys.argv[1:])
