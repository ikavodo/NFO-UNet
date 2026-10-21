"""Per-frame runtime on the NFO windows, same GPU: ours (adapter/inference.fast_masks, the benchmark path) vs B0
(SAM2.1 video) / T1 (SAMURAI) through baselines/run_sam2_video.run_trial, the path that produced their masks.
Median over windows, first window excluded as warm-up. Ours excludes the once-per-window empty-scene median.
Measured 2026-10-20, RTX PRO 4000 laptop: ours 61.1 ms, B0 53 ms (the original two-encode adapter path was 445 ms).

    TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.timing ours|b0|t1 [n_windows]
"""
import json
import sys

import numpy as np
import torch

from benchmark.nfo_vos import trials as TR


def main(method='ours', n=12):
    trials = [t for t in json.load(open(TR.OUT)) if t['admissible']][:int(n)]
    print('GPU', torch.cuda.get_device_name(0), 'torch', torch.__version__)
    if method == 'ours':
        from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
        from benchmark.nfo_vos.adapter.inference import fast_masks
        from benchmark.nfo_vos.eval.dump_arms import CKPT, load_net
        pred = image_predictor()
        net = load_net(pred, CKPT)
        secs = [fast_masks(pred, net, t)[1] for t in trials]
    else:
        from benchmark.nfo_vos.baselines.run_sam2_video import build_predictor, run_trial
        vp = build_predictor(method)
        secs = [run_trial(vp, t)['sec_per_frame'] for t in trials]
    print(f'{method}: {1000 * np.median(secs[1:]):.1f} ms per frame (median over {len(secs) - 1} windows)')


if __name__ == '__main__':
    main(*sys.argv[1:])
