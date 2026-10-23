"""Masks of our arms on one NFO window per call (baselines B0 / T1: baselines/run_sam2_video.py).
Writes results/benchmark/rr/masks/<arm>/<trial_id>.npz in the B0/T1 schema (masks [40, 600, 800] bool, frames [40]);
'trkbox' also stores boxes [40, 4] (native xyxy). Skips a window whose file exists.
  trkbox          kernel-f16 tracker box itself (fixed t0 size), masks = filled box
  trk             per-frame SAM2, tracker box + point, no memory
  adapter-pf      per-frame adapter, v3.4 loss, step 200, seed 0 (frames < 6 have no composite -> trk)
  adapter-s1/-s2  the same, seeds 1 / 2
  <arm>-noC       the adapter with the composite removed at inference (C := empty scene, a := 0)
  <arm>-h<n>      the adapter with n past frames in the composite instead of 6 (n in the training range 4..8)

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.eval.dump_arms ARM TRIAL_INDEX
"""
import json
import os
import re
import sys

import numpy as np
import torch

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.fusion_adapter import Fusion, image_predictor
from benchmark.nfo_vos.adapter.inference import fast_masks

CKPT_DIR = 'results/benchmark/rr/layers_fusion_adapter'
CKPT = f'{CKPT_DIR}/token_seed0/fusion_it200.pt'
ARM_CKPT = {'adapter-pf': CKPT, 'adapter-s1': f'{CKPT_DIR}/token_seed1/fusion_it200.pt',
            'adapter-s2': f'{CKPT_DIR}/token_seed2/fusion_it200.pt'}


def load_net(pred, ckpt):
    net = Fusion().cuda(); net.init_fallback(pred.model)
    net.load_state_dict(torch.load(ckpt)); net.eval()
    return net


def main(arm, idx):
    t = json.load(open(TR.OUT))[int(idx)]
    path = f'{TR.RES}/masks/{arm}/{t["id"]}.npz'
    if os.path.exists(path):
        return print('exists', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred = image_predictor()
    if arm == 'trkbox':
        from benchmark.nfo_vos.adapter.tracker import kernel16_track
        kc, (w, h) = kernel16_track(t, pred)
        boxes = np.concatenate([kc - (w / 2, h / 2), kc + (w / 2, h / 2)], 1).astype(np.float32)
        m = np.zeros((len(boxes), TR.NATIVE_H, TR.NATIVE_W), bool)
        for j, b in enumerate(np.round(boxes).astype(int)):
            m[j, max(b[1], 0):b[3] + 1, max(b[0], 0):b[2] + 1] = True
        np.savez_compressed(path, masks=m, frames=np.array(t['frames']), boxes=boxes)
        return print(arm, t['id'], 'saved', path)
    h = re.search(r'-h(\d+)$', arm)
    base = arm.removesuffix(h.group(0)) if h else arm
    net = None if base == 'trk' else load_net(pred, ARM_CKPT[base.removesuffix('-noC')])
    m, sec = fast_masks(pred, net, t, noC=base.endswith('-noC'), hist=int(h.group(1)) if h else 6)
    np.savez_compressed(path, masks=m, frames=np.array(t['frames']))
    print(arm, t['id'], 'saved', path, f'{1000 * sec:.0f} ms/frame', 'mean area px', int(m.sum((1, 2)).mean()))


if __name__ == '__main__':
    main(*sys.argv[1:])
