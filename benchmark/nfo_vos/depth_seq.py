"""Empty-scene disparity for seq2–4, as seq1's results/benchmark/rr/depth/seq1_bg_disp_1036.npy (2026-10-16):
Depth Anything V2 Large (diag_depth.depth_fn, short side 1036) on the median of the sequence's longest warm-up range,
min–max normalised to [0, 1] (larger = nearer).      ~/miniconda3/envs/dvas/bin/python -m benchmark.nfo_vos.depth_seq
"""
import json

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = 'depth-anything/Depth-Anything-V2-Large-hf'


def depth_fn(short):      # = diag_depth.depth_fn (copied: diag_depth imports sklearn, absent in the dvas env)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = AutoModelForDepthEstimation.from_pretrained(MODEL).cuda().eval()

    @torch.inference_mode()
    def f(gray):
        x = proc(images=cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB), return_tensors='pt',
                 size={'height': short, 'width': short})
        d = net(**{k: v.cuda() for k, v in x.items()}).predicted_depth[None]
        return torch.nn.functional.interpolate(d, (600, 800), mode='bilinear')[0, 0].cpu().numpy()
    return f


f = depth_fn(1036)
T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
for seq in ('seq2', 'seq3', 'seq4'):
    w0, w1 = max({tuple(t['warmup']) for t in T if t['seq'] == seq}, key=lambda w: w[1] - w[0])
    bg = np.median(np.stack([V.native(i, seq) for i in range(w0, w1 + 1)]), 0)
    d = f(bg); d = (d - d.min()) / (d.max() - d.min())
    np.save(f'{TR.RES}/depth/{seq}_bg_disp_1036.npy', d.astype(np.float32))
    cv2.imwrite(f'images/benchmark/rr_checks/depth/{seq}_bg_disp_1036.jpg',
                cv2.resize(np.hstack([bg, d * 255]).astype(np.uint8), None, fx=0.5, fy=0.5))
    print(seq, (w0, w1), np.percentile(d, [5, 50, 95]).round(3))
