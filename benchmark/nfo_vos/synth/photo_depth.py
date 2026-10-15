"""Disparity of the foliage template (Depth Anything V2 Large, as diag_depth.py; larger = nearer), for the depth-test
occluder in gpjatk_foliage.py. Run in the dvas env (has transformers):
    ~/miniconda3/envs/dvas/bin/python -m benchmark.nfo_vos.synth.photo_depth
Template rows 0 … 0.6·H (the foliage part). Inference on overlapping 1036-px tiles is avoided: one pass at short side
1036 (diag_depth's high-res setting), output bilinearly resized to the template resolution. Disparity is relative
(affine-ambiguous); only its ordering is used (quantile thresholds).
"""
import os

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from benchmark.nfo_vos.synth.gpjatk_foliage import TEMPLATE

MODEL = 'depth-anything/Depth-Anything-V2-Large-hf'
OUT = 'data/synthetic/foliage_depth'


def main():
    t = cv2.imread(TEMPLATE)[:, :, ::-1]
    t = t[:int(0.6 * t.shape[0])]
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = AutoModelForDepthEstimation.from_pretrained(MODEL).cuda().eval()
    s = 756 / min(t.shape[:2])   # 1036 (diag_depth's setting) runs out of memory on 16 GB at this 2.5:1 aspect
    h, w = int(round(t.shape[0] * s / 14)) * 14, int(round(t.shape[1] * s / 14)) * 14
    x = proc(images=t, return_tensors='pt', do_resize=True, size={'height': h, 'width': w}, keep_aspect_ratio=False,
             ensure_multiple_of=14)['pixel_values'].cuda()
    with torch.inference_mode():
        d = net(pixel_values=x).predicted_depth[0].float().cpu().numpy()
    d = cv2.resize(d, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_LINEAR)
    os.makedirs(OUT, exist_ok=True)
    np.save(f'{OUT}/veronica_disp.npy', d.astype(np.float16))
    v = cv2.applyColorMap((255 * (d - d.min()) / (d.max() - d.min())).astype(np.uint8), cv2.COLORMAP_MAGMA)
    g = cv2.cvtColor(t[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    os.makedirs('images/benchmark/gpjatk_foliage', exist_ok=True)
    cv2.imwrite('images/benchmark/gpjatk_foliage/template_depth.png',
                cv2.resize(np.concatenate([np.stack([g] * 3, -1), v], 0), (960, 768)))
    print('disparity', d.shape, 'range', float(d.min()), float(d.max()), 'input', (h, w))


if __name__ == '__main__':
    main()
