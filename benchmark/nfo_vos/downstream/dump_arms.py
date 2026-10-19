"""Masks of the arms the downstream benchmark still lacks, one window per call (plan_downstream_benchmark.md, Task 3).
Writes results/benchmark/rr/masks/<arm>/<trial_id>.npz in the B0/T1 schema (masks [40, 600, 800] bool, frames [40]);
'trkbox' also stores boxes [40, 4] (native xyxy). Skips a window whose file exists.
  trkbox            kernel-f16 tracker box itself (fixed t0 size), masks = filled box
  trk               per-frame SAM2, tracker box + point, no memory
  adapter-pf        per-frame adapter v3.4 it200 (frames < 6 have no composite → trk)
  vid-plain         SAM2 video, GT box + point at t0 only (must reproduce masks/b0: the SC)
  vid-prompts       SAM2 video + tracker box + point every frame (decoder only)
  vid-adapter-pre   + adapter residual before memory attention
  vid-adapter-noC   vid-adapter-pre with the composite removed (C := empty scene, a := 0)
  vid-oracle-pseudo ORACLE: vid-prompts, but the frozen pseudo-GT is written into memory (causal: past frames only)
  vid-oracle-ceil   ORACLE: same with the SAM2-ceiling (both bound what memory could deliver; scored on hand J /
                    box / recognition, never on the label written in, which made the 2026-10-16 number circular)
    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.downstream.dump_arms ARM TRIAL_INDEX [ckpt]
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR
from benchmark.nfo_vos.layers import fusion_data as FD
from benchmark.nfo_vos.layers.fusion_adapter import Fusion, decode
from benchmark.nfo_vos.layers.video_adapter import S1, per_frame_inputs, run_window

CKPT = 'results/benchmark/rr/layers_fusion_adapter/token_seed0/fusion_it200.pt'
VIDEO = {'vid-plain': 'plain', 'vid-prompts': 'prompts', 'vid-adapter-pre': 'adapter-pre', 'vid-adapter-noC': 'adapter-pre',
         'vid-oracle-pseudo': 'oracle-mem', 'vid-oracle-ceil': 'oracle-mem'}


def _up(low):
    return cv2.resize(low[0, 0].float().cpu().numpy(), (TR.NATIVE_W, TR.NATIVE_H), interpolation=cv2.INTER_LINEAR) > 0


@torch.no_grad()
def per_frame_masks(pred, net, t, inputs, arm):
    from benchmark.nfo_vos import velocity as V
    out = np.zeros((len(t['frames']), TR.NATIVE_H, TR.NATIVE_W), bool)
    for j, (k, d) in enumerate(zip(t['frames'], inputs)):
        it = d.get('it') or dict(X=V.native(k, t['seq'])[None], Ct=V.native(k, t['seq'])[None], a=np.zeros((1, 256, 256), np.float32),
                                 V=np.zeros((1, 256, 256), np.float32), box=d['box'][None], box_trk=d['box'][None],
                                 trk=d['pt'][None], mode=np.zeros(1, int), bg=V.native(k, t['seq']))
        f = FD.encode(pred, [it])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if arm == 'adapter-pf' and d.get('it') is not None:
                o = net(f, sam=pred.model)
                low = decode(pred.model, o, token=o['token'])
            else:
                low = decode(pred.model, dict(f, box=f['box_trk']), point=True)
        out[j] = _up(low)
    return out


def oracle_labels(arm, t):
    """{local frame index: bool mask} written into memory by the oracle arms (video_adapter enc hook), else None."""
    if arm == 'vid-oracle-pseudo':
        L = np.load(f"{TR.RES}/labels_sil_frozen/{t['id']}.npz")
        return {int(k): L[k] for k in L.files}
    if arm == 'vid-oracle-ceil':
        ms = {j: TR.gt_mask_native(k, t['seq']) for j, k in enumerate(t['frames'])}
        return {j: m for j, m in ms.items() if m is not None}
    return None


def main(arm, idx, ckpt=CKPT):
    t = json.load(open(TR.OUT))[int(idx)]
    path = f'{TR.RES}/masks/{arm}/{t["id"]}.npz'
    if os.path.exists(path):
        return print('exists', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pred = R.image_predictor()
    net = None
    if arm in ('adapter-pf', 'vid-adapter-pre', 'vid-adapter-noC'):
        net = Fusion(token=True).cuda(); net.init_fallback(pred.model); net.load_state_dict(torch.load(ckpt)); net.eval()
    inputs = per_frame_inputs(pred, net if arm not in ('trk', 'trkbox') else None, t, noC=arm == 'vid-adapter-noC')
    extra = {}
    if arm == 'trkbox':
        boxes = np.stack([d['box'] / S1 for d in inputs])
        m = np.zeros((len(boxes), TR.NATIVE_H, TR.NATIVE_W), bool)
        for j, b in enumerate(np.round(boxes).astype(int)):
            m[j, max(b[1], 0):b[3] + 1, max(b[0], 0):b[2] + 1] = True
        extra['boxes'] = boxes
    elif arm in ('trk', 'adapter-pf'):
        m = per_frame_masks(pred, net, t, inputs, arm)
    else:
        from benchmark.nfo_vos.run_sam2_video import build_predictor
        dt = torch.float16 if os.environ.get('NFO_VID_FP16') else torch.bfloat16   # fp16 = run_sam2_video's protocol (SC)
        m = run_window(build_predictor('b0'), t, inputs, VIDEO[arm], labels=oracle_labels(arm, t), dtype=dt)
    np.savez_compressed(path, masks=m, frames=np.array(t['frames']), **extra)
    print(arm, t['id'], 'saved', path, 'mean area px', int(m.sum((1, 2)).mean()))


if __name__ == '__main__':
    main(*sys.argv[1:])
