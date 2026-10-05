"""Composite-prepend (master_thesis docs/integrated_memory_proposal_2026-09-11.md): one synthetic
frame, the de-occluded person pasted onto a clean background at its t0 position, is prepended to
the untouched raw window. B0/T1 are prompted on it, so their conditioning memory is the clean
appearance. Ported from master_thesis@999b22f experiments/prototypes/wild/composite_prepend_tracking.py
(composite rule, FEATHER, match_moments). Deviations, each a pilot ruling:
  - causal horizon: N frames <= t0 at stride s (streaming tracker's 7 x 2), not the whole clip;
  - alignment: GT box centres (plausibility check, as asked), whole-frame translation, native res;
  - background: median of the trial's person-free warm-up frames (the original's mean of raw
    frames would keep a ghost of the person with only N = 7 frames);
  - S: SAM2 base_plus image predictor on the integrated image, prompted with the t0 GT box + p*.

    python -m benchmark.nfo_vos.composite       # writes results/benchmark/pilot/composites/<id>.jpg
"""
import json
import os

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

N, STRIDE = 7, 2
FEATHER = 9                      # composite_prepend_tracking.py:69 (px, kept at native res)
NATIVE = 'data/nfo_final/nfo_final'
OUT = 'results/benchmark/pilot/composites'
TRACE = 'images/benchmark/pilot_checks/composites'


def match_moments(src, ref, mask=None):
    """Verbatim port of composite_prepend_tracking.py:90-98."""
    sel = mask if mask is not None else np.ones_like(src, bool)
    ss, sm = src[sel].std(), src[sel].mean()
    rs, rm = ref.std(), ref.mean()
    if ss < 1e-6:
        return src
    return (src - sm) * (rs / ss) + rm


def horizon_frames(t0, n, stride, has_gt):
    return [k for k in (t0 - stride * j for j in range(n - 1, -1, -1)) if k >= 0 and has_gt(k)]


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def native(idx):
    return cv2.imread(f'{NATIVE}/{TR.SEQ}/{idx:05d}.jpg', 0)


def gt_centre_native(idx):
    x0, y0, x1, y1 = gt_to_native(TR.bbs()[idx][0], TR.NATIVE_W, TR.NATIVE_H)
    return (x0 + x1) / 2, (y0 + y1) / 2


def build(trial, n=N, stride=STRIDE):
    t0 = trial['t0']
    has_gt = lambda k: k in TR.bbs() and TR.bbs()[k] and TR.bbs()[k][0].x >= 0
    ks = horizon_frames(t0, n, stride, has_gt)
    cx0, cy0 = gt_centre_native(t0)
    aligned = []
    for k in ks:
        cx, cy = gt_centre_native(k)
        aligned.append(shift(native(k), cx0 - cx, cy0 - cy).astype(np.float32))
    integrated = np.median(np.stack(aligned), axis=0)
    w0, w1 = trial['warmup']
    background = np.median(np.stack([native(i).astype(np.float32) for i in range(w0, w1 + 1)]), axis=0)
    raw0 = native(t0).astype(np.float32)

    R.use_base_plus()
    pred = R.gt_sam_gate._predictor
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        pred.set_image(np.stack([integrated.clip(0, 255).astype(np.uint8)] * 3, -1))
        m, _, _ = pred.predict(point_coords=np.array([trial['point_native']]), point_labels=np.array([1]),
                               box=np.array(trial['box_native']), multimask_output=False)
    S = m[0] > 0.5
    if not S.any():
        return None, dict(frames=ks, S=0)
    integ = match_moments(integrated, raw0, mask=S)
    alpha = cv2.GaussianBlur(S.astype(np.float32), (0, 0), FEATHER)
    comp = (alpha * integ + (1 - alpha) * background).clip(0, 255).astype(np.uint8)
    return comp, dict(frames=ks, S=int(S.sum()), integrated=integrated, background=background, raw0=raw0, mask=S)


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TRACE, exist_ok=True)
    meta = {}
    for t in [t for t in json.load(open(TR.OUT)) if t['admissible']]:
        comp, info = build(t)
        meta[t['id']] = dict(frames=info['frames'], S_px=info['S'], ok=comp is not None)
        if comp is None:
            print(f"{t['id']}: empty S -> no composite"); continue
        cv2.imwrite(f"{OUT}/{t['id']}.jpg", comp)
        x0, y0, x1, y1 = (int(v) for v in t['box_native'])
        pad = int(0.6 * (y1 - y0))
        sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
        tiles = []
        for name, im in (('raw t0', info['raw0']), (f"integrated ({len(info['frames'])} fr)", info['integrated']),
                         ('composite', comp.astype(np.float32))):
            tile = cv2.cvtColor(cv2.resize(im[sl].clip(0, 255).astype(np.uint8), (180, 240)), cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(tile)
        cv2.imwrite(f"{TRACE}/{t['id']}.png", np.hstack(tiles))
        print(f"{t['id']}: {len(info['frames'])} frames in horizon, S = {info['S']} px")
    json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
