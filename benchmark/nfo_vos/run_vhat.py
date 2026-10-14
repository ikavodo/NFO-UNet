"""V̂_T1 on every admissible window, GT-free, fixed configuration (2026-10-14), for scoring once seq2–4 human labels
exist. Configuration fixed on seq1 (factorised_gtfree.py, factorised_extras.py) before any seq2–4 frame is scored:

  V̂_T1(k) = (M_C(k) ∧ z(pix_bg)(k) > 7.5) ∪ T1(k)       for local index j ≥ 6, else T1(k)
  composite: history k-6..k-1, GT-free cleaning, aligned by a Theil–Sen fit of T1 mask centroids over local [j-12, j]
  M_C: SAM2 image predictor (sam2.1 b+) on the composite, prompt = blob tracker's merged box at k + its centre,
       else its merged centre point only
  z(pix_bg): per-frame robust z (median, 1.4826 MAD) of log(1 + 5×5 mean (R_k − B)²), B = warm-up median

Writes results/benchmark/rr/masks/t1-vhat/<trial>.npz ('masks', same layout as masks/t1) and a per-window summary.
"""
import json
import os
import time

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.factorised_extras import pixz
from benchmark.nfo_vos.factorised_gtfree import composite, interior_point, theil_at
from tracking.core.blob_tracker import merged_center

TZ = 7.5
OUT = f'{TR.RES}/masks/t1-vhat'


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    os.makedirs(OUT, exist_ok=True)
    pred = R.image_predictor()
    for n, t in enumerate(T):
        path = f"{OUT}/{t['id']}.npz"
        if os.path.exists(path):
            continue
        t_start, seq = time.time(), t['seq']
        M1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks'].astype(bool)
        out = M1.copy()
        w0, w1 = t['warmup']
        bg = np.median(np.stack([V.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
        Tk = R.track(t)
        added = []
        for j in range(6, len(t['frames'])):
            k = t['frames'][j]
            span = range(max(0, j - 12), j + 1)
            cm = [(l, *np.argwhere(M1[l])[:, ::-1].mean(0)) for l in span if M1[l].any()]
            if len(cm) < 2:
                continue
            ev = [j] + list(range(j - 6, j))
            fx = theil_at([c[0] for c in cm], [c[1] for c in cm], ev)
            fy = theil_at([c[0] for c in cm], [c[2] for c in cm], ev)
            sh = {t['frames'][l]: (fx[j] - fx[l], fy[j] - fy[l]) for l in range(j - 6, j)}
            C = composite(seq, k, sh, bg)
            anchor = Tk['chain'].get(j)
            if anchor is None:
                continue
            cx, cy, bb = merged_center(Tk['dets'][j], *anchor, Tk['kw']['merge_radius'], return_box=True)
            pt = TR.to_native(cx, cy)
            bx = None if bb is None else np.array([*TR.to_native(bb[0], bb[1]), *TR.to_native(bb[2], bb[3])], np.float32)
            with torch.inference_mode():
                pred.set_image(np.stack([np.clip(C, 0, 255).astype(np.uint8)] * 3, -1))
                m, _, _ = pred.predict(point_coords=np.array([pt]), point_labels=np.array([1]), box=bx, multimask_output=False)
            pb = np.log(cv2.boxFilter((V.native(k, seq).astype(np.float32) - bg) ** 2, -1, (5, 5)) + 1)
            add = (m[0] > 0) & (pixz(pb) > TZ) & ~M1[j]
            out[j] |= add
            added.append(add.sum() / max(M1[j].sum(), 1))
        np.savez_compressed(path, masks=out)
        print(f"{n + 1}/{len(T)} {t['id']}: {len(added)} frames, added area {np.mean(added) if added else 0:.2f}× T1, "
              f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == '__main__':
    main()
