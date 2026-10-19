"""Third dense-label candidate (user, 2026-10-19): SAM2 on a GT-motion-aligned temporal composite.

For frame t: frames t−h..t+h translated onto t by the GT-box centre shift (recognition.gt_shifts), combined with
weights w_i = λ^|i−t| (λ = 1: uniform), no masks. SAM2.1 b+ image predictor, prompt = GT box + its centre (as the
SAM2-ceiling), single mask. The composite blurs the static foliage and integrates the person, so the mask is expected
to lean toward the full body (high R, lower P vs modal hand labels): this script measures that.
Step 1 (this file): agreement with the 43 hand frames beside SAM2-ceiling and the frozen pseudo-GT on the same frames.
Step 2 (only if step 1 is competitive): dense labels on all windows and the proxy-validity test (rank agreement).

  python -m benchmark.nfo_vos.downstream.composite_label
"""
import json
import os

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR
from benchmark.nfo_vos.downstream import box_metrics as BM, report as RP
from benchmark.nfo_vos.downstream.recognition import frame, gt_shifts, masked_composite

VARIANTS = {'h3 λ0.7': (3, 0.7), 'h3 uniform': (3, 1.0), 'h1 λ0.7': (1, 0.7), 'h0 (frame t only)': (0, 1.0)}


def composite_at(seq, k, h, lam):
    gtb = BM.gt_boxes(seq)
    idx = [i for i in range(k - h, k + h + 1) if i in gtb and os.path.exists(f'data/nfo_final/nfo_final/{seq}/{i:05d}.jpg')]
    imgs = [frame(seq, i) for i in idx]
    return masked_composite(imgs, [np.ones(imgs[0].shape[:2], bool)] * len(imgs), gt_shifts(gtb, idx, k), imgs[0],
                            weights=[lam ** abs(i - k) for i in idx])


@torch.no_grad()
def sam2_mask(pred, img, box):
    pred.set_image(img[..., ::-1].copy())
    c = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], np.float32)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        m, _, _ = pred.predict(point_coords=c, point_labels=np.array([1]), box=box[None], multimask_output=False)
    return m[0] > 0


def score(m, H, keep):
    if keep is not None:
        m, H = m & keep, H & keep
    tp = (m & H).sum()
    return tp / max((m | H).sum(), 1), tp / max(m.sum(), 1), tp / max(H.sum(), 1)


def main():
    pred = R.image_predictor()
    trials = json.load(open('results/benchmark/rr/trials.json'))
    pseudo = {}
    for t in trials:                                  # pseudo-GT is stored per window (local index); first hit wins
        L = np.load(f"results/benchmark/rr/labels_sil_frozen/{t['id']}.npz")
        for j in L.files:
            pseudo.setdefault((t['seq'], t['frames'][int(j)]), L[j])
    res, keep_imgs = {}, []
    for seq in ('seq1', 'seq2', 'seq3', 'seq4'):
        for f in sorted(os.listdir(f'data/nfo_processed/{seq}_gt')):
            if not f.endswith('_handmask_raw.png' if seq == 'seq1' else '_handAv.png'):
                continue
            k = int(f[:5]); lab = RP.hand_label(seq, k); box = BM.gt_boxes(seq).get(k)
            if lab is None or box is None:
                continue
            H, keep = lab
            ref = {'SAM2-ceiling': TR.gt_mask_native(k, seq), 'pseudo-GT': pseudo.get((seq, k))}
            for name, (h, lam) in VARIANTS.items():
                C = composite_at(seq, k, h, lam)
                ref[f'composite {name}'] = sam2_mask(pred, C, box)
                if name == 'h3 λ0.7' and len(keep_imgs) < 8 and k % 2 == 0:
                    keep_imgs.append((seq, k, C, ref[f'composite {name}'], ref['SAM2-ceiling'], H, box))
            for name, m in ref.items():
                if m is not None:
                    res.setdefault(name, []).append((seq, *score(m, H, keep)))
    for grp, sel in (('seq1 (brush)', ('seq1',)), ('seq2-4 (handAv)', ('seq2', 'seq3', 'seq4'))):
        print(f'\n{grp}\n{"label":24s}  n    J      P      R')
        for name, v in res.items():
            v = np.array([x[1:] for x in v if x[0] in sel], float)
            if len(v):
                print(f'{name:24s} {len(v):2d}  {v[:, 0].mean():.3f}  {v[:, 1].mean():.3f}  {v[:, 2].mean():.3f}')
    _montage(keep_imgs)


def _montage(items):
    rows = []
    for seq, k, C, mc, ms, H, b in items:
        x0, y0, x1, y1 = b; cx, cy, w, hh = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 60) * 2.2, (y1 - y0) * 1.5
        X0, Y0, X1, Y1 = int(max(cx - w / 2, 0)), int(max(cy - hh / 2, 0)), int(min(cx + w / 2, 800)), int(min(cy + hh / 2, 600))
        panes = []
        for title, m, col in (('frame t + hand (green)', H, (0, 255, 0)), ('composite + its SAM2 mask (red)', mc, (0, 0, 255)),
                              ('frame t + SAM2-ceiling (blue)', ms, (255, 128, 0))):
            base = (C if 'composite' in title else frame(seq, k)).copy()
            if m is not None:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(base, cs, -1, col, 2)
            p = cv2.resize(base[Y0:Y1, X0:X1], (220, 300))
            cv2.putText(p, f'{seq} {k}' if 'hand' in title else '', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            panes.append(p)
        rows.append(np.hstack(panes))
    os.makedirs(BM.IMG, exist_ok=True)
    cv2.imwrite(f'{BM.IMG}/composite_label_vs_hand.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
