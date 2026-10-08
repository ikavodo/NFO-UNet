"""Can a GT-free visible-part mask at t0 approach the GT mask on the hard starts? (2026-10-08)

    python -m benchmark.nfo_vos.diag_init_mask        # CPU + a few SAM2 image calls per start

Decomposition: visible(t0) = extent ∩ foreground(t0), with foreground = |frame t0 - person-free
background| (median of the trial's warm-up frames; static camera) above a threshold that is fixed
in advance: 'fixed' (20 grey levels) or 'otsu' (Otsu inside the extent). Extent arms:
  box      - the shared prompt box every method gets (GT-free in benchmark terms)
  a4       - SAM2 cut-out of the A/4 composite C0 (tracker velocity), box + p* prompt (GT-free)
  gtcomp   - SAM2 cut-out of the GT-aligned composite (ORACLE extent)
References: B0's and T1's own t0 masks, and the oracle ceiling (GT mask itself).
Hard starts = init failures (J(t0) < 0.5) of B0 or T1 in results/benchmark/rr/per_trial.csv.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

from benchmark.nfo_vos import a4, a4_build as B, metrics, run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

FIXED_THR = 20
OUT_CSV = f'{TR.RES}/diag_init_mask.csv'
TRACE = f'{TR.IMG}_checks/init_mask_test'


def visible_in_extent(frame, bg, extent, mode='fixed'):
    d = cv2.absdiff(frame, bg)
    if mode == 'otsu':
        vals = d[extent].reshape(-1, 1).astype(np.uint8)
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) if vals.size else (FIXED_THR, None)
    else:
        thr = FIXED_THR
    return (d > thr) & extent


def sam_cutout(img, trial):
    pred = R.image_predictor()
    pred.set_image(np.stack([img] * 3, -1))
    m, _, _ = pred.predict(point_coords=np.array([trial['point_native']]), point_labels=np.array([1]),
                           box=np.array(trial['box_native']), multimask_output=False)
    return m[0] > 0.5


def composites(trial):
    """C0 two ways: A/4 (tracker velocity, as built for the benchmark) and GT-box aligned."""
    seq, f = trial['seq'], trial['t0']
    ks = [k for k in range(f - a4.N + 1, f + 1) if k >= 0]
    raw = [V.native(k, seq) for k in ks]
    vx = B.velocity_at(R.track(trial)['chain'], 0, B.fallback_velocity(trial))
    c_free = a4.recency_fusion(raw, [vx * (f - k) for k in ks])

    def centre(k):
        b = TR.bbs(seq)
        if k not in b or not b[k] or b[k][0].x < 0:
            return None
        x0, y0, x1, y1 = gt_to_native(b[k][0], TR.NATIVE_W, TR.NATIVE_H)
        return (x0 + x1) / 2, (y0 + y1) / 2
    g0 = centre(f)
    keep = [(fr, centre(k)) for fr, k in zip(raw, ks) if centre(k) is not None]
    w = a4.recency_weights(a4.N)[-len(keep):]
    c_gt = a4.recency_fusion([fr for fr, _ in keep], [g0[0] - c[0] for _, c in keep], [g0[1] - c[1] for _, c in keep]) \
        if len(keep) == len(w) else raw[-1]
    return c_free, c_gt, vx, len(keep)


def main():
    T = {x['id']: x for x in json.load(open(TR.OUT)) if x['admissible']}
    pt = pd.read_csv(f'{TR.RES}/per_trial.csv')
    hard = sorted(set(pt[pt.method.isin(['b0', 't1']) & pt.init_fail].trial))
    os.makedirs(TRACE, exist_ok=True)
    rows = []
    with __import__('torch').inference_mode(), __import__('torch').autocast('cuda', dtype=__import__('torch').float16):
        for tid in hard:
            t = T[tid]; seq, f = t['seq'], t['t0']
            G = metrics.native_to_224(gt_mask_native(seq, f))
            frame = V.native(f, seq)
            w0, w1 = t['warmup']
            bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.uint8)
            x0, y0, x1, y1 = (int(round(v)) for v in t['box_native'])
            box = np.zeros(frame.shape, bool); box[max(0, y0):y1, max(0, x0):x1] = True
            c_free, c_gt, vx, n_gt = composites(t)
            ext = {'box': box, 'a4': sam_cutout(c_free, t), 'gtcomp': sam_cutout(c_gt, t)}
            j = lambda M: metrics.frame_metrics(metrics.native_to_224(M), G)['J']
            r = dict(trial=tid, seq=seq, v=t['v'], n_gt_hist=n_gt,
                     b0=j(np.load(f'{TR.RES}/masks/b0/{tid}.npz')['masks'][0]),
                     t1=j(np.load(f'{TR.RES}/masks/t1/{tid}.npz')['masks'][0]))
            masks = {}
            for e, E in ext.items():
                r[f'ext_{e}'] = j(E)
                for mode in ('fixed', 'otsu'):
                    M = visible_in_extent(frame, bg, E, mode)
                    r[f'{e}&fg_{mode}'] = j(M); masks[f'{e}&fg_{mode}'] = M
            rows.append(r)
            sl = (slice(max(0, y0 - 40), min(600, y1 + 40)), slice(max(0, x0 - 40), min(800, x1 + 40)))
            tiles = []
            for name, M in [('GT t0', gt_mask_native(seq, f)), ('T1 t0', np.load(f'{TR.RES}/masks/t1/{tid}.npz')['masks'][0]),
                            ('box&fg_otsu', masks['box&fg_otsu']), ('a4&fg_otsu', masks['a4&fg_otsu']), ('gtcomp&fg_otsu', masks['gtcomp&fg_otsu'])]:
                im = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR); im[M] = (0.4 * im[M] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
                im = cv2.resize(im[sl], (180, 240)); cv2.putText(im, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                tiles.append(im)
            cv2.imwrite(f'{TRACE}/{tid}.png', np.hstack(tiles))
            print(f"{tid}: T1 {r['t1']:.2f}  box&fg {r['box&fg_otsu']:.2f}  a4&fg {r['a4&fg_otsu']:.2f}  gtcomp&fg {r['gtcomp&fg_otsu']:.2f}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT_CSV, index=False)
    cols = ['b0', 't1'] + [c for c in d.columns if c.startswith('ext_') or '&fg_' in c]
    print(f'\nJ vs GT t0 mask (224), mean over {len(d)} hard starts:')
    print(d[cols].mean().round(3).to_string())
    print(f'-> {OUT_CSV}, traces {TRACE}/')


if __name__ == '__main__':
    main()
