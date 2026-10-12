"""Is re-initialisation from the causal blob tracker sound? Two CPU checks on the rr run (2026-10-12).

    python -m benchmark.nfo_vos.diag_reinit tracker   # A: is the tracker's prompt point on the person when T1 fails?
    python -m benchmark.nfo_vos.diag_reinit t0cands [all]  # B: do SAM2's 3 candidates at t0 contain a good mask?

A  Per frame t >= 1 of every admissible window: the tracker's re-prompt point p(t) (deepest pixel of
   the merged person blob, exactly as T4-1n builds it, run_t4.py:161-166), its lock L(t) (followed
   track detected at t and in >= 3 of the last 5 frames), whether p(t) lies on the GT (visible-part)
   mask and on T1's mask, T1's J and SAM2 predicted IoU q(t), and d(t) = distance of T1's mask
   centroid to the tracker's whole-person centre / person height.
B  Hard starts (B0 or T1 init failure): SAM2 image predictor on the native t0 frame, the shared
   box + point prompt, multimask_output=True -> J of each of the 3 candidates vs the GT t0 mask.
   Bounds what choosing a candidate at t0 ("warm start") can buy without re-prompting.
"""
import json
import sys

import cv2
import numpy as np
import pandas as pd

from benchmark.nfo_vos import metrics, run_t4 as R, trials as TR
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from tracking.core.blob_tracker import merged_center

TRACE = f'{TR.IMG}_checks/reinit'
LOCK_N, LOCK_K = 5, 3


def trials():
    return {t['id']: t for t in json.load(open(TR.OUT)) if t['admissible']}


def hard_ids():
    pt = pd.read_csv(f'{TR.RES}/per_trial.csv')
    return sorted(set(pt[pt.method.isin(['b0', 't1']) & pt.init_fail].trial))


def tracker_rows(t):
    P = R.track(t)
    masks, dets, kw, chain = P['masks'], P['dets'], P['kw'], P['chain']
    T1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")
    h_nat = (t['box_native'][3] - t['box_native'][1])
    rows = []
    for k in range(1, len(P['frames'])):
        idx = t['frames'][k]
        ax, ay, _ = R.readout_line(chain, k)
        cx, cy = merged_center(dets[k], ax, ay, kw['merge_radius'])
        blob = R.restrict_to_nearby((masks[k] > 0).astype(np.uint8), masks[k], dets[k], cx, cy,
                                    kw['merge_radius']) > 0
        p224, _ = R.prompt_from_blob(blob)
        px, py = TR.to_native(*(p224 if p224 is not None else (cx, cy)))
        G = gt_mask_native(t['seq'], idx)
        M = T1['masks'][k]
        inside = lambda A: bool(A[int(np.clip(round(py), 0, A.shape[0] - 1)), int(np.clip(round(px), 0, A.shape[1] - 1))])
        ccx, ccy = TR.to_native(cx, cy)
        ys, xs = np.nonzero(M)
        d = np.hypot(xs.mean() - ccx, ys.mean() - ccy) / h_nat if len(xs) else np.nan
        lock = k in chain and sum(j in chain for j in range(k - LOCK_N + 1, k + 1)) >= LOCK_K
        m224, g224 = metrics.native_to_224(M), metrics.native_to_224(G)
        rows.append(dict(trial=t['id'], seq=t['seq'], t=k, lock=lock, blob=p224 is not None,
                         p_in_gt=inside(G), p_in_t1=inside(M), d=d, q=float(T1['pred_iou'][k]),
                         J=metrics.frame_metrics(m224, g224)['J'], gt_empty=not G.any()))
    return rows


def auc(score, y):
    from scipy.stats import mannwhitneyu
    s, y = np.asarray(score), np.asarray(y, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    return mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())


def part_tracker():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    T, hard = trials(), set(hard_ids())
    rows = []
    for i, tid in enumerate(T):
        rows += tracker_rows(T[tid])
        print(f'{i + 1}/{len(T)} {tid}', flush=True)
    d = pd.DataFrame(rows)
    d = d[~d.gt_empty]
    d['hard'] = d.trial.isin(hard)
    d.to_csv(f'{TR.RES}/diag_reinit_tracker.csv', index=False)
    fail = d.J < 0.5
    print(f'\nframes {len(d)} (hard-start frames {d.hard.sum()}), T1 J<0.5 on {fail.mean():.3f}')
    print(f'lock rate {d.lock.mean():.2f}; P(p in GT | lock) {d[d.lock].p_in_gt.mean():.2f}, | no lock {d[~d.lock].p_in_gt.mean():.2f}')
    need = d[fail]
    print(f'T1-failure frames: lock {need.lock.mean():.2f}; P(p in GT | lock) {need[need.lock].p_in_gt.mean():.2f}; '
          f'P(p in T1 mask | lock) {need[need.lock].p_in_t1.mean():.2f}')
    for tau in (0.5, 0.6, 0.7):
        trig = d[(d.q < tau) & d.lock]
        print(f'trigger q<{tau} & lock: {len(trig)} frames ({len(trig) / len(d):.3f}), P(J<.5) {(trig.J < .5).mean():.2f}, '
              f'P(p in GT) {trig.p_in_gt.mean():.2f}, P(p in GT & not in T1) {(trig.p_in_gt & ~trig.p_in_t1).mean():.2f}')
    print(f'AUC for J<0.5: 1-q {auc(-d.q, fail):.3f}; d {auc(d.d, fail):.3f}; '
          f'd on q>=0.7 frames {auc(d[d.q >= .7].d, d[d.q >= .7].J < .5):.3f} (n fail {int((d[d.q >= .7].J < .5).sum())})')
    import os; os.makedirs(TRACE, exist_ok=True)
    for tid in sorted(hard):
        s = d[d.trial == tid]
        fig, ax = plt.subplots(figsize=(9, 2.4), dpi=80)
        ax.fill_between(s.t, 0, 1, where=s.lock, color='0.88', step='mid', label='tracker locked L(t)')
        ax.plot(s.t, s.J, 'k-', label='T1 J(t)')
        ax.plot(s.t, s.q, '-', color='tab:orange', label='SAM2 q(t)')
        ax.plot(s.t, np.clip(s.d, 0, 1), '-', color='tab:purple', label='d(t) (clip 1)')
        ax.plot(s.t[s.p_in_gt], np.full(s.p_in_gt.sum(), 1.04), 'g|', ms=6, label='p(t) on GT')
        ax.plot(s.t[~s.p_in_gt], np.full((~s.p_in_gt).sum(), 1.04), 'r|', ms=6, label='p(t) off GT')
        ax.set_ylim(0, 1.08); ax.set_xlabel('frame after t0'); ax.set_title(tid, fontsize=9)
        ax.legend(fontsize=6, ncol=6, loc='lower left')
        fig.tight_layout(); fig.savefig(f'{TRACE}/timeline_{tid}.png'); plt.close(fig)


def part_t0cands(which='hard'):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml', R.CKPT, device='cpu'))
    T, rows = trials(), []
    import os; os.makedirs(TRACE, exist_ok=True)
    ids = hard_ids() if which == 'hard' else sorted(T)
    for tid in ids:
        t = T[tid]; f = t['t0']
        G = gt_mask_native(t['seq'], f); g224 = metrics.native_to_224(G)
        img = cv2.imread(f"data/nfo_final/nfo_final/{t['seq']}/{f:05d}.jpg")
        with torch.inference_mode():
            pred.set_image(img[:, :, ::-1].copy())
            r = dict(trial=tid)
            for name, box in (('box+pt', np.asarray(t['box_native'])), ('pt', None)):
                C, q, _ = pred.predict(point_coords=np.array([t['point_native']]), point_labels=np.array([1]),
                                       box=box, multimask_output=True)
                J = [metrics.frame_metrics(metrics.native_to_224(c > 0.5), g224)['J'] for c in C]
                r.update({f'{name}_J{i}': J[i] for i in range(3)})
                r[f'{name}_qpick'], r[f'{name}_best'] = J[int(np.argmax(q))], max(J)
                if name == 'box+pt':
                    cands, qs, Js = C, q, J
        r['t1'] = metrics.frame_metrics(metrics.native_to_224(np.load(f'{TR.RES}/masks/t1/{tid}.npz')['masks'][0]), g224)['J']
        rows.append(r)
        x0, y0, x1, y1 = (int(round(v)) for v in t['box_native'])
        sl = (slice(max(0, y0 - 40), min(600, y1 + 40)), slice(max(0, x0 - 40), min(800, x1 + 40)))
        tiles = []
        for name, M in [('GT t0', G)] + [(f'cand{i} q{qs[i]:.2f} J{Js[i]:.2f}', cands[i] > 0.5) for i in range(3)]:
            im = img.copy(); im[M] = (0.4 * im[M] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            im = cv2.resize(im[sl], (180, 240)); cv2.putText(im, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)
            tiles.append(im)
        if which == 'hard':
            cv2.imwrite(f'{TRACE}/t0cands_{tid}.png', np.hstack(tiles))
        print(f"{tid}: T1 {r['t1']:.2f}  box+pt cands {[round(j, 2) for j in Js]} best {r['box+pt_best']:.2f}  "
              f"pt-only best {r['pt_best']:.2f}", flush=True)
    d = pd.DataFrame(rows); d['hard'] = d.trial.isin(hard_ids()); d.to_csv(f'{TR.RES}/diag_reinit_t0cands_{which}.csv', index=False)
    print(d.drop(columns='trial').groupby('hard').mean().round(3).T.to_string())


if __name__ == '__main__':
    {'tracker': part_tracker, 't0cands': part_t0cands}[sys.argv[1]](*sys.argv[2:])
