"""Score every cached (method, trial) mask stack against the pseudo-GT (spec §5).

    python -m benchmark.nfo_vos.score                # NATIVE (default): per_frame.csv, per_trial.csv, montages
    python -m benchmark.nfo_vos.score --res 224      # legacy 224 scoring -> per_frame_224.csv, per_trial_224.csv

Reads the mask cache only - never re-infers.
- native (2026-10-12): native 800x600 stacks vs the native pseudo-GT (*_sammask_native.png, written by
  gen_nfo_pseudo_masks --native-only, the same SAM2 masks before the 224 downscale). Removes the ~3.6 px
  staircase of the upsampled 224 GT. 224 stacks (T4 family) are skipped.
- 224: native stacks are mapped to 224 with the GT's own downscale (metrics.native_to_224); 224 stacks
  pass through.
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos import metrics, trials as TR

RES = TR.RES
IMG = TR.IMG
HORIZONS = (10, 25, 50)          # secondary, descriptive trajectory readouts
INIT_FAIL_J = 0.5                 # standard IoU success threshold (OTB success rate, PASCAL)


def to_224(masks):
    if masks.shape[1:] == (TR.SIZE, TR.SIZE):
        return masks
    return np.stack([metrics.native_to_224(m) for m in masks])


def gt_and_box(idx, seq, res):
    bb = TR.bbs(seq)[idx][0]
    if res == 'native':
        from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
        return TR.gt_mask_native(idx, seq), tuple(gt_to_native(bb, TR.NATIVE_W, TR.NATIVE_H))
    return TR.gt_mask(idx, seq), (bb.x * TR.SIZE, bb.y * TR.SIZE, (bb.x + bb.w) * TR.SIZE, (bb.y + bb.h) * TR.SIZE)


def score_trial(method, trial, P, conf=None, res='native'):
    rows, Gs, Ps, boxes = [], [], [], []
    for dt, idx in enumerate(trial['frames']):
        G, box = gt_and_box(idx, trial['seq'], res)
        if G is None:
            continue
        m = metrics.frame_metrics(P[dt], G)
        rows.append(dict(method=method, trial=trial['id'], seq=trial['seq'], seg=trial['seg_idx'], t0=trial['t0'],
                         t=idx, dt=dt, v=trial['v'], n_frag=trial['n_frag'],
                         x=(box[0] + box[2]) / 2, cerr=metrics.centre_error_norm(P[dt], box),
                         pred_iou=conf['pred_iou'][dt] if conf else np.nan,
                         obj_score=conf['obj_score'][dt] if conf else np.nan, **m))
        Gs.append(G); Ps.append(P[dt]); boxes.append(box)
    J = {r['dt']: r['J'] for r in rows}
    # headline aggregates follow DAVIS semi-supervised: drop the first (prompted) and last frame
    # (davis2017-evaluation/davis2017/evaluation.py:85, all_gt_masks[:, 1:-1])
    last = len(trial['frames']) - 1
    keep = [i for i, r in enumerate(rows) if 0 < r['dt'] < last]
    Js = np.array([rows[i]['J'] for i in keep]); Fs = np.array([rows[i]['F'] for i in keep])
    dre, nre = metrics.dre_nre(np.stack([Ps[i] for i in keep]), np.stack([Gs[i] for i in keep]))
    J0 = J.get(0, np.nan)
    summary = dict(method=method, trial=trial['id'], seq=trial['seq'], seg=trial['seg_idx'], t0=trial['t0'],
                   **{k: trial.get(k) for k in ('dir', 'gait', 'half', 'phase', 'm', 'x0')},
                   v=trial['v'], n_frag=trial['n_frag'],
                   JF=float(np.mean((Js + Fs) / 2)), J_mean=float(Js.mean()), F_mean=float(Fs.mean()),
                   J_decay=metrics.decay(Js), F_decay=metrics.decay(Fs), DRE=dre, NRE=nre,
                   P_norm=metrics.pnorm(np.stack([Ps[i] for i in keep]), [boxes[i] for i in keep]),
                   J0=J0, init_fail=bool(J0 < INIT_FAIL_J),
                   **{f'J@{h}': J.get(h, np.nan) for h in HORIZONS},
                   n_empty_gt=int(sum(rows[i]['area_g'] == 0 for i in keep)))
    return rows, summary


def montage_dts(n):
    """Six frame offsets spread over a window of n frames, first and last included."""
    return [int(round(v)) for v in np.linspace(0, n - 1, 6)]


def montage(trial, stacks, path, res='native'):
    """Rows = methods, cols = montage_dts(window). Green = GT contour, red fill = prediction."""
    rows = []
    for method, P in stacks.items():
        tiles = []
        for dt in montage_dts(len(trial['frames'])):
            idx = trial['frames'][dt]
            src = (f"data/nfo_final/nfo_final/{trial['seq']}/{idx:05d}.jpg" if res == 'native'
                   else f"{TR.seq_dir(trial['seq'])}/{idx:05d}_or.jpg")
            img = cv2.cvtColor(cv2.imread(src, 0), cv2.COLOR_GRAY2BGR)
            img[P[dt]] = (0.45 * img[P[dt]] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            G = gt_and_box(idx, trial['seq'], res)[0]
            if G is not None:
                cs, _ = cv2.findContours(G.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (0, 255, 0), 1)
            if res == 'native':
                img = cv2.resize(img, (300, 225), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f'{method} f{idx} +{dt}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--montage-methods', default='b0,t1,t1-a4,t4-1n')
    p.add_argument('--res', choices=('native', '224'), default='native')
    a = p.parse_args()
    suffix = '' if a.res == 'native' else '_224'
    wanted = a.montage_methods.split(',')
    img_dir = IMG if a.res == 'native' else f'{IMG}_224' 
    trials = {t['id']: t for t in json.load(open(TR.OUT)) if t['admissible']}
    methods = sorted(os.listdir(f'{RES}/masks'))
    all_rows, summaries, stacks = [], [], {}
    for method in methods:
        for path in sorted(glob.glob(f'{RES}/masks/{method}/*.npz')):
            tid = os.path.basename(path)[:-4]
            z = np.load(path)
            if a.res == 'native':
                P = z['masks']
                if P.shape[1:] != (TR.NATIVE_H, TR.NATIVE_W):
                    continue                    # 224 stacks (T4 family) are not scored at native
            else:
                P = to_224(z['masks'])
            conf = {k: z[k] for k in ('pred_iou', 'obj_score')} if 'pred_iou' in z else None
            rows, s = score_trial(method, trials[tid], P, conf, a.res)
            if not rows:
                raise SystemExit(f'no GT for {tid} at res={a.res} (native GT missing? run gen_nfo_pseudo_masks --native-only)')
            all_rows += rows; summaries.append(s)
            if method in wanted:              # keep only montage stacks (native stacks are large)
                stacks.setdefault(tid, {})[method] = P
    for name, data in (('per_frame', all_rows), ('per_trial', summaries)):
        with open(f'{RES}/{name}{suffix}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    os.makedirs(img_dir, exist_ok=True)
    for tid, st in stacks.items():
        montage(trials[tid], {m: st[m] for m in wanted if m in st}, f'{img_dir}/{tid}.png', a.res)
    print(f'{len(summaries)} (method, trial) pairs at res={a.res}, methods {methods} -> {RES}/per_*{suffix}.csv, {img_dir}/')


if __name__ == '__main__':
    main()
