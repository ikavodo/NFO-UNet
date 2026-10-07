"""Score every cached (method, trial) mask stack against the 224 pseudo-GT (spec §5).

    python -m benchmark.nfo_vos.score     # per_frame.csv, per_trial.csv, montages

Reads the mask cache only - never re-infers. Native-resolution stacks (B0, T1) are mapped to 224
with the GT's own downscale (metrics.native_to_224); 224 stacks (T4 family) pass through.
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


def score_trial(method, trial, P, conf=None):
    rows, Gs, Ps, boxes = [], [], [], []
    for dt, idx in enumerate(trial['frames']):
        G = TR.gt_mask(idx, trial['seq'])
        if G is None:
            continue
        bb = TR.bbs(trial['seq'])[idx][0]
        box = (bb.x * TR.SIZE, bb.y * TR.SIZE, (bb.x + bb.w) * TR.SIZE, (bb.y + bb.h) * TR.SIZE)
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


def montage(trial, stacks, path):
    """Rows = methods, cols = montage_dts(window). Green = GT contour, red fill = prediction."""
    rows = []
    for method, P in stacks.items():
        tiles = []
        for dt in montage_dts(len(trial['frames'])):
            idx = trial['frames'][dt]
            img = cv2.cvtColor(cv2.imread(f"{TR.seq_dir(trial['seq'])}/{idx:05d}_or.jpg", 0), cv2.COLOR_GRAY2BGR)
            img[P[dt]] = (0.45 * img[P[dt]] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            G = TR.gt_mask(idx, trial['seq'])
            if G is not None:
                cs, _ = cv2.findContours(G.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (0, 255, 0), 1)
            cv2.putText(img, f'{method} f{idx} +{dt}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--montage-methods', default='b0,t1,t1-a4,t4-1n')
    a = p.parse_args()
    trials = {t['id']: t for t in json.load(open(TR.OUT)) if t['admissible']}
    methods = sorted(os.listdir(f'{RES}/masks'))
    all_rows, summaries, stacks = [], [], {}
    for method in methods:
        for path in sorted(glob.glob(f'{RES}/masks/{method}/*.npz')):
            tid = os.path.basename(path)[:-4]
            z = np.load(path)
            P = to_224(z['masks'])
            conf = {k: z[k] for k in ('pred_iou', 'obj_score')} if 'pred_iou' in z else None
            rows, s = score_trial(method, trials[tid], P, conf)
            all_rows += rows; summaries.append(s)
            stacks.setdefault(tid, {})[method] = P
    for name, data in (('per_frame', all_rows), ('per_trial', summaries)):
        with open(f'{RES}/{name}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    os.makedirs(IMG, exist_ok=True)
    wanted = a.montage_methods.split(',')
    for tid, st in stacks.items():
        montage(trials[tid], {m: st[m] for m in wanted if m in st}, f'{IMG}/{tid}.png')
    print(f'{len(summaries)} (method, trial) pairs, methods {methods} -> {RES}/per_*.csv, {IMG}/')


if __name__ == '__main__':
    main()
