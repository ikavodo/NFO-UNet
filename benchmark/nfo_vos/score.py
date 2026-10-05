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

RES = 'results/benchmark/pilot'
IMG = 'images/benchmark/pilot'
MONTAGE_DT = (0, 10, 20, 30, 40, 50)
HORIZONS = (10, 25, 50)


def to_224(masks):
    if masks.shape[1:] == (TR.SIZE, TR.SIZE):
        return masks
    return np.stack([metrics.native_to_224(m) for m in masks])


def score_trial(method, trial, P):
    rows, Gs, Ps, boxes = [], [], [], []
    for dt, idx in enumerate(trial['frames']):
        G = TR.gt_mask(idx)
        if G is None:
            continue
        bb = TR.bbs()[idx][0]
        box = (bb.x * TR.SIZE, bb.y * TR.SIZE, (bb.x + bb.w) * TR.SIZE, (bb.y + bb.h) * TR.SIZE)
        m = metrics.frame_metrics(P[dt], G)
        rows.append(dict(method=method, trial=trial['id'], seg=trial['seg_idx'], t0=trial['t0'],
                         t=idx, dt=dt, v=trial['v'], n_frag=trial['n_frag'],
                         cerr=metrics.centre_error_norm(P[dt], box), **m))
        Gs.append(G); Ps.append(P[dt]); boxes.append(box)
    J = {r['dt']: r['J'] for r in rows}
    Js = np.array([r['J'] for r in rows]); Fs = np.array([r['F'] for r in rows])
    dre, nre = metrics.dre_nre(np.stack(Ps), np.stack(Gs))
    summary = dict(method=method, trial=trial['id'], seg=trial['seg_idx'], t0=trial['t0'],
                   v=trial['v'], n_frag=trial['n_frag'], **{f'J@{h}': J.get(h, np.nan) for h in HORIZONS},
                   JF=float(np.mean((Js + Fs) / 2)), J_mean=float(Js.mean()),
                   J_decay=metrics.decay(Js), F_decay=metrics.decay(Fs), DRE=dre, NRE=nre,
                   P_norm=metrics.pnorm(np.stack(Ps), boxes),
                   n_empty_gt=int(sum(r['area_g'] == 0 for r in rows)))
    return rows, summary


def montage(trial, stacks, path):
    """Rows = methods, cols = MONTAGE_DT. Green = GT contour, red fill = prediction."""
    rows = []
    for method, P in stacks.items():
        tiles = []
        for dt in MONTAGE_DT:
            idx = trial['frames'][dt]
            img = cv2.cvtColor(cv2.imread(f'{TR.SEQ_DIR}/{idx:05d}_or.jpg', 0), cv2.COLOR_GRAY2BGR)
            img[P[dt]] = (0.45 * img[P[dt]] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            G = TR.gt_mask(idx)
            if G is not None:
                cs, _ = cv2.findContours(G.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (0, 255, 0), 1)
            cv2.putText(img, f'{method} f{idx} +{dt}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    cv2.imwrite(path, np.vstack(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--montage-methods', default='b0,t1,t4,t4-1,t4-2*,t4-3')
    a = p.parse_args()
    trials = {t['id']: t for t in json.load(open(TR.OUT)) if t['admissible']}
    methods = sorted(os.listdir(f'{RES}/masks'))
    all_rows, summaries, stacks = [], [], {}
    for method in methods:
        for path in sorted(glob.glob(f'{RES}/masks/{method}/*.npz')):
            tid = os.path.basename(path)[:-4]
            P = to_224(np.load(path)['masks'])
            rows, s = score_trial(method, trials[tid], P)
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
