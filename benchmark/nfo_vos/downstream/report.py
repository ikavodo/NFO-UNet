"""Downstream benchmark table (docs/vos_benchmark/plan_downstream_benchmark.md, Task 5).

Hand-label J / P / R per arm: every (window, hand frame inside the window) pair; seq1 brush masks (selection), seq2–4
handAv with void (value 2) ignored (single test). Paired window-bootstrap Δ for the named pairs, on box AUC
(results/benchmark/rr/downstream/box_gt.csv), recognition confidence (recog_N7_uniform_meanfill.csv) and hand J.

  python -m benchmark.nfo_vos.downstream.report
"""
import csv
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos.downstream import box_metrics as BM

ARMS = ('b0-c2', 't1-c2', 'trkbox', 'trk', 'adapter-pf', 'adapter-bal', 'adapter-pf-noC', 'vid-plain', 'vid-prompts', 'vid-adapter-pre',
        'vid-adapter-noC', 'vid-oracle-ceil', 'vid-oracle-pseudo')
PAIRS = (('adapter-bal', 'adapter-pf'), ('adapter-bal', 't1-c2'), ('vid-adapter-pre', 't1-c2'), ('vid-adapter-pre', 'trkbox'), ('vid-adapter-pre', 'trk'),
         ('vid-adapter-pre', 'vid-prompts'), ('vid-adapter-pre', 'vid-adapter-noC'), ('vid-prompts', 'vid-plain'),
         ('adapter-pf', 'trk'), ('trk', 't1-c2'), ('adapter-pf', 't1-c2'), ('adapter-pf', 'adapter-pf-noC'),
         ('vid-oracle-ceil', 'vid-prompts'), ('vid-oracle-ceil', 'adapter-pf'), ('vid-oracle-pseudo', 'adapter-pf'))
GROUPS = (('seq1 (selection)', ('seq1',)), ('seq2-4 (test)', ('seq2', 'seq3', 'seq4')))


def hand_label(seq, k):
    """(person bool, keep bool) or None."""
    d = f'data/nfo_processed/{seq}_gt'
    if seq == 'seq1':
        p = f'{d}/{k:05d}_handmask_raw.png'
        return None if not os.path.exists(p) else (cv2.imread(p, 0) > 0, None)
    p = f'{d}/{k:05d}_handAv.png'
    if not os.path.exists(p):
        return None
    v = cv2.imread(p, 0)
    return v == 1, v != 2


def hand_rows(arms=ARMS):
    trials = json.load(open('results/benchmark/rr/trials.json'))
    rows = []
    for t in trials:
        lab = {k: hand_label(t['seq'], k) for k in t['frames']}
        lab = {k: v for k, v in lab.items() if v is not None}
        if not lab:
            continue
        for a in arms:
            la = BM.load_arm(a, t['id'])
            if la is None:
                continue
            ms, fr = la
            fr = [int(x) for x in fr]
            for k, (H, keep) in lab.items():
                m = ms[fr.index(k)]
                if keep is not None:
                    m, H = m & keep, H & keep
                tp = (m & H).sum()
                rows.append(dict(arm=a, id=t['id'], seq=t['seq'], frame=k, J=tp / max((m | H).sum(), 1),
                                 P=tp / max(m.sum(), 1), R=tp / max(H.sum(), 1)))
    return rows


def per_window(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r['arm'], {}).setdefault(r['id'], []).append(float(r[key]))
    return {a: {i: np.mean(v) for i, v in d.items()} for a, d in out.items()}


def seq_of(rows):
    return {r['id']: r['seq'] for r in rows}


def delta(win, a, b, ids, B=2000):
    ids = [i for i in ids if i in win.get(a, {}) and i in win.get(b, {})]
    d = np.array([win[a][i] - win[b][i] for i in ids])
    bs = d[np.random.default_rng(0).integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.quantile(bs, .025), np.quantile(bs, .975), len(d)


def main():
    hr = hand_rows()
    box = list(csv.DictReader(open(f'{BM.OUT}/box_gt.csv')))
    rec = list(csv.DictReader(open(f"{BM.OUT}/recog_N7_uniform_meanfill{os.environ.get('NFO_RECOG_TAG', '')}.csv")))
    sq = {**seq_of(box), **seq_of(rec), **seq_of(hr)}
    W = {'hand J': per_window(hr, 'J'), 'hand P': per_window(hr, 'P'), 'hand R': per_window(hr, 'R'),
         'box AUC': per_window(box, 'auc'), 'recog conf': per_window(rec, 'conf')}
    for name, sel in GROUPS:
        print(f'\n== {name}\n{"arm":17s}' + ''.join(f'{m:>11s}' for m in W))
        arms = list(ARMS) + [a for a in ('nomask', 'sam2ceil', 'pseudo') if a in W['recog conf']]
        for a in arms:
            if a not in W['box AUC']:
                continue
            vals = []
            for m in W:
                v = [x for i, x in W[m].get(a, {}).items() if sq[i] in sel]
                vals.append(f'{np.mean(v):11.3f}' if v else f'{"—":>11s}')
            print(f'{a:17s}' + ''.join(vals))
        print('paired Δ (window bootstrap 95% CI):')
        for a, b in PAIRS:
            parts = []
            if a not in W['box AUC'] or b not in W['box AUC']:
                continue
            for m in ('hand J', 'box AUC', 'recog conf'):
                ids = [i for i in sq if sq[i] in sel]
                mu, lo, hi, n = delta(W[m], a, b, ids)
                parts.append(f'{m} {mu:+.3f} [{lo:+.3f},{hi:+.3f}] n{n}')
            print(f'  {a} − {b}:  ' + ' | '.join(parts))
    with open(f'{BM.OUT}/hand_rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hr[0])); w.writeheader(); w.writerows(hr)


if __name__ == '__main__':
    main()
