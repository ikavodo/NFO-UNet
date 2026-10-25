"""The benchmark table (design: docs/vos_benchmark/plan_downstream_benchmark.md, Task 5).

Hand-label J / P / R per arm: every (window, hand frame inside the window) pair; seq1 brush masks (no void class),
seq2–4 handAv with void (value 2) ignored. Paired window-bootstrap Δ for the named pairs, on box AUC
(results/benchmark/rr/downstream/box_gt.csv), recognition confidence (recog_N<NFO_RECOG_N>_uniform_meanfill.csv) and
hand J. Groups: all sequences (the headline since 2026-10-23), seq1 (where step 200 was selected, so not independent)
and seq2–4 (the clean held-out check).
Recognition N = 5 frames per recognition stack (NFO_RECOG_N): the N at which SAM2.1 video and SAMURAI score highest in the
N sweep (eval/recog_vs_N.py), i.e. the baselines' best case. It replaced the a-priori N = 7 on 2026-10-23.

  python -m benchmark.nfo_vos.eval.report
"""
import csv
import json
import os

import numpy as np

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.eval import box_metrics as BM

# final arms: baselines B0 (SAM2.1 video) / T1 (SAMURAI), the tracker controls, the adapter over 3 seeds and its
# no-integral-image controls (the integral image's contribution); b0trk-c2 = SAM2.1 video + tracker prompts (memory on
# top of trk; the tracker on top of memory), full-pass protocol only
SEEDS = ('adapter-pf', 'adapter-s1', 'adapter-s2')
ARMS = ('b0-c2', 't1-c2', 'b0trk-c2', 'trkbox', 'trk', *SEEDS, *(f'{a}-noC' for a in SEEDS))
PAIRS = (('trk', 't1-c2'), ('b0trk-c2', 'trk'), ('b0trk-c2', 'b0-c2'), *((a, 't1-c2') for a in SEEDS), ('adapter-pf', 'trk'), *((a, f'{a}-noC') for a in SEEDS))
# extra arms (e.g. history variants adapter-pf-h4): appended to the table, paired with their base arm
EXTRA = tuple(os.environ.get('NFO_EXTRA_ARMS', '').split())
ARMS, PAIRS = ARMS + EXTRA, PAIRS + tuple((a, a.rsplit('-h', 1)[0]) for a in EXTRA if '-h' in a)
GROUPS = (('seq1-4 (all)', TR.ALL_SEQS), ('seq1 (selection)', ('seq1',)), ('seq2-4 (test)', ('seq2', 'seq3', 'seq4')))
RECOG_N = int(os.environ.get('NFO_RECOG_N', 5))


def hand_rows(arms=ARMS):
    trials = json.load(open(TR.OUT))
    rows = []
    for t in trials:
        lab = {k: TR.hand_label(t['seq'], k) for k in t['frames']}
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
    rec = list(csv.DictReader(open(f"{BM.OUT}/recog_N{RECOG_N}_uniform_meanfill{os.environ.get('NFO_RECOG_TAG', '')}.csv")))
    sq = {**seq_of(box), **seq_of(rec), **seq_of(hr)}
    W = {'hand J': per_window(hr, 'J'), 'hand P': per_window(hr, 'P'), 'hand R': per_window(hr, 'R'),
         'box AUC': per_window(box, 'auc'), 'recog conf': per_window(rec, 'conf')}
    for name, sel in GROUPS:
        print(f'\n== {name}\n{"arm":17s}' + ''.join(f'{m:>11s}' for m in W))
        arms = list(ARMS) + [a for a in ('nomask', 'sam2ceil') if a in W['recog conf']]
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
