"""G2 verdict (plan docs/superpowers/plans/2026-11-19-layer-sam-feature-integral.md, Task 8): learned feature integral vs
the pixel adapter on seq1 WHOLE CROSSINGS (NFO_RUN=fullpass, clips 0–7), one environment (masks dumped with MASK_TAG=@lap).

  box AUC  per crossing, from fullpass/downstream/box_gt.csv (run box_metrics score first)
  hand J   per seq1 hand frame (20 brush masks, no void), from the crossing's mask at that frame
Paired bootstrap (2,000 resamples, seed 0) of NEW − REF over crossings (AUC) and over hand frames (J).
Pre-registered PASS: mean(NEW) ≥ mean(REF) − 0.02 on both.

    NFO_RUN=fullpass PYTHONPATH=. python -m benchmark.nfo_vos.eval.g2_score [NEW] [REF] [OTHER ...]
"""
import csv
import json
import os
import sys

import numpy as np

from benchmark.nfo_vos import trials as TR


def hand_j(arm):
    trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    out = {}
    for k in TR.hand_frames('seq1'):
        t = next((t for t in trials if k in t['frames']), None)
        p = f'{TR.RES}/masks/{arm}/{t["id"]}.npz' if t else None
        if not p or not os.path.exists(p):
            continue
        m = np.load(p)['masks'][t['frames'].index(k)]
        H = TR.hand_label('seq1', k)[0]
        out[k] = (m & H).sum() / max((m | H).sum(), 1)
    return out


def box_auc(arm):
    rows = csv.DictReader(open(f'{TR.RES}/downstream/box_gt.csv'))
    return {r['id']: float(r['auc']) for r in rows if r['arm'] == arm and r['seq'] == 'seq1'}


def paired(a, b, seed=0):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(keys)


def main(new='adapter-feat@lap', ref='adapter-pf@lap', *others):
    arms = [new, ref, *others]
    J = {a: hand_j(a) for a in arms}
    A = {a: box_auc(a) for a in arms}
    print(f"{'arm':>20} | box AUC (n crossings) | hand J (n frames)")
    for a in arms:
        print(f'{a:>20} | {np.mean(list(A[a].values())) if A[a] else float("nan"):.3f} ({len(A[a])})'
              f'            | {np.mean(list(J[a].values())) if J[a] else float("nan"):.3f} ({len(J[a])})')
    if not (A[new] and J[new]):
        return print(f'{new}: no masks / box scores yet')
    da, dj = paired(A[new], A[ref]), paired(J[new], J[ref])
    print(f'{new} − {ref}: box AUC {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}] (n={da[3]}) | '
          f'hand J {dj[0]:+.3f} [{dj[1]:+.3f}, {dj[2]:+.3f}] (n={dj[3]})')
    ok = da[0] >= -0.02 and dj[0] >= -0.02
    print(f"G2 {'PASS' if ok else 'FAIL'} (rule: mean difference ≥ −0.02 on both)")


if __name__ == '__main__':
    main(*sys.argv[1:])
