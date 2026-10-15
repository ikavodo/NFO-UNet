"""Keyframe M_C + motion warp vs per-frame M_C, offline on cached seq1 features (2026-10-15).

Question: the composite (and its SAM2 mask M_C) changes slowly; can M_C be computed every K frames and shifted in
between by the independent blob tracker's velocity, without losing accuracy?
Per frame j: key(j) = latest keyframe (a new key once j − key ≥ K); M_C(j) ≈ shift_x(M_C(key), fx(j) − fx(key)),
fx = causal Theil–Sen fit of the tracker chain over local [j−12, j] (same motion model as the composite alignment).
The change map pb and T1 stay per frame (current-frame evidence). V̂_anch = T1 ∪ (M_C ∧ ¬T1 ∧ u > 0.85).
Control 'nowarp': M_C(key) reused unshifted (isolates the motion model).
Label: frozen automatic silhouette label (vhat_vs_silhouette.py). Window-level bootstrap, 15 windows.
Pre-registered: K is acceptable if ΔJ vs K = 1 ≥ −0.005 (point estimate) with its CI upper end ≥ 0.
"""
import json

import cv2
import numpy as np

from benchmark.nfo_vos import run_t4 as R, trials as TR
from benchmark.nfo_vos.factorised_gtfree import theil_at
from benchmark.nfo_vos.learned_thresh import add_anchored, anchors
from benchmark.nfo_vos.vhat_vs_silhouette import FDIR, LDIR, SEQ, U, st

KS = (1, 2, 4, 8, 16)


def shift_x(m, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(m.astype(np.uint8), M, m.shape[::-1], flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    arms = [f'K{k}' for k in KS] + [f'K{k}nowarp' for k in KS[1:]]
    W = {a: [] for a in arms}
    for t in T:
        L, F = dict(np.load(f'{LDIR}/{t["id"]}.npz')), np.load(f'{FDIR}/{t["id"]}.npz')
        MC, PB = F['MC'], F['pb'].astype(np.float32)
        M1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks'].astype(bool)
        Tk = R.track(t)
        js = sorted(int(j) for j in L)
        rows = {a: [] for a in arms}
        for K in KS:
            key = None
            for j in js:
                if key is None or j - key >= K or not MC[key].any():
                    key = j
                cs = [l for l in range(max(0, j - 12), j + 1) if l in Tk['chain']]
                fx = theil_at(cs, [TR.to_native(*Tk['chain'][l])[0] for l in cs], [j, key])
                dx = fx[j] - fx[key] if np.isfinite(fx[j] - fx[key]) else 0.0
                for a, mc in ((f'K{K}', shift_x(MC[key], dx) if key != j else MC[j]),
                              (f'K{K}nowarp', MC[key])):
                    if a not in rows:
                        continue
                    d = {'T1': M1[j], 'pb': PB[j], 'cand': mc & ~M1[j]}
                    d['mb'], d['sb'], d['mp'] = anchors(mc, M1[j], PB[j])
                    rows[a].append(st(M1[j] | add_anchored(d, U), L[str(j)]))
        for a in arms:
            W[a].append(np.mean(rows[a], 0))
        print(f"{t['id']}: {len(js)} frames", flush=True)
    rng = np.random.default_rng(0)
    print(f'\nseq1, {len(T)} windows, vs frozen silhouette label: J / P / R / F2, and ΔJ vs K1')
    for a in arms:
        dj = np.array(W[a])[:, 0] - np.array(W['K1'])[:, 0]
        bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(5000)]
        print(f'  {a:10s} ' + ' / '.join(f'{v:.3f}' for v in np.mean(W[a], 0)) +
              f'   ΔJ {dj.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]')


if __name__ == '__main__':
    main()
