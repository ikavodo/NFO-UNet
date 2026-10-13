"""Pixel-level background suppression of T1 (SAMURAI) masks: delete mask pixels where |frame - empty scene| < tau.
Cheap test of "static-background evidence removes leaks" before the feature-level (per-cell) version. 2026-10-13

    python -m benchmark.nfo_vos.bg_suppress

PRE-REGISTERED (written before running):
- Empty scene = per-pixel median of the trial's warm-up frames (person-free), grayscale, native 800x600.
- tau grid TAUS, chosen on seq1 by mean window J vs pseudo-GT. Applied unchanged to seq2-4.
- PASS: on seq2-4, J(T1+bg) - J(T1) vs pseudo-GT > 0 with 95% window-bootstrap CI excluding 0.
- Reference points: t1-om (oracle-init) gain; seq1 hand frames (polygon/brush) reported descriptively (P/R).
- Also reported: the same gains on windows that do NOT start on a confirmed-clear frame.
"""
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

TAUS = (4, 8, 12, 16, 24)
IMG = 'images/benchmark/rr_checks/bg_suppress'


def J(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def boot(x, B=10000):
    x = np.asarray(x); r = np.random.default_rng(0)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    clear = {}
    for seq in sorted({t['seq'] for t in T}):
        for s in TR.segments(seq):
            clear[(seq, s['idx'])] = set(TR.clear_frames(s))
    hand = {tag: {int(f[:5]): cv2.imread(f'data/nfo_processed/seq1_gt/{f}', 0) == 1
                  for f in os.listdir('data/nfo_processed/seq1_gt') if f.endswith(f'_{tag}.png')}
            for tag in ('handpoly', 'handmask_raw')}
    bgs, rows, hrows, keep = {}, [], {tag: [] for tag in hand}, {}
    for i, t in enumerate(T):
        seq, w = t['seq'], tuple(t['warmup'])
        if (seq, w) not in bgs:
            bgs[(seq, w)] = np.median(np.stack([V.native(k, seq) for k in range(w[0], w[1] + 1)]), 0).astype(np.float32)
        bg = bgs[(seq, w)]
        M = {m: np.load(f"{TR.RES}/masks/{m}/{t['id']}.npz")['masks'] for m in ('b0', 't1', 't1-om')}
        r = {m: [] for m in ('b0', 't1', 't1-om')} | {tau: [] for tau in TAUS}
        for j in range(1, len(t['frames']) - 1):
            k = t['frames'][j]
            P = gt_mask_native(seq, k)
            H = {tag: hand[tag].get(k) for tag in hand} if seq == 'seq1' else {}
            if (P is None or not P.any()) and not any(h is not None for h in H.values()):
                continue
            diff = np.abs(V.native(k, seq).astype(np.float32) - bg)
            sup = {tau: M['t1'][j] & (diff >= tau) for tau in TAUS}
            if P is not None and P.any():
                for m in M:
                    r[m].append(J(M[m][j], P))
                for tau in TAUS:
                    r[tau].append(J(sup[tau], P))
            for tag, h in H.items():
                if h is not None:
                    st = lambda a: (J(a, h), (a & h).sum() / max(a.sum(), 1), (a & h).sum() / h.sum())
                    hrows[tag].append([st(M['t1'][j]), st(M['t1-om'][j])] + [st(sup[tau]) for tau in TAUS])
                    if tag == 'handpoly':
                        keep.setdefault(k, (seq, j, M['t1'][j], sup, h, P))
        rows.append(dict(seq=seq, clear_start=t['frames'][0] in clear[(seq, t['seg_idx'])],
                         **{str(m): np.nanmean(v) for m, v in r.items()}))
        print(f'{i + 1}/{len(T)} {t["id"]}', flush=True)

    s1 = [x for x in rows if x['seq'] == 'seq1']
    print('\nseq1 (tuning): mean window J vs pseudo-GT')
    for m in ('b0', 't1', 't1-om') + TAUS:
        print(f'  {"t1+bg tau=" + str(m) if m in TAUS else m:12s} {np.mean([x[str(m)] for x in s1]):.3f}')
    tau = max(TAUS, key=lambda tt: np.mean([x[str(tt)] for x in s1]))
    print(f'chosen tau = {tau}')

    def gains(sub, name):
        print(f'\n{name} (n={len(sub)} windows): differences vs pseudo-GT, mean [95% CI]')
        for a, b in ((str(tau), 't1'), ('t1-om', 't1'), ('t1', 'b0')):
            x = np.array([w[a] for w in sub]) - np.array([w[b] for w in sub])
            lo, hi = boot(x)
            print(f"  {('t1+bg' if a == str(tau) else a)} - {b}: {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    test = [x for x in rows if x['seq'] != 'seq1']
    gains(test, 'TEST seq2-4')
    gains([x for x in test if not x['clear_start']], 'TEST seq2-4, windows not starting on a clear frame')
    gains(s1, 'seq1 (tuning, for reference)')
    ti = 2 + TAUS.index(tau)
    for tag, v in hrows.items():
        v = np.array(v)
        print(f'\nseq1 {tag} (n={len(v)} frames): J / P / R')
        for name, c in (('t1', 0), ('t1-om', 1), (f't1+bg tau={tau}', ti)):
            print(f'  {name:14s} ' + ' / '.join(f'{z:.3f}' for z in v[:, c].mean(0)))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = sorted(keep)[::max(1, len(keep) // 5)][:5]
    os.makedirs(IMG, exist_ok=True)
    fig, ax = plt.subplots(1, len(ks), figsize=(10, 4))
    for a_, k in zip(np.atleast_1d(ax), ks):
        seq, j, m, sup, h, P = keep[k]
        ys, xs = np.nonzero(m | h)
        y0, y1, x0, x1 = max(0, ys.min() - 15), min(600, ys.max() + 15), max(0, xs.min() - 15), min(800, xs.max() + 15)
        a_.imshow(V.native(k, seq)[y0:y1, x0:x1], cmap='gray')
        rm = (m & ~sup[tau])[y0:y1, x0:x1]
        ov = np.zeros(rm.shape + (4,)); ov[rm] = (1, 0, 0, 0.6)
        a_.imshow(ov)
        for mm, col in ((h, 'cyan'), (sup[tau], 'yellow')):
            a_.contour(mm[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=col, linewidths=0.8)
        a_.set_title(str(k)); a_.axis('off')
    fig.suptitle(f'cyan = your polygon, yellow = T1 after suppression (tau={tau}), red fill = T1 pixels removed '
                 f'(|frame - empty scene| < tau)', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{IMG}/t1_bg_suppress.png', dpi=80)
    print(f'-> {IMG}/t1_bg_suppress.png')


if __name__ == '__main__':
    main()
