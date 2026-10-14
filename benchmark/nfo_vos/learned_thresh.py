"""Learned change threshold for V̂_T1, and first scoring on seq2–4 human labels (2026-10-14, fixed before scoring).

V̂_T1(k) = (M_C(k) ∧ add(x)) ∪ T1(k). M_C, composite, alignment and prompt exactly as run_vhat.py (frozen GT-free arm:
blob-tracker alignment, blob-tracker point). Only the per-pixel decision add(x) on candidates M_C ∧ ¬T1 changes.

pb(x) = log(1 + 5×5 mean (R_k − B)²), B = warm-up median. Per-frame anchors, all GT-free:
  μ_P = median pb over T1(k)                         (what the person's change level looks like in this frame)
  μ_B, σ_B = median, 1.4826·MAD of pb outside dilate(M_C ∪ T1, 31 px)   (background change level and noise)
Arms (all parameters fitted on seq1 brush labels only, leave-one-frame-out for the seq1 numbers):
  z_fixed   pixz(pb) > 7.5                            (frozen rule; transfer failure logged 2026-10-14)
  anchored  u = (pb − μ_B) / (μ_P − μ_B) > u*         (one parameter u*, grid search on mean J vs brush)
  logistic  σ(w·[pb, μ_B, σ_B, μ_P] + b) > 0.5        (per-pixel logistic on candidate pixels; the normalisation
                                                       itself is learned)
Selection: arm with best seq1 LOFO mean J vs brush. Test: seq2–4 handA (user's labels, tier A, local index 20),
ΔJ(arm − T1), frame bootstrap; also per sequence. Also reported there: B0, SAM2-ceiling, and size-binned recall of
the hand label's connected components (bins in stride-16 cells, as h2_fragments.py).
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import stats
from benchmark.nfo_vos.factorised_extras import pixz
from benchmark.nfo_vos.factorised_gtfree import composite, theil_at
from tracking.core.blob_tracker import merged_center

OUT = f'{TR.RES}/learned_thresh'
CELL = 16 * 800 / 1024   # one stride-16 cell in native px (SAM2 input 1024 wide)
UGRID = np.round(np.arange(0.0, 1.21, 0.05), 2)


def gt(seq, k, tag):
    p = f'data/nfo_processed/{seq}_gt/{k:05d}_{tag}.png'
    return cv2.imread(p, 0) == 1 if os.path.exists(p) else None


def features(t, j, Tk, bg, pred, M1):
    seq, k = t['seq'], t['frames'][j]
    cs = [l for l in range(max(0, j - 12), j + 1) if l in Tk['chain']]
    if len(cs) < 2:
        return None
    fx = theil_at(cs, [TR.to_native(*Tk['chain'][l])[0] for l in cs], [j] + list(range(j - 6, j)))
    C = composite(seq, k, {t['frames'][l]: (fx[j] - fx[l], 0.0) for l in range(j - 6, j)}, bg)
    anchor = Tk['chain'].get(j) or tuple(R.readout_line(Tk['chain'], j)[:2])
    pt = TR.to_native(*merged_center(Tk['dets'][j], *anchor, Tk['kw']['merge_radius']))
    with torch.inference_mode():
        pred.set_image(np.stack([np.clip(C, 0, 255).astype(np.uint8)] * 3, -1))
        m, _, _ = pred.predict(point_coords=np.array([pt]), point_labels=np.array([1]), multimask_output=False)
    return m[0] > 0, np.log(cv2.boxFilter((V.native(k, seq).astype(np.float32) - bg) ** 2, -1, (5, 5)) + 1)


def anchors(MC, T1, pb):
    far = ~cv2.dilate((MC | T1).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    mb = np.median(pb[far]); sb = 1.4826 * np.median(np.abs(pb[far] - mb))
    mp = np.median(pb[T1]) if T1.sum() > 50 else np.nan
    return mb, max(sb, 1e-6), mp


def frames():
    """[(seq, k, tags, trial, j)]: hand frames with a window where local index j >= 6."""
    T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    out = []
    for seq, tags in (('seq1', ('handmask_raw', 'handpoly')), ('seq2', ('handA',)), ('seq3', ('handA',)), ('seq4', ('handA',))):
        d = f'data/nfo_processed/{seq}_gt'
        ks = sorted({int(f[:5]) for f in os.listdir(d) for g in tags if f.endswith(f'_{g}.png')})
        for k in ks:
            win = [(t, t['frames'].index(k)) for t in T if t['seq'] == seq and k in t['frames']]
            win = [(t, j) for t, j in win if j >= 6]
            if win:
                out.append((seq, k, tuple(g for g in tags if gt(seq, k, g) is not None), *win[0]))
    return out


def cache():
    os.makedirs(OUT, exist_ok=True)
    pred, trk, bgs = R.image_predictor(), {}, {}
    for seq, k, tags, t, j in frames():
        p = f'{OUT}/{seq}_{k:05d}.npz'
        if os.path.exists(p):
            continue
        if t['id'] not in trk:
            trk[t['id']] = R.track(t)
            w0, w1 = t['warmup']
            bgs[t['id']] = np.median(np.stack([V.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
        M1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks'][j].astype(bool)
        B0 = np.load(f"{TR.RES}/masks/b0/{t['id']}.npz")['masks'][j].astype(bool)
        f = features(t, j, trk[t['id']], bgs[t['id']], pred, M1)
        MC, pb = f if f is not None else (np.zeros_like(M1), np.zeros(M1.shape, np.float32))
        cer = cv2.imread(f'data/nfo_processed/{seq}_gt/{k:05d}_sammask_native.png', 0)
        np.savez_compressed(p, MC=MC, pb=pb.astype(np.float16), T1=M1, B0=B0, trial=t['id'], j=j,
                            ceil=(cer > 127) if cer is not None else np.zeros_like(M1))
        print(f'{seq} {k} ({t["id"]}, j={j}) cached', flush=True)


def load(seq, k):
    z = np.load(f'{OUT}/{seq}_{k:05d}.npz')
    d = {n: z[n] for n in ('MC', 'T1', 'B0', 'ceil')}
    d['pb'] = z['pb'].astype(np.float32)
    d['mb'], d['sb'], d['mp'] = anchors(d['MC'], d['T1'], d['pb'])
    d['cand'] = d['MC'] & ~d['T1']
    return d


def X(d, sel):
    pb = d['pb'][sel]
    return np.stack([pb, np.full_like(pb, d['mb']), np.full_like(pb, d['sb']), np.full_like(pb, d['mp'])], 1)


def add_anchored(d, u):
    if not np.isfinite(d['mp']) or d['mp'] <= d['mb']:
        return np.zeros_like(d['T1'])
    return d['cand'] & ((d['pb'] - d['mb']) / (d['mp'] - d['mb']) > u)


def add_logistic(d, clf):
    a = np.zeros_like(d['T1'])
    if d['cand'].any() and np.isfinite(d['mp']):
        a[d['cand']] = clf(X(d, d['cand'])) > 0.5
    return a


def fit_u(D):
    J = [np.mean([stats(d['T1'] | add_anchored(d, u), h)[0] for d, h in D]) for u in UGRID]
    return UGRID[int(np.argmax(J))]


def fit_logistic(D, it=300):
    """Per-pixel logistic regression on standardised features, Newton–Raphson (IRLS), tiny L2 for stability."""
    Xs, ys = [], []
    for d, h in D:
        if d['cand'].any() and np.isfinite(d['mp']):
            Xs.append(X(d, d['cand'])); ys.append(h[d['cand']])
    Xs, ys = np.concatenate(Xs).astype(np.float64), np.concatenate(ys).astype(np.float64)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    A = np.c_[(Xs - mu) / sd, np.ones(len(Xs))]
    w = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - ys) + 1e-3 * w
        H = (A * (p * (1 - p))[:, None]).T @ A + 1e-3 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    clf = lambda Z: 1 / (1 + np.exp(-np.c_[(Z - mu) / sd, np.ones(len(Z))] @ w))
    clf.raw = (w[:-1] / sd, w[-1] - (w[:-1] * mu / sd).sum())
    return clf


def binned_recall(m, h):
    n, lab = cv2.connectedComponents(h.astype(np.uint8), connectivity=8)
    out = []
    for c in range(1, n):
        cc = lab == c
        out.append((cc.sum() / CELL ** 2, (m & cc).sum() / cc.sum()))
    return out


def main():
    cache()
    F = frames()
    tr = [(load(s, k), gt(s, k, 'handmask_raw'), k) for s, k, tags, _, _ in F if s == 'seq1' and 'handmask_raw' in tags]
    trp = {k: gt('seq1', k, 'handpoly') for s, k, tags, _, _ in F if s == 'seq1' and 'handpoly' in tags}
    te = [(s, k, load(s, k), gt(s, k, 'handA')) for s, k, tags, _, _ in F if s != 'seq1']
    print(f'train (seq1 brush) n={len(tr)}; seq1 polygons n={len(trp)}; test (seq2-4 handA) n={len(te)}')
    rng = np.random.default_rng(0)
    ci = lambda x: np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)], [2.5, 97.5])

    # seq1 leave-one-frame-out
    lofo = {'z_fixed': [], 'anchored': [], 'logistic': []}
    for i, (d, h, k) in enumerate(tr):
        rest = [(dd, hh) for n, (dd, hh, _) in enumerate(tr) if n != i]
        u, clf = fit_u(rest), fit_logistic(rest)
        base = stats(d['T1'], h)[0]
        lofo['z_fixed'].append(stats(d['T1'] | (d['cand'] & (pixz(d['pb']) > 7.5)), h)[0] - base)
        lofo['anchored'].append(stats(d['T1'] | add_anchored(d, u), h)[0] - base)
        lofo['logistic'].append(stats(d['T1'] | add_logistic(d, clf), h)[0] - base)
    print('\nseq1 LOFO ΔJ vs brush (arm − T1), mean [95% CI]')
    for a, v in lofo.items():
        v = np.array(v); lo, hi = ci(v)
        print(f'  {a:9s} {v.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  better {(v > 0).sum()}/{len(v)}')
    best = max(('anchored', 'logistic'), key=lambda a: np.mean(lofo[a]))
    D = [(d, h) for d, h, _ in tr]
    u, clf = fit_u(D), fit_logistic(D)
    print(f'chosen learned arm: {best}; fitted on all seq1 brush: u* = {u}; logistic w(pb, μ_B, σ_B, μ_P) = '
          f'{np.round(clf.raw[0], 2)}, b = {clf.raw[1]:.2f}')

    arms = {'T1': lambda d: d['T1'], 'B0': lambda d: d['B0'], 'ceiling': lambda d: d['ceil'],
            'z_fixed': lambda d: d['T1'] | (d['cand'] & (pixz(d['pb']) > 7.5)),
            'anchored': lambda d: d['T1'] | add_anchored(d, u), 'logistic': lambda d: d['T1'] | add_logistic(d, clf)}
    # seq1 polygons, parameters fitted on all seq1 brush (polygons were never used for fitting)
    print('\nseq1 polygons (n=%d): J / P / R' % len(trp))
    for a, f in arms.items():
        s = np.mean([stats(f(load('seq1', k)), h) for k, h in trp.items()], 0)
        print(f'  {a:9s} ' + ' / '.join(f'{v:.3f}' for v in s))
    print('\nseq2-4 handA (TEST, n=%d): J / P / R, and added area / T1 area' % len(te))
    S = {a: np.array([stats(f(d), h) for _, _, d, h in te]) for a, f in arms.items()}
    for a in arms:
        add = np.mean([(arms[a](d) & ~d['T1']).sum() / max(d['T1'].sum(), 1) for _, _, d, _ in te])
        print(f'  {a:9s} ' + ' / '.join(f'{v:.3f}' for v in S[a].mean(0)) + f'   added {add:.2f}×')
    for a in ('B0', 'ceiling', 'z_fixed', 'anchored', 'logistic'):
        dj = S[a][:, 0] - S['T1'][:, 0]; lo, hi = ci(dj)
        per = '  '.join(f'{s}: {dj[[i for i, x in enumerate(te) if x[0] == s]].mean():+.3f}' for s in ('seq2', 'seq3', 'seq4'))
        print(f'  ΔJ {a:9s} − T1 {dj.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] better {(dj > 0).sum()}/{len(dj)}   {per}')
    dj = S[best][:, 0] - S['T1'][:, 0]
    print(f'pre-registered test ({best} − T1 on seq2-4): {"PASS" if ci(dj)[0] > 0 else "FAIL"}')

    print('\nseq2-4 recall of hand-label components by size (stride-16 cells): n / T1 / B0 / ' + best)
    bins = [(0, .5), (.5, 1), (1, 4), (4, 1e9)]
    rows = {a: [r for _, _, d, h in te for r in binned_recall(arms[a](d), h)] for a in ('T1', 'B0', best)}
    for lo_, hi_ in bins:
        v = {a: [r for s_, r in rows[a] if lo_ <= s_ < hi_] for a in rows}
        print(f'  [{lo_}, {hi_ if hi_ < 1e9 else "inf"}) cells: {len(v["T1"])} / ' + ' / '.join(
            f'{np.mean(v[a]):.2f}' if v[a] else 'n/a' for a in ('T1', 'B0', best)))
    np.save(f'{OUT}/test_stats.npy', {'te': [(s, k) for s, k, _, _ in te], 'S': S, 'u': u}, allow_pickle=True)


def loso():
    """Leave-one-sequence-out (2026-10-14, after the seq2-4 test above): fit on the other three sequences' labels
    (seq1 brush, seq2-4 handA), score the held-out sequence. Also: share of hand-label area in sub-cell components,
    and J of T1 if every sub-cell component were added perfectly (the ceiling of a sub-cell-only fix)."""
    F = frames()
    lab = {'seq1': 'handmask_raw', 'seq2': 'handA', 'seq3': 'handA', 'seq4': 'handA'}
    data = {s: [(load(s_, k), gt(s_, k, lab[s_])) for s_, k, tags, _, _ in F if s_ == s and lab[s_] in tags] for s in lab}
    rng = np.random.default_rng(0)
    print('LOSO ΔJ(arm − T1) on the held-out sequence (n frames): anchored [u*] / logistic')
    allv = {'anchored': [], 'logistic': []}
    for s in lab:
        D = [x for s2 in lab if s2 != s for x in data[s2]]
        u, clf = fit_u(D), fit_logistic(D)
        da = [stats(d['T1'] | add_anchored(d, u), h)[0] - stats(d['T1'], h)[0] for d, h in data[s]]
        dl = [stats(d['T1'] | add_logistic(d, clf), h)[0] - stats(d['T1'], h)[0] for d, h in data[s]]
        allv['anchored'] += da; allv['logistic'] += dl
        print(f'  {s} (n={len(da)}): {np.mean(da):+.3f} [u*={u}] / {np.mean(dl):+.3f}   w={np.round(clf.raw[0], 2)}')
    for a, v in allv.items():
        v = np.array(v); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
        print(f'  pooled {a:9s} {v.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] '
              f'better {(v > 0).sum()}/{len(v)}')
    for s in lab:
        sh, jc, j0 = [], [], []
        for d, h in data[s]:
            n, L, st, _ = cv2.connectedComponentsWithStats(h.astype(np.uint8), connectivity=8)
            small = np.isin(L, [c for c in range(1, n) if st[c, cv2.CC_STAT_AREA] < CELL ** 2])
            sh.append(small.sum() / max(h.sum(), 1)); j0.append(stats(d['T1'], h)[0]); jc.append(stats(d['T1'] | small, h)[0])
        print(f'  {s}: sub-cell components hold {np.mean(sh) * 100:.1f}% of label area; T1 J {np.mean(j0):.3f} -> '
              f'{np.mean(jc):.3f} if all of them were added')


if __name__ == '__main__':
    loso() if sys.argv[1:] == ['loso'] else main()
