"""Pilot analysis (spec §5-§6): method table, the §1 comparisons, frame fixed-effects init model,
J@50-vs-v scatter, montages with T4-2*, run.json and summary.md.

    python -m benchmark.nfo_vos.analyze

Descriptive only: two segments, no confidence intervals (spec §5). Leave-one-trial-out ranges
are reported to show how much a single trial moves each estimate, not as intervals.
"""
import json
import os
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark.nfo_vos import score, trials as TR

RES, IMG = score.RES, score.IMG
MAIN = ['b0', 't1', 't1-a4', 't1-g0.5', 't1-g0.6', 't1-g0.7', 't1-g0.8', 't4-1n']
COLS = ['JF', 'J_mean', 'F_mean', 'J_decay', 'DRE', 'NRE', 'P_norm', 'J@10', 'J@25', 'J@50']
COMPARISONS = [('t1', 'b0', 'SAMURAI vs SAM2'),
               ('t1-a4', 't1', 'fixed A/4 composite insertion, SAMURAI'),
               *[(f't1-g{tau:g}', 't1', f'confidence-gated A/4 (tau = {tau:g}), SAMURAI') for tau in (0.5, 0.6, 0.7, 0.8)],
               ('t4-1n', 'b0', 'per-frame SAM2 + tracker prompt (no memory) vs memory propagation')]


def md_table(df):
    head = '| method | ' + ' | '.join(df.columns) + ' |'
    rows = [f'| {i} | ' + ' | '.join(f'{v:.3f}' for v in r) + ' |' for i, r in df.iterrows()]
    return '\n'.join([head, '|' + '---|' * (len(df.columns) + 1), *rows])


XBIN = 8          # px (224 space), ~ a third of a body width


def fe_fit(df, effect='frame'):
    """J = a + b v + c dt + d v dt + e n_frag. effect='frame': one fixed effect per frame (old
    overlapping-window pilot). effect='pos': one per (sequence, x-bin), the shared occluder field.
    Visibility at the same x correlates r ~ 0.7 across segments (rev. 5), so non-overlapping
    windows from different segments identify b..e. Fitted on cells hit by >= 2 trials."""
    if effect == 'pos':
        df = df.assign(_k=df.seq.astype(str) + '_' + (df.x // XBIN).astype(int).astype(str))
    else:
        df = df.assign(_k=df.seg.astype(str) + '_' + df.t.astype(str))
    df = df[df.groupby('_k').trial.transform('nunique') >= 2]
    key = df._k
    D = pd.get_dummies(key).values.astype(float)
    X = np.column_stack([D, df.v, df.dt, df.v * df.dt, df.n_frag])
    coef, *_ = np.linalg.lstsq(X, df.J.values, rcond=None)
    b, c, d, e = coef[-4:]
    return dict(b=b, c=c, d=d, e=e, n_obs=len(df), n_frames=key.nunique(), n_trials=df.trial.nunique())


def cluster_bootstrap_ci(df, col, cluster, B=10000, seed=0, strata=None):
    """95% CI of mean(df[col]) resampling whole clusters with replacement (cluster/block bootstrap,
    Davison & Hinkley 1997), optionally within strata (e.g. sequence), so the correlation of rows
    inside a cluster (starts of one twin group share position and scene) is kept."""
    rng = np.random.default_rng(seed)
    groups = {k: g[col].values for k, g in df.groupby(cluster)}
    if strata is None:
        by_stratum = {None: list(groups)}
    else:
        by_stratum = {s: list(g[cluster].unique()) for s, g in df.groupby(strata)}
    means = []
    for _ in range(B):
        vals = [groups[k] for keys in by_stratum.values() for k in rng.choice(keys, len(keys))]
        means.append(np.concatenate(vals).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def twin_reliability(pt, metric='JF'):
    """Cover vs backup twins (same seq, dir, gait, phase, window m): test-retest of a metric."""
    if 'half' not in pt or pt.half.isna().all():
        return None
    w = pt.pivot_table(index=['seq', 'dir', 'gait', 'm'], columns='half', values=metric).dropna()
    if len(w) < 3:
        return None
    return dict(n=len(w), r=float(np.corrcoef(w.cover, w.backup)[0, 1]), mad=float((w.cover - w.backup).abs().mean()))


def fe_with_loo(df, effect='frame'):
    est = fe_fit(df, effect)
    loo = pd.DataFrame([fe_fit(df[df.trial != tr], effect) for tr in df.trial.unique()])
    return est, {k: (loo[k].min(), loo[k].max()) for k in 'bcde'}


def run_record():
    git = lambda *a, cwd='.': subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
    return dict(nfo_unet_commit=git('rev-parse', 'HEAD'), nfo_unet_dirty=bool(git('status', '--porcelain', '--', 'benchmark')),
                samurai_commit=git('rev-parse', 'HEAD', cwd='../samurai'),
                checkpoint='sam2.1_hiera_base_plus.pt', precision='fp16 autocast (all SAM2 calls)',
                envs=dict(b0_t4='../master_thesis/.venv (sam2 1.1.0)', t1='../samurai/.venv'),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                host=platform.node(), trials=TR.OUT, run=TR.RUN, window=TR.W, min_D=TR.MIN_D,
                note='timings were measured with several methods sharing the GPU: not comparable')


def main():
    pt = pd.read_csv(f'{RES}/per_trial.csv')
    pf = pd.read_csv(f'{RES}/per_frame.csv')
    pf = pf[(pf.dt > 0) & (pf.dt < pf.groupby('trial').dt.transform('max'))]   # DAVIS range: first/last dropped
    present = [x for x in MAIN if x in set(pt.method)]

    sec = {}
    for meth in present:
        d = f'{RES}/masks/{meth}'
        sec[meth] = np.median([float(np.load(f'{d}/{f}')['sec_per_frame']) for f in os.listdir(d)])

    L = [f'# NFO VOS benchmark: results ({TR.RUN})', '',
         f'{pt.trial.nunique()} starts, {TR.W}-frame windows (round-robin schedule, spec §3a), pseudo-GT at 224.', '']
    tab = pt[pt.method.isin(present)].groupby('method')[COLS].mean().reindex(present)
    tab.insert(1, 'init fails', pt[pt.method.isin(present)].groupby('method').init_fail.sum())
    tab['s/frame*'] = pd.Series(sec)
    L += ['## Method means over trials', '', md_table(tab), '',
          'JF/J_mean/F_mean/decay/DRE/NRE/P_norm: DAVIS-style, frames t0+1..t0+49 (first and last '
          'dropped). init fails: starts whose mask at the prompted frame t0 has J < 0.5 (of 21). '
          'J@10/25/50: single-frame trajectory readouts, secondary. *s/frame measured with methods '
          'sharing the GPU, not comparable (re-profile on the cluster).', '']

    L += ['## Comparisons answering §1 (paired over trials)', '',
          'ΔJ&F with a 95% cluster-bootstrap CI (twin groups, stratified by sequence).', '',
          '| comparison | question | ΔJ&F mean [95% CI] | trials A > B (J&F) | ΔJ@50 mean |', '|---|---|---|---|---|']
    for a, b, q in COMPARISONS:
        if a in present and b in present:
            A = pt[pt.method == a].set_index('trial'); B = pt[pt.method == b].set_index('trial')
            dJ, dF = (A['J@50'] - B['J@50']), (A.JF - B.JF)
            ci = ''
            if {'seq', 'dir', 'gait'} <= set(A.columns):
                d = pd.DataFrame({'d': dF}).join(A[['seq', 'dir', 'gait']]).dropna()
                d['cl'] = d.seq.astype(str) + d.dir.astype(str) + d.gait.astype(str)
                lo, hi = cluster_bootstrap_ci(d, 'd', 'cl', strata='seq')
                ci = f' [{lo:+.3f}, {hi:+.3f}]'
            L.append(f'| {a} − {b} | {q} | {dF.mean():+.3f}{ci} | {(dF > 0).sum()}/{len(dF)} | {dJ.mean():+.3f} |')
    L += ['']

    effect = 'pos' if 'x' in pf and pf.x.notna().all() else 'frame'
    L += [f"## Init robustness: {'position (sequence, x-bin)' if effect == 'pos' else 'frame'} fixed effects (spec §5)", '',
          'J = a + b·v + c·Δt + d·v·Δt + e·n_frag; [leave-one-trial-out min, max]. '
          'd > 0: a better prompt slows the drift.', '',
          '| method | b | c | d | e | trials |', '|---|---|---|---|---|---|']
    for meth in present:
        est, rng = fe_with_loo(pf[pf.method == meth], effect)
        cell = lambda k, f: f'{est[k]:{f}} [{rng[k][0]:{f}}, {rng[k][1]:{f}}]'
        L.append(f'| {meth} | {cell("b", "+.3f")} | {cell("c", "+.4f")} | '
                 f'{cell("d", "+.4f")} | {cell("e", "+.3f")} | {est["n_trials"]} |')
    L += ['']
    tw = [(meth, twin_reliability(pt[pt.method == meth])) for meth in present]
    if any(r for _, r in tw):
        L += ['## Cover vs backup twins (test-retest, J&F)', '', '| method | pairs | Pearson r | mean |Δ| |', '|---|---|---|---|']
        L += [f"| {m} | {r['n']} | {r['r']:+.2f} | {r['mad']:.3f} |" for m, r in tw if r]
        L += ['']

    os.makedirs(IMG, exist_ok=True)
    fig, axs = plt.subplots(1, len(present), figsize=(10, 2.4), sharey=True)
    for ax, meth in zip(np.atleast_1d(axs), present):
        s = pt[pt.method == meth].dropna(subset=['JF'])
        col = s.gait.map({'walk': 'C0', 'run': 'C1'}).fillna('C2') if 'gait' in s else 'C0'
        ax.scatter(s.v, s.JF, s=12, c=col)
        if len(s) > 1:
            k, c0 = np.polyfit(s.v, s.JF, 1)
            xs = np.linspace(s.v.min(), s.v.max(), 2); ax.plot(xs, k * xs + c0, 'k-', lw=0.8)
            ax.set_title(f'{meth} slope {k:+.2f}', fontsize=8)
        ax.set_xlabel('v(t0)', fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel('J&F (blue walk, orange run)', fontsize=7)
    fig.tight_layout(); fig.savefig(f'{IMG}/jf_vs_v.png', dpi=80); plt.close(fig)
    L += [f'Scatter: `{IMG}/jf_vs_v.png`. Montages (rows = methods): `{IMG}/<trial>.png`.']

    trials = {t['id']: t for t in json.load(open(TR.OUT))}
    for tid in sorted(pt.trial.unique()):
        stacks = {}
        for meth in present:
            path = f'{RES}/masks/{meth}/{tid}.npz'
            if os.path.exists(path):
                stacks[meth] = score.to_224(np.load(path)['masks'])
        score.montage(trials[tid], stacks, f'{IMG}/{tid}.png')

    with open(f'{RES}/run.json', 'w') as f:
        json.dump(run_record(), f, indent=1)
    with open(f'{RES}/summary.md', 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
