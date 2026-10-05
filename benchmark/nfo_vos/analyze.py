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
MAIN = ['b0', 't1', 't4', 't4-1', 't4-2*', 't4-3']
COLS = ['J@10', 'J@25', 'J@50', 'JF', 'J_decay', 'DRE', 'NRE', 'P_norm']
COMPARISONS = [('t4', 't4-1', 'integration effect, with SAM2'),
               ('t4-2*', 't4-3', 'integration effect, without SAM2'),
               ('t4', 't4-2*', 'does SAM2 add to integration'),
               ('t4', 'b0', 'integration vs memory propagation (SAM2)'),
               ('t4', 't1', 'integration vs memory propagation (SAMURAI)')]


def md_table(df):
    head = '| method | ' + ' | '.join(df.columns) + ' |'
    rows = [f'| {i} | ' + ' | '.join(f'{v:.3f}' for v in r) + ' |' for i, r in df.iterrows()]
    return '\n'.join([head, '|' + '---|' * (len(df.columns) + 1), *rows])


def pick_vote_m(per_trial):
    """T4-2*: one m for the whole pilot, best mean J@50 (spec §4). Favours the control."""
    v = per_trial[per_trial.method.str.startswith('t4-2_m')]
    return int(v.groupby('method')['J@50'].mean().idxmax().split('_m')[1])


def fe_fit(df):
    """J(t|t0) = a_t + b v + c dt + d v dt + e n_frag, one fixed effect a_t per frame, fitted on
    frames scored by >= 2 trials (the only frames that identify b..e)."""
    df = df[df.groupby(['seg', 't']).trial.transform('nunique') >= 2]
    key = df.seg.astype(str) + '_' + df.t.astype(str)
    D = pd.get_dummies(key).values.astype(float)
    X = np.column_stack([D, df.v, df.dt, df.v * df.dt, df.n_frag])
    coef, *_ = np.linalg.lstsq(X, df.J.values, rcond=None)
    b, c, d, e = coef[-4:]
    return dict(b=b, c=c, d=d, e=e, n_obs=len(df), n_frames=key.nunique(), n_trials=df.trial.nunique())


def fe_with_loo(df):
    est = fe_fit(df)
    loo = pd.DataFrame([fe_fit(df[df.trial != tr]) for tr in df.trial.unique()])
    return est, {k: (loo[k].min(), loo[k].max()) for k in 'bcde'}


def stride20(df):
    seg_start = df.groupby('seg').t0.transform('min')
    return df[(df.t0 - seg_start) % 20 == 0]


def rename_star(df, m):
    df = df.copy()
    df.loc[df.method == f't4-2_m{m}', 'method'] = 't4-2*'
    return df


def run_record():
    git = lambda *a, cwd='.': subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True).stdout.strip()
    return dict(nfo_unet_commit=git('rev-parse', 'HEAD'), nfo_unet_dirty=bool(git('status', '--porcelain', '--', 'benchmark')),
                samurai_commit=git('rev-parse', 'HEAD', cwd='../samurai'),
                checkpoint='sam2.1_hiera_base_plus.pt', precision='fp16 autocast (all SAM2 calls)',
                envs=dict(b0_t4='../master_thesis/.venv (sam2 1.1.0)', t1='../samurai/.venv'),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                host=platform.node(), trials=TR.OUT, stride=TR.STRIDE, window=TR.WINDOW, min_D=TR.MIN_D,
                note='timings were measured with several methods sharing the GPU: not comparable')


def main():
    pt = pd.read_csv(f'{RES}/per_trial.csv')
    pf = pd.read_csv(f'{RES}/per_frame.csv')
    m = pick_vote_m(pt) if pt.method.str.startswith('t4-2_m').any() else None
    vote = pt[pt.method.str.startswith('t4-2_m')].groupby('method')['J@50'].mean()
    if m is not None:
        pt, pf = rename_star(pt, m), rename_star(pf, m)
    present = [x for x in MAIN if x in set(pt.method)]
    lowv = pt.v < pt.drop_duplicates('trial').v.median()

    sec = {}
    for meth in present:
        d = f'{RES}/masks/{"t4-2_m%d" % m if meth == "t4-2*" else meth}'
        sec[meth] = np.median([float(np.load(f'{d}/{f}')['sec_per_frame']) for f in os.listdir(d)])

    L = ['# NFO VOS pilot: results', '',
         'seq1 segments 4-5, 21 starts, 51-frame windows, pseudo-GT at 224. **Descriptive only** '
         '(two segments; spec §5).', '']
    if m is not None:
        L += [f'T4-2* = vote m = {m} (best mean J@50 of m=1..7: ' +
              ', '.join(f'm{k.split("_m")[1]} {v:.3f}' for k, v in vote.items()) + ').', '']
    tab = pt[pt.method.isin(present)].groupby('method')[COLS].mean().reindex(present)
    tab['J@50 low-v'] = pt[lowv & pt.method.isin(present)].groupby('method')['J@50'].mean()
    tab['s/frame*'] = pd.Series(sec)
    L += ['## Method means over trials', '', md_table(tab), '',
          '*low-v = trials with v(t0) below the median start. s/frame measured with methods sharing '
          'the GPU, not comparable (re-profile on the cluster).', '']

    L += ['## Comparisons answering §1 (paired over trials)', '',
          '| comparison | question | ΔJ@50 mean | ΔJF mean | trials A > B (J@50) |', '|---|---|---|---|---|']
    for a, b, q in COMPARISONS:
        if a in present and b in present:
            A = pt[pt.method == a].set_index('trial'); B = pt[pt.method == b].set_index('trial')
            dJ, dF = (A['J@50'] - B['J@50']), (A.JF - B.JF)
            L.append(f'| {a} − {b} | {q} | {dJ.mean():+.3f} | {dF.mean():+.3f} | {(dJ > 0).sum()}/{len(dJ)} |')
    L += ['']

    L += ['## Init robustness: frame fixed effects (spec §5)', '',
          'J = a_t + b·v + c·Δt + d·v·Δt + e·n_frag; [leave-one-trial-out min, max]. '
          'd > 0: a better prompt slows the drift.', '',
          '| method | stride | b | c | d | e | trials |', '|---|---|---|---|---|---|---|']
    for meth in present:
        for name, sub in (('10', pf[pf.method == meth]), ('20', stride20(pf[pf.method == meth]))):
            est, rng = fe_with_loo(sub)
            cell = lambda k, f: f'{est[k]:{f}} [{rng[k][0]:{f}}, {rng[k][1]:{f}}]'
            L.append(f'| {meth} | {name} | {cell("b", "+.3f")} | {cell("c", "+.4f")} | '
                     f'{cell("d", "+.4f")} | {cell("e", "+.3f")} | {est["n_trials"]} |')
    L += ['']

    os.makedirs(IMG, exist_ok=True)
    fig, axs = plt.subplots(1, len(present), figsize=(10, 2.4), sharey=True)
    for ax, meth in zip(np.atleast_1d(axs), present):
        s = pt[pt.method == meth]
        ax.scatter(s.v, s['J@50'], s=12, c=s.seg.map({4: 'C0', 5: 'C1'}))
        k, c0 = np.polyfit(s.v, s['J@50'], 1)
        xs = np.linspace(s.v.min(), s.v.max(), 2); ax.plot(xs, k * xs + c0, 'k-', lw=0.8)
        ax.set_title(f'{meth} slope {k:+.2f}', fontsize=8); ax.set_xlabel('v(t0)', fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel('J@50 (blue seg4, orange seg5)', fontsize=7)
    fig.tight_layout(); fig.savefig(f'{IMG}/j50_vs_v.png', dpi=80); plt.close(fig)
    L += [f'Scatter: `{IMG}/j50_vs_v.png`. Montages (rows = methods): `{IMG}/<trial>.png`.']

    trials = {t['id']: t for t in json.load(open(TR.OUT))}
    for tid in sorted(pt.trial.unique()):
        stacks = {}
        for meth in present:
            src = f't4-2_m{m}' if meth == 't4-2*' else meth
            path = f'{RES}/masks/{src}/{tid}.npz'
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
