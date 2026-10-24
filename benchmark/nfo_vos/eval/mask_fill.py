"""Does the mask shrink over a clip? Fill f(k) = mask area / GT box area at clip frame k (native px), per arm.
The GT box is the amodal person box, so f is invariant to the person's apparent size (its area stays at 0.98–1.00 of
frame 0 up to k = 30, all 61 windows). Trend per clip: least-squares slope of log f over k = 6…end (from k = 6 on every
arm runs in its steady mode: the adapter needs 6 frames of history, before that it uses the tracker prompts), reported
as % change per 10 frames. Ours = the 3 adapter seeds averaged per clip. 95% CI: bootstrap over passes (segments), since
clips of the same pass are not independent (1–3 benchmark windows per pass; overlapping clips in the pass run).

Two runs (NFO_RUN):
  rr     the benchmark windows, 40 frames each
  pass   the same 61 starts (t0, prompt, warm-up) run to the end of their pass: 41–155 frames. Support shrinks with k
         (only clips longer than k count at k); the median curve is drawn while ≥ MIN_N clips remain.

Late size ratio (pass run): per clip, mean over k = K1…end of log(f_arm(k) / f_arm(K0)) − log(f_ours(k) / f_ours(K0)),
reported as exp(mean) = the memory arm's mask size relative to ours once the early drop is over; same CI over passes.

Area is not recall: a shrinking mask may lose person or background pixels; the hand frames are too sparse to tell
per k. trk keeps a fixed-size tracker box prompt, so its flat trend is partly by construction.

  python -m benchmark.nfo_vos.eval.mask_fill score   # results/benchmark/$NFO_RUN/downstream/mask_fill.csv (≈ 3 min for rr)
  python -m benchmark.nfo_vos.eval.mask_fill         # trends + images/benchmark/downstream/mask_fill_trend[_<run>].png
                                                     # (pass run also mask_fill_ratio_pass.png: memory arm / ours)
Pass run on the cluster: eval/pass_run.sh.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.eval import box_metrics as BM
from benchmark.nfo_vos.eval.report import SEEDS

ARMS = ('b0-c2', 't1-c2', 'trk', *SEEDS)
K0 = 6                       # first steady-state frame (adapter composite history)
K1 = 30                      # late-ratio start: the memory arms' drop is over by then (pass run)
MIN_N = 15                   # median curve drawn while at least this many clips are still running
CSV = f'{BM.OUT}/mask_fill.csv'
OUT = f"{BM.IMG}/mask_fill_trend{'' if TR.RUN == 'rr' else '_' + TR.RUN}.png"
GROUPS = (('seq1-4 (all)', TR.ALL_SEQS), ('seq1 (selection)', ('seq1',)), ('seq2-4 (test)', ('seq2', 'seq3', 'seq4')))
# recog_vs_N palette
PLOT = (('b0-c2', 'SAM2.1 video (memory)', '#1baf7a'), ('t1-c2', 'SAMURAI (memory)', '#eb6834'),
        ('ours', 'ours (no memory, 3 seeds)', '#2a78d6'))


def score(arms=ARMS):
    rows = []
    for t in json.load(open(TR.OUT)):
        gt = BM.gt_boxes(t['seq'])
        for a in arms:
            la = BM.load_arm(a, t['id'])
            if la is None:
                continue
            ms, fr = la
            for k, (m, f) in enumerate(zip(ms, fr)):
                b = gt[int(f)]
                rows.append(dict(arm=a, id=t['id'], seq=t['seq'], k=k, frame=int(f), area=int(m.sum()),
                                 box_area=float((b[2] - b[0]) * (b[3] - b[1]))))
    os.makedirs(BM.OUT, exist_ok=True)
    with open(CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote', CSV, len(rows), 'rows')


def pass_of(clip_id):
    return clip_id.rsplit('_t', 1)[0]           # 'seq2_s3_t45' -> 'seq2_s3'


def fills(rows):
    """{arm: {id: f[k] / f[0]}} (1 px floor on the area so log f stays finite), plus 'ours' = seed mean of the ratios."""
    acc = {}
    for r in rows:
        acc.setdefault(r['arm'], {}).setdefault(r['id'], {})[int(r['k'])] = max(int(r['area']), 1) / float(r['box_area'])
    F = {a: {i: np.array([d[k] for k in sorted(d)]) for i, d in w.items()} for a, w in acc.items()}
    F = {a: {i: f / f[0] for i, f in w.items()} for a, w in F.items()}
    ids = set.intersection(*(set(F[a]) for a in SEEDS))
    F['ours'] = {i: np.mean([F[a][i] for a in SEEDS], 0) for i in ids}
    return F


def trends(F):
    """{arm: {id: Δ log f per 10 frames over k = K0…end of that clip}}."""
    return {a: {i: 10 * np.polyfit(np.arange(K0, len(f)), np.log(f[K0:]), 1)[0] for i, f in w.items()}
            for a, w in F.items()}


def boot_mean(per_clip, ids, B=2000):
    """Mean over clips, 95% CI from resampling passes (all clips of a drawn pass enter together)."""
    groups = {}
    for i in ids:
        groups.setdefault(pass_of(i), []).append(per_clip[i])
    g = [groups[p] for p in sorted(groups)]      # fixed order: ids come from sets, whose order changes per run
    draws = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
    bs = [np.mean([x for j in d for x in g[j]]) for d in draws]
    return np.mean([x for v in g for x in v]), np.quantile(bs, .025), np.quantile(bs, .975), len(ids), len(g)


def late_ratio(F, a, ids):
    """{id: mean over k = K1…end of the log size ratio arm / ours, each relative to frame K0}, clips longer than K1."""
    rel = lambda f: np.log(f[K1:] / f[K0])
    return {i: np.mean(rel(F[a][i]) - rel(F['ours'][i])) for i in ids if len(F['ours'][i]) > K1}


def median_curve(F_a, k, B=1000):
    """Median of f(j) / f(K0) over the clips still running at j, 5-frame smoothed, for j in k; 95% band from resampling
    passes (all clips of a drawn pass enter together)."""
    ids = sorted(F_a)
    R = np.full((len(ids), len(k)), np.nan)
    for r, i in enumerate(ids):
        f = F_a[i]; n = min(len(f) - K0, len(k))
        R[r, :n] = f[K0:K0 + n] / f[K0]
    smooth = lambda m: np.convolve(np.pad(m, 2, mode='edge'), np.ones(5) / 5, mode='valid')
    rows = {}
    for r, i in enumerate(ids):
        rows.setdefault(pass_of(i), []).append(r)
    g = [rows[p] for p in sorted(rows)]
    draws = np.random.default_rng(0).integers(0, len(g), (B, len(g)))
    bs = np.array([smooth(np.nanmedian(R[[r for j in d for r in g[j]]], 0)) for d in draws])
    return smooth(np.nanmedian(R, 0)), np.quantile(bs, .025, 0), np.quantile(bs, .975, 0)


def plot_ratio(F, kmax, lens):
    """Pass run: per clip and frame, the memory arm's size / ours (both relative to frame K0); median over the clips
    still running with its 95% band over passes. Paired, so the pass-to-pass difficulty shared by all arms cancels."""
    k = np.arange(K0, kmax + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    for a, label, col in PLOT[:2]:
        if a not in F:
            continue
        y, lo, hi = median_curve({i: F[a][i] / F['ours'][i] for i in F['ours'] if i in F[a]}, k)
        ax.fill_between(k, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(k, y, color=col, lw=2.5, label=label.split(' (')[0] + ' / ours')
    for j in (K0, 40, 60, 80, kmax):
        ax.annotate(f'n={sum(L > j for L in lens)}', (j, 0.02), xycoords=('data', 'axes fraction'), fontsize=7.5,
                    color='#52514e', ha='center')
    ax.axhline(1, color='#2a78d6', lw=1.2, ls='--')
    ax.annotate('same size as ours', (K0 + 1, 1), xytext=(0, 4), textcoords='offset points', va='bottom', fontsize=8.5,
                color='#2a78d6')
    ax.set_xlim(K0, kmax); ax.set_ylim(0, 1.25); ax.set_xlabel('frame in the clip')
    ax.set_ylabel(f'mask size relative to ours\n(each relative to frame {K0})')
    ax.set_title(f'Memory models vs ours over whole passes, NFO seq1–4\n(median of {len(lens)} clips, '
                 '95% CI over passes; n = clips still running)', fontsize=9, loc='left')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.06), fontsize=8.5, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT.replace('mask_fill_trend', 'mask_fill_ratio')
    fig.savefig(out)
    print('wrote', out)


def pct(v):
    return f'{(np.exp(v) - 1) * 100:+.0f}%'


def main():
    if sys.argv[1:2] == ['score']:
        return score()
    rows = list(csv.DictReader(open(CSV)))
    sq = {r['id']: r['seq'] for r in rows}
    F = fills(rows)
    S = trends(F)
    lens = [len(f) for f in F['ours'].values()]
    print(f'run {TR.RUN}: {len(lens)} clips of {min(lens)}–{max(lens)} frames')
    for name, sel in GROUPS:
        ids = [i for i in S['ours'] if sq[i] in sel]
        print(f'\n== {name}, n = {len(ids)} clips; mask fill trend per 10 frames, k = {K0}…end (95% CI over passes)')
        for a in ('b0-c2', 't1-c2', 'trk', *SEEDS, 'ours'):
            if a in S:
                m, lo, hi, n, g = boot_mean(S[a], [i for i in ids if i in S[a]])
                print(f'  {a:12s} {pct(m):>5s}  [{pct(lo)}, {pct(hi)}]  ({n} clips, {g} passes)')
        for a in ('b0-c2', 't1-c2'):
            if a in S:
                both = [i for i in ids if i in S[a]]
                m, lo, hi, n, g = boot_mean({i: S[a][i] - S['ours'][i] for i in both}, both)
                print(f'  {a} − ours: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] log fill / 10 frames, {n} clips, {g} passes')
        if len(set(lens)) > 1:
            for a in ('b0-c2', 't1-c2'):
                if a in F:
                    r = late_ratio(F, a, [i for i in ids if i in F[a]])
                    m, lo, hi, n, g = boot_mean(r, list(r))
                    print(f'  {a} / ours mask size, k = {K1}…end: ×{np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]'
                          f'  ({n} clips, {g} passes)')
    # fitted trend (mean slope, band = CI) relative to frame K0; with clips of different length (pass run) also the
    # median of f(k) / f(K0) over the clips still running at k, while at least MIN_N remain
    kmax = max(k for k in range(K0, max(lens)) if sum(L > k for L in lens) >= MIN_N)
    k = np.arange(K0, kmax + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    for a, label, col in PLOT:
        if a not in S:
            continue
        if len(set(lens)) > 1:
            # clips of different length (pass run): the drop is front-loaded (≈ 20% within 30 frames, then level), so
            # a straight-line trend misdescribes it; draw the median over the clips still running, 5-frame smoothed
            y, lo, hi = median_curve(F[a], k)
            ax.fill_between(k, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(k, y, color=col, lw=2.5, label=label)
            ax.annotate(f'{y[-1]:.2f}', (k[-1], y[-1]), xytext=(6, 0), textcoords='offset points', va='center',
                        fontsize=9, color=col, annotation_clip=False)
            continue
        m, lo, hi, *_ = boot_mean(S[a], list(S[a]))
        y = lambda s: np.exp(s / 10 * (k - K0))
        ax.fill_between(k, y(lo), y(hi), color=col, alpha=0.15, lw=0)
        ax.plot(k, y(m), color=col, lw=2.5, label=label)
        ax.annotate(f'{pct(m)} per 10 frames', (k[-1], y(m)[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=col, annotation_clip=False)
    if len(set(lens)) > 1:
        for j in (K0, 40, 60, 80, kmax):
            ax.annotate(f'n={sum(L > j for L in lens)}', (j, 0.02), xycoords=('data', 'axes fraction'), fontsize=7.5,
                        color='#52514e', ha='center')
    ax.axhline(1, color='#52514e', lw=0.6, ls=':')
    ax.set_xlim(K0, kmax); ax.set_xlabel('frame in the clip')
    ax.set_ylabel(f'mask size relative to frame {K0}\n(mask area / person box area)')
    if len(set(lens)) == 1:
        ax.set_title(f'Fitted trend of mask size, NFO seq1–4 ({len(lens)} clips of 40 frames, 95% CI)', fontsize=9, loc='left')
    else:
        ax.set_title(f'Mask size over whole passes, NFO seq1–4 (median of {len(lens)} clips, 95% CI over passes;\n'
                     f'n = clips still running)',
                     fontsize=9, loc='left')
    ax.legend(loc='lower left', fontsize=8.5, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.subplots_adjust(right=0.78)
    fig.savefig(OUT)
    print('wrote', OUT)
    if len(set(lens)) > 1:
        plot_ratio(F, kmax, lens)


if __name__ == '__main__':
    main()
