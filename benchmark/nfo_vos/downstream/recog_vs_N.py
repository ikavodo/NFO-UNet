"""Recognition vs composite length N (seq2–4 test): YOLO11m person AP50 on masked GT-motion composites of N frames,
N ∈ {1, 3, 5, 7, 11, 15, 21}, uniform masked mean, stride max(1, ⌊N/2⌋). All three CSVs from one interpreter (NFO_RECOG_TAG=_ablN,
2026-10-20); absolute AP is only comparable within that set. 95% CI: window bootstrap (AP recomputed per resample, as
recognition.report).
N = 1 is the same for every arm by construction: unmasked pixels are filled with the plain aligned mean, which for a
single frame is the frame itself, so every composite is the raw frame (the no-integration reference point).

    python3 -m benchmark.nfo_vos.downstream.recog_vs_N
"""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark.nfo_vos.downstream import box_metrics as BM
from benchmark.nfo_vos.downstream.recognition import ap50

NS = (1, 3, 5, 7, 11, 15, 21)
# dataviz default categorical slots 1-4 (validated reference palette); no-mask baseline in neutral gray, dashed
ARMS = (('adapter-pf', 'adapter (ours, per-frame)', '#2a78d6', '-'), ('t1-c2', 'SAMURAI', '#eb6834', '-'),
        ('b0-c2', 'SAM2.1 video', '#1baf7a', '-'), ('trk', 'SAM2.1 per-frame (tracker box)', '#eda100', '-'),
        ('nomask', 'no mask', '#8a8984', '--'))
OUT = f'{BM.IMG}/recog_vs_N.png'


def ap_ci(rows, arm, B=1000):
    win = {}
    for r in rows:
        if r['arm'] == arm and r['seq'] != 'seq1':
            win.setdefault(r['id'], []).append([tuple(x) for x in json.loads(r['dets'])])   # one det list per composite
    ids = sorted(win)
    boot = np.random.default_rng(0).integers(0, len(ids), (B, len(ids)))
    bs = [ap50([d for i in b for d in win[ids[i]]]) for b in boot]
    return ap50([d for i in ids for d in win[i]]), np.quantile(bs, .025), np.quantile(bs, .975)


def main():
    res = {}
    for N in NS:
        rows = list(csv.DictReader(open(f'{BM.OUT}/recog_N{N}_uniform_meanfill_ablN.csv')))
        for a, *_ in ARMS:
            res[a, N] = ap_ci(rows, a)
            print(f'N={N} {a:12s} AP50 {res[a, N][0]:.3f} [{res[a, N][1]:.3f}, {res[a, N][2]:.3f}]')
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=100)
    for a, label, col, ls in ARMS:
        y = np.array([res[a, N] for N in NS])
        ax.fill_between(NS, y[:, 1], y[:, 2], color=col, alpha=0.12, lw=0)
        ax.plot(NS, y[:, 0], ls, color=col, lw=2, marker='o', ms=6, mec='white', mew=1.5, label=label)
    # direct labels at N = 7, spread so close end points (SAMURAI / SAM2 video / ours) do not collide
    ends = sorted(((res[a, NS[-1]][0], label) for a, label, *_ in ARMS), reverse=True)
    pos, gap = [], 0.022
    for y, _ in ends:
        pos.append(min(y, pos[-1] - gap) if pos else y)
    for (y, label), p in zip(ends, pos):
        ax.annotate(label, (NS[-1], y), xytext=(NS[-1] + 0.6, p), textcoords='data', va='center', fontsize=8.5,
                    color='#0b0b0b', annotation_clip=False)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_xticks(NS); ax.set_xlim(0.5, NS[-1] + 0.5)
    ax.annotate('N = 1: raw frame,\nidentical for all arms', (1, res['nomask', 1][0]), xytext=(1.25, res['nomask', 1][0] - 0.06),
                fontsize=8, color='#52514e')
    ax.set_xlabel('frames per composite N'); ax.set_ylabel('YOLO11m person AP50')
    ax.set_title('Recognition on masked temporal composites (NFO seq2–4, 95% CI)', fontsize=10, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.subplots_adjust(right=0.68)
    fig.savefig(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
