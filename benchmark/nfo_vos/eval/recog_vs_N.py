"""Recognition vs stack length N (seq2–4 test): YOLO11m person AP50 on masked GT-motion stacks of N frames,
N ∈ {1, 3, 5, 7, 11, 15, 21}, uniform masked mean, stride max(1, ⌊N/2⌋). All three CSVs from one interpreter (NFO_RECOG_TAG=_ablN,
2026-10-20); absolute AP is only comparable within that set. 95% CI: window bootstrap (AP recomputed per resample, as
recognition.report).
N = 1 is the same for every arm by construction: unmasked pixels are filled with the plain aligned mean, which for a
single frame is the frame itself, so every stack is the raw frame (the no-integration reference point).

    python3 -m benchmark.nfo_vos.eval.recog_vs_N
    NFO_RUN=fullpass python3 -m benchmark.nfo_vos.eval.recog_vs_N   # the 32 whole crossings, seq1–4, + b0trk-c2
Whole crossings (2026-10-25): CSVs from eval/recog_sweep.sbatch (one per N and arm, _fpN_<arm>) and N = 5 from the
headline CSV (_fp); -> recog_vs_N_fullpass.png.
"""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.eval import box_metrics as BM
from benchmark.nfo_vos.eval.recognition import ap50

NS = (1, 3, 5, 7, 11, 15, 21)
# dataviz default categorical slots 1-4 (validated reference palette); no-mask baseline in neutral gray, dashed
ARMS = (('adapter-pf', 'adapter (ours, per-frame)', '#2a78d6', '-'), ('t1-c2', 'SAMURAI', '#eb6834', '-'),
        ('b0-c2', 'SAM2.1 video', '#1baf7a', '-'), ('trk', 'SAM2.1 per-frame (tracker box)', '#eda100', '-'),
        ('nomask', 'no mask', '#8a8984', '--'))
FULLPASS = TR.RUN == 'fullpass'
if FULLPASS:
    ARMS = ARMS[:4] + (('b0trk-c2', 'SAM2.1 video + tracker box', '#7a5ac8', '-'),) + ARMS[4:]
SEQS = TR.ALL_SEQS if FULLPASS else ('seq2', 'seq3', 'seq4')     # whole crossings: the seq1–4 headline
OUT = f"{BM.IMG}/recog_vs_N{'_fullpass' if FULLPASS else ''}.png"


def ap_ci(rows, arm, B=1000):
    win = {}
    for r in rows:
        if r['arm'] == arm and r['seq'] in SEQS:
            win.setdefault(r['id'], []).append([tuple(x) for x in json.loads(r['dets'])])   # one det list per stack
    ids = sorted(win)
    boot = np.random.default_rng(0).integers(0, len(ids), (B, len(ids)))
    bs = [ap50([d for i in b for d in win[ids[i]]]) for b in boot]
    return ap50([d for i in ids for d in win[i]]), np.quantile(bs, .025), np.quantile(bs, .975)


def main():
    res = {}
    for N in NS:
        if not FULLPASS:
            rows = list(csv.DictReader(open(f'{BM.OUT}/recog_N{N}_uniform_meanfill_ablN.csv')))
        for a, *_ in ARMS:
            if FULLPASS:
                rows = list(csv.DictReader(open(f"{BM.OUT}/recog_N{N}_uniform_meanfill_{'fp' if N == 5 else 'fpN_' + a}.csv")))
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
    ax.set_xlabel('frames per stack N'); ax.set_ylabel('YOLO11m person AP50')
    ax.set_title(f"Recognition on masked temporal stacks (NFO {'32 whole crossings, seq1–4' if FULLPASS else 'seq2–4'}, 95% CI)",
                 fontsize=10, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.subplots_adjust(right=0.68)
    fig.savefig(OUT)
    print('wrote', OUT)


if __name__ == '__main__':
    main()


def montage(Ns=(7, 21), arms=(('nomask', 'no mask'), ('adapter-pf', 'adapter'), ('t1-c2', 'SAMURAI'))):
    """Stacks ending at the last frame of the first seq2 / seq3 / seq4 window, rows = window × N, columns = arm,
    cropped around the GT box (green); YOLO's best-matching person box in magenta with its confidence. For judging
    why masked stacks lose to the plain mean at long N (seams vs ghost limbs)."""
    import cv2
    from ultralytics import YOLO as Y
    from benchmark.nfo_vos.eval.recognition import YOLO, arm_masks, background, frame, gt_shifts, \
        masked_stack, score_yolo
    model = Y(YOLO)
    trials = json.load(open(TR.OUT))
    rows = []
    for seq in ('seq2', 'seq3', 'seq4'):
        t = next(x for x in trials if x['seq'] == seq)
        gtb, fr = BM.gt_boxes(seq), t['frames']
        bg = background(seq, *t['warmup'])
        j = len(fr) - 1
        g = gtb[fr[j]]
        cx, cy, w, h = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2, max(g[2] - g[0], 60) * 2.4, (g[3] - g[1]) * 1.6
        X0, Y0, X1, Y1 = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)), int(min(cx + w / 2, 800)), int(min(cy + h / 2, 600))
        for N in Ns:
            idx = list(range(j - N + 1, j + 1))
            panes = []
            for a, name in arms:
                M, _ = arm_masks(a, t, gtb)
                C = masked_stack([frame(seq, fr[i]) for i in idx], [M[i] for i in idx],
                                     gt_shifts(gtb, [fr[i] for i in idx], fr[j]), bg)
                s = score_yolo(model, C, g)
                v = C.copy()
                cv2.rectangle(v, tuple(int(x) for x in g[:2]), tuple(int(x) for x in g[2:]), (0, 255, 0), 1)
                if s['box'] is not None:
                    cv2.rectangle(v, tuple(int(x) for x in s['box'][:2]), tuple(int(x) for x in s['box'][2:]), (255, 0, 255), 1)
                p = cv2.resize(v[Y0:Y1, X0:X1], (240, 320))
                cv2.putText(p, f'{seq} N={N} {name}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                cv2.putText(p, f'conf {s["conf"]:.2f}', (4, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                panes.append(p)
            rows.append(np.hstack(panes))
    path = f'{BM.IMG}/stack_N7_vs_N21.jpg'
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('wrote', path)
