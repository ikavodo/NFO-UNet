"""One-pass tracking metrics on NFO (docs/vos_benchmark/plan_downstream_benchmark.md, Task 1; D2).

Every arm's masks become boxes the way SAMURAI's own inference does (samurai@76ba195 scripts/main_inference.py:92-98:
tight box of the nonzero pixels, w = x_max − x_min, empty mask → [0, 0, 0, 0]), and are scored with the LaSOT one-pass
protocol as ported below from samurai@76ba195 lib/test/analysis/extract_results.py (pytracking):
  AUC     mean over τ ∈ {0, 0.05, …, 1} of the share of frames with IoU > τ
  P       share of frames with centre error ≤ 20 px
  P_norm  share of frames with centre error / GT size ≤ 0.20
  fail    share of frames with IoU = 0 (stand-in for VOT robustness; EAO/A/R are out, D2)
A window (40 frames) is one sequence; group scores are means over windows.

  python -m benchmark.nfo_vos.eval.box_metrics ceiling            # hand-mask box vs GT box, 43 hand frames
  python -m benchmark.nfo_vos.eval.box_metrics score b0 t1 ...   # results/benchmark/rr/downstream/box_gt.csv
"""
import csv
import functools
import json
import os
import sys

import numpy as np

from benchmark.nfo_vos import trials as TR
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

MASKS = 'results/benchmark/rr/masks'
OUT = 'results/benchmark/rr/downstream'
IMG = 'images/benchmark/downstream'
# samurai@76ba195 lib/test/analysis/extract_results.py:115-117 (plot_bin_gap 0.05); P / P_norm read at index 20
TH_OVERLAP = np.arange(0.0, 1.0 + 0.05, 0.05)
TH_CENTER = np.arange(0, 51, dtype=np.float64)
TH_CENTER_NORM = np.arange(0, 51, dtype=np.float64) / 100.0


def mask_box(m):
    """Tight xyxy box (inclusive max) of a bool mask, None if empty."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)


def _xywh(b):
    return np.zeros(4) if b is None else np.array([b[0], b[1], b[2] - b[0], b[3] - b[1]], np.float64)


def _err_center(p, a, normalized=False):
    # samurai@76ba195 extract_results.py:17-26
    pc, ac = p[:, :2] + 0.5 * (p[:, 2:] - 1.0), a[:, :2] + 0.5 * (a[:, 2:] - 1.0)
    if normalized:
        pc, ac = pc / a[:, 2:], ac / a[:, 2:]
    return np.sqrt(((pc - ac) ** 2).sum(1))


def _iou(p, a):
    # samurai@76ba195 extract_results.py:29-38
    tl = np.maximum(p[:, :2], a[:, :2])
    br = np.minimum(p[:, :2] + p[:, 2:] - 1.0, a[:, :2] + a[:, 2:] - 1.0)
    inter = np.clip(br - tl + 1.0, 0, None).prod(1)
    return inter / (p[:, 2:].prod(1) + a[:, 2:].prod(1) - inter)


def ope(pred, gt):
    """pred: list of xyxy | None; gt: list of xyxy (same length). One sequence's scores."""
    p, a = np.stack([_xywh(b) for b in pred]), np.stack([_xywh(b) for b in gt])
    iou, ce, cn = _iou(p, a), _err_center(p, a), _err_center(p, a, normalized=True)
    return dict(auc=float((iou[:, None] > TH_OVERLAP[None]).mean(0).mean()),
                p20=float((ce[:, None] <= TH_CENTER[None]).mean(0)[20]),
                pnorm=float((cn[:, None] <= TH_CENTER_NORM[None]).mean(0)[20]),
                fail=float((iou <= 0).mean()), iou=iou, cerr=ce)


@functools.lru_cache(maxsize=None)
def gt_boxes(seq):
    """{frame: native xyxy} for every frame with a GT box (first box of the frame)."""
    return {k: np.array(gt_to_native(v[0], TR.NATIVE_W, TR.NATIVE_H), np.float64) for k, v in TR.bbs(seq).items()}


def load_arm(arm, trial_id):
    """(masks [T,600,800] bool, frames [T]) or None if the arm has no file for the window."""
    p = f'{MASKS}/{arm}/{trial_id}.npz'
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d['masks'], d['frames']


def hand_frames():
    """[(seq, frame, person-mask bool)] for the 43 hand frames: seq1 brush masks, seq2–4 handAv (value 1 = person)."""
    return [(seq, k, TR.hand_label(seq, k)[0]) for seq in TR.ALL_SEQS for k in TR.hand_frames(seq)]


def ceiling():
    """Task 1 Step 5: how far is a perfect visible-part mask's box from the GT box?"""
    rows = []
    for seq, k, h in hand_frames():
        g = gt_boxes(seq).get(k)
        if g is None:
            continue
        r = ope([mask_box(h)], [g])
        rows.append((seq, k, float(r['iou'][0]), float(r['cerr'][0]), h, g))
    print(f'{"group":8s} n   IoU mean / median   centre err px mean')
    for name, sel in [('seq1', ('seq1',)), ('seq2-4', ('seq2', 'seq3', 'seq4')), ('all', ('seq1', 'seq2', 'seq3', 'seq4'))]:
        r = [x for x in rows if x[0] in sel]
        iou = np.array([x[2] for x in r])
        print(f'{name:8s} {len(r):2d}  {iou.mean():.3f} / {np.median(iou):.3f}       {np.mean([x[3] for x in r]):.1f}')
    _ceiling_montage(sorted(rows, key=lambda x: x[2])[:6])
    return rows


def _ceiling_montage(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmark.nfo_vos import trials as TR
    os.makedirs(IMG, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.5))
    for a, (seq, k, iou, ce, h, g) in zip(ax.flat, rows):
        im = TR.native(k, seq).astype(np.float32) / 255
        im = np.repeat(im[..., None], 3, -1) if im.ndim == 2 else im
        im[h] = 0.5 * im[h] + 0.5 * np.array([1, 0, 1])
        a.imshow(im)
        b = mask_box(h)
        for bb, c in ((g, 'lime'), (b, 'magenta')):
            a.add_patch(plt.Rectangle(bb[:2], bb[2] - bb[0], bb[3] - bb[1], fill=False, ec=c, lw=1.5))
        a.set_title(f'{seq} {k}  IoU {iou:.2f}', fontsize=9)
        a.axis('off')
    fig.suptitle('6 lowest IoU: GT box (green) vs hand-mask box (magenta), hand mask tinted')
    fig.tight_layout()
    fig.savefig(f'{IMG}/hand_vs_gt_box.png', dpi=80)
    plt.close(fig)


def score(arms, ref='gt'):
    """Per window × arm scores vs the reference boxes; prints group means and window-bootstrap Δ vs t1."""
    trials = json.load(open('results/benchmark/rr/trials.json'))
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for t in trials:
        gtb = gt_boxes(t['seq'])
        for arm in arms:
            la = load_arm(arm, t['id'])
            if la is None:
                continue
            ms, fr = la
            keep = [i for i, k in enumerate(fr) if int(k) in gtb]
            r = ope([mask_box(ms[i]) for i in keep], [gtb[int(fr[i])] for i in keep])
            rows.append(dict(arm=arm, id=t['id'], seq=t['seq'], n=len(keep), **{k: r[k] for k in ('auc', 'p20', 'pnorm', 'fail')}))
    with open(f'{OUT}/box_{ref}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, arms)
    return rows


def report(rows, arms, base=os.environ.get('NFO_BASE', 't1'), B=2000):
    rng = np.random.default_rng(0)
    for name, sel in [('seq1 (selection)', ('seq1',)), ('seq2-4 (test)', ('seq2', 'seq3', 'seq4'))]:
        print(f'\n{name}\n{"arm":14s} n_win   AUC    P@20   P_norm  fail    ΔAUC vs {base} [95% CI]')
        by = {a: {r['id']: r for r in rows if r['arm'] == a and r['seq'] in sel} for a in arms}
        for a in arms:
            if not by[a]:
                continue
            m = {k: np.mean([r[k] for r in by[a].values()]) for k in ('auc', 'p20', 'pnorm', 'fail')}
            ci = ''
            ids = sorted(set(by[a]) & set(by.get(base, {})))
            if a != base and ids:
                d = np.array([by[a][i]['auc'] - by[base][i]['auc'] for i in ids])
                bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
                ci = f'{d.mean():+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]'
            print(f'{a:14s} {len(by[a]):3d}   {m["auc"]:.3f}  {m["p20"]:.3f}  {m["pnorm"]:.3f}   {m["fail"]:.3f}   {ci}')


if __name__ == '__main__':
    if sys.argv[1] == 'ceiling':
        ceiling()
    else:
        score(sys.argv[2:])
