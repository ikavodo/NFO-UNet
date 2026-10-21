"""Person recognition on masked temporal composites (docs/vos_benchmark/plan_downstream_benchmark.md, Task 4; D4).

For frame t of a window (local index ≥ N−1): frames t−N+1..t are translated onto t by the GT-box centre shift (same
motion for every arm, so only the masks differ), and combined as the uniform masked mean
    C(p) = Σ_i w_i M_i(p) I_i(p) / Σ_i w_i M_i(p),     C(p) = (1/N) Σ_i I_i(p) where Σ_i M_i(p) = 0
Fill = the plain aligned mean (2026-10-19, after the controls): the empty-scene fill re-inserted SHARP static foliage
around the fragments and lost to the unmasked mean (seq2–4 AP50 0.48 vs 0.80); with the mean fill a method scores
above 'nomask' only if its masked pixels are more recognisable than the synthetic-aperture blur (FP foliage re-sharpens,
FN falls back to the blur). fill='bg' keeps the first design. YOLO11m (data/yolo11m.pt,
COCO class 0) on C; scored against the GT box of t: AP50 over all composites of a sequence group (ultralytics
ap_per_class) and the confidence of the best person detection with IoU ≥ 0.5 (0 if none).
Ancestor: airborne optical sectioning (Kurmi / Schedl / Bimber), integral images of aligned views before person
classification; here the integration is temporal and gated by each segmentation method's masks.
Arms: any masks/<arm> plus controls 'nomask' (M ≡ 1: plain aligned mean, classic synthetic aperture) and 'gtbox'
(M = GT box of frame i). N = 1 is the single masked frame. NFO frames are grayscale (3 identical channels).

  python -m benchmark.nfo_vos.eval.recognition N WEIGHTS arm1 arm2 ...      (WEIGHTS: uniform | recency)
"""
import csv
import functools
import json
import os
import sys

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.eval import box_metrics as BM

YOLO = 'data/yolo11m.pt'
RECENCY = 0.7          # w_i = RECENCY^(t−i), sensitivity only (D4)


def gt_shifts(boxes, frames, t):
    """(dx, dy) per frame i that moves frame i onto frame t: GT-box centre of t minus centre of i."""
    c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    ct = c(boxes[t])
    return [(float(ct[0] - c(boxes[i])[0]), float(ct[1] - c(boxes[i])[1])) for i in frames]


def masked_composite(imgs, masks, shifts, bg, weights=None, fill='mean'):
    """Uniform (or weighted) masked mean of the shifted frames; where no mask covers a pixel: the plain aligned mean
    (fill 'mean') or the empty scene (fill 'bg')."""
    H, W = bg.shape[:2]
    w = np.ones(len(imgs)) if weights is None else np.asarray(weights, np.float64)
    num, den, plain = np.zeros(bg.shape, np.float64), np.zeros((H, W), np.float64), np.zeros(bg.shape, np.float64)
    for im, m, (dx, dy), wi in zip(imgs, masks, shifts, w):
        A = np.float32([[1, 0, dx], [0, 1, dy]])
        imw = cv2.warpAffine(im, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mw = cv2.warpAffine(m.astype(np.float32), A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        num += wi * mw[..., None] * imw
        plain += imw / len(imgs)
        den += wi * mw
    out = np.where(den[..., None] > 1e-6, num / np.maximum(den, 1e-6)[..., None], plain if fill == 'mean' else bg)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _iou(a, b):
    return float(BM.ope([a], [b])['iou'][0])


def score_yolo(model, img, gt_box):
    """All person detections (conf ≥ 0.01) as (conf, iou); plus the summary of the best match (IoU ≥ 0.5)."""
    r = model.predict(img, classes=[0], conf=0.01, verbose=False)[0]
    xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    dets = [(float(c), _iou(b, gt_box)) for b, c in zip(xyxy, conf)]
    hits = [c for c, i in dets if i >= 0.5]
    best = int(np.argmax(conf)) if len(conf) else None
    return dict(dets=dets, conf=max(hits) if hits else 0.0, box=None if best is None else xyxy[best])


def ap50(per_composite):
    """AP50 over composites, one GT person each; at most one TP per composite (highest-confidence match)."""
    from ultralytics.utils.metrics import ap_per_class
    tp, cf = [], []
    for dets in per_composite:
        used = False
        for c, i in sorted(dets, reverse=True):
            ok = i >= 0.5 and not used
            used |= ok
            tp.append(ok); cf.append(c)
    if not tp:
        return 0.0
    out = ap_per_class(np.array(tp)[:, None], np.array(cf), np.zeros(len(cf)), np.zeros(len(per_composite)))
    return float(out[5][0, 0])          # ap: [n_classes, n_iou]


@functools.lru_cache(maxsize=None)
def frame(seq, k):
    return cv2.imread(f'data/nfo_final/nfo_final/{seq}/{k:05d}.jpg')


@functools.lru_cache(maxsize=None)
def background(seq, w0, w1):
    return np.median(np.stack([frame(seq, i) for i in range(w0, w1 + 1, max(1, (w1 - w0) // 40))]), 0).astype(np.uint8)


def arm_masks(arm, t, gtb):
    """[T, H, W] bool masks of the window, plus avail [T] (False where the arm has no label for that frame), or None.
    Label arm 'sam2ceil' (SAM2-ceiling, *_sammask_native.png: SAM2.1-L, GT box at confirmed-clear frames, propagated
    both ways: uses future frames and the GT)."""
    T = len(t['frames'])
    if arm == 'sam2ceil':
        ms = [TR.gt_mask_native(k, t['seq']) for k in t['frames']]
        return np.stack([m if m is not None else np.zeros((TR.NATIVE_H, TR.NATIVE_W), bool) for m in ms]), np.array([m is not None for m in ms])
    M = _arm_masks(arm, t, gtb)
    return None if M is None else (M, np.ones(T, bool))


def _arm_masks(arm, t, gtb):
    if arm == 'nomask':
        return np.ones((len(t['frames']), TR.NATIVE_H, TR.NATIVE_W), bool)
    if arm == 'gtbox':
        m = np.zeros((len(t['frames']), TR.NATIVE_H, TR.NATIVE_W), bool)
        for j, k in enumerate(t['frames']):
            x0, y0, x1, y1 = np.round(gtb[k]).astype(int)
            m[j, max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
        return m
    la = BM.load_arm(arm, t['id'])
    return None if la is None else la[0]


def main(N=7, weights='uniform', arms=('nomask', 'gtbox', 't1'), stride=None, montage_ids=None):
    """stride: frames between composites; default ⌊N/2⌋ (half-overlapping windows, every frame still covered)."""
    from ultralytics import YOLO as Y
    model = Y(YOLO)
    N, trials = int(N), json.load(open('results/benchmark/rr/trials.json'))
    stride = stride or max(1, N // 2)
    rows, dets, keep = [], {}, {}
    for t in trials:
        gtb, fr = BM.gt_boxes(t['seq']), t['frames']
        if not all(k in gtb for k in fr):
            continue
        bg = background(t['seq'], *t['warmup'])
        av = [arm_masks(a, t, gtb) for a in arms]
        if any(a is None for a in av):
            continue                                    # every arm scored on the same composites
        cov = np.all([a[1] for a in av], 0)
        ok = [j >= N - 1 and cov[j - N + 1:j + 1].all() for j in range(len(fr))]
        for arm in arms:
            M = arm_masks(arm, t, gtb)
            if M is None:
                continue
            M, _ = M
            for j in range(N - 1, len(fr), stride):
                if not ok[j]:
                    continue
                idx = list(range(j - N + 1, j + 1))
                w = None if weights == 'uniform' else [RECENCY ** (j - i) for i in idx]
                C = masked_composite([frame(t['seq'], fr[i]) for i in idx], [M[i] for i in idx],
                                     gt_shifts(gtb, [fr[i] for i in idx], fr[j]), bg, w)
                s = score_yolo(model, C, gtb[fr[j]])
                rows.append(dict(arm=arm, id=t['id'], seq=t['seq'], frame=fr[j], conf=s['conf'],
                                 dets=json.dumps([[round(c, 4), round(i, 4)] for c, i in s['dets']])))
                dets.setdefault((arm, t['seq']), []).append(s['dets'])
                if montage_ids and t['id'] in montage_ids and j == max(i for i in range(len(fr)) if ok[i]):
                    keep[(t['id'], arm)] = (C, s['box'], gtb[fr[j]])
        print(t['id'], flush=True)
    os.makedirs(BM.OUT, exist_ok=True)
    with open(f"{BM.OUT}/recog_N{N}_{weights}_meanfill{os.environ.get('NFO_RECOG_TAG', '')}.csv", 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    report(rows, dets, arms)
    if keep:
        _montage(keep, arms, N, weights)
    return rows, dets


def report(rows, dets, arms, base=os.environ.get('NFO_BASE', 't1'), B=1000):
    """Group AP50 and mean conf; paired window-bootstrap Δ vs base for both (AP recomputed on each resample of windows)."""
    rng = np.random.default_rng(0)
    for name, sel in [('seq1 (selection)', ('seq1',)), ('seq2-4 (test)', ('seq2', 'seq3', 'seq4'))]:
        print(f'\n{name}\n{"arm":17s} n_comp  AP50   conf   ΔAP50 vs {base} [95% CI]      Δconf vs {base} [95% CI]')
        win = {a: {} for a in arms}
        for r in rows:
            if r['seq'] in sel:
                d = r['dets'] if isinstance(r['dets'], list) else json.loads(r['dets'])
                win[r['arm']].setdefault(r['id'], []).append((float(r['conf']), [tuple(x) for x in d]))
        ids = sorted(set.intersection(*[set(win[a]) for a in arms if win[a]]))
        boot = rng.integers(0, len(ids), (B, len(ids)))
        ap = lambda a, sub: ap50([d for i in sub for _, d in win[a][ids[i]]])
        cf = {a: np.array([np.mean([c for c, _ in win[a][i]]) for i in ids]) for a in arms if win[a]}
        apb = {a: np.array([ap(a, b) for b in boot]) for a in cf}
        for a in cf:
            full = ap(a, range(len(ids)))
            s_ = f'{a:17s} {sum(len(win[a][i]) for i in ids):5d}  {full:.3f}  {cf[a].mean():.3f}'
            if a != base and base in cf:
                dap = apb[a] - apb[base]; dc = cf[a] - cf[base]; dcb = dc[boot].mean(1)
                s_ += f'  {full - ap(base, range(len(ids))):+.3f} [{np.quantile(dap, .025):+.3f}, {np.quantile(dap, .975):+.3f}]'
                s_ += f'   {dc.mean():+.3f} [{np.quantile(dcb, .025):+.3f}, {np.quantile(dcb, .975):+.3f}]'
            print(s_)


def _montage(keep, arms, N, weights):
    ids = sorted({k[0] for k in keep})
    rows = []
    for i in ids:
        panes = []
        for a in arms:
            if (i, a) not in keep:
                continue
            C, b, g = keep[(i, a)]
            v = C.copy()
            cv2.rectangle(v, tuple(int(x) for x in g[:2]), tuple(int(x) for x in g[2:]), (0, 255, 0), 2)
            if b is not None:
                cv2.rectangle(v, tuple(int(x) for x in b[:2]), tuple(int(x) for x in b[2:]), (255, 0, 255), 2)
            cv2.putText(v, f'{i} {a}', (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panes.append(cv2.resize(v, (400, 300)))
        rows.append(np.hstack(panes))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    os.makedirs(BM.IMG, exist_ok=True)
    cv2.imwrite(f'{BM.IMG}/composite_N{N}_{weights}_meanfill.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    if sys.argv[1] == 'report':                      # re-report from the saved CSV, no YOLO
        rows = list(csv.DictReader(open(f'{BM.OUT}/recog_N7_uniform_meanfill.csv')))
        report(rows, None, list(dict.fromkeys(r['arm'] for r in rows))); sys.exit()
    trials = json.load(open('results/benchmark/rr/trials.json'))
    ids = {trials[0]['id']} | {next(x['id'] for x in trials if x['seq'] == s) for s in ('seq2', 'seq3')}
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3:]), montage_ids=ids)
