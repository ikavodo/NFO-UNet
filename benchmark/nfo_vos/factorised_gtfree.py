"""V̂_T1 without GT (2026-10-14, fixed before running). V̂_T1 = (M_C ∧ pix_bg > τ) ∪ T1, τ = 6.31 fixed
(factorised_label.py). Replaces the two oracle inputs of factorised_t1.py:

1. Composite alignment (history k-6..k-1 shifted onto frame k):
   gt    GT box centres (oracle reference)
   blob  the independent causal blob tracker (run_t4.track, started from the benchmark's t0 box): Theil–Sen fit
         of its chain x over local frames [j-12, j], x-only (its motion model is horizontal), dy = 0
   t1    centroids of T1's own masks over [j-12, j], Theil–Sen fit in x and y
2. Prompt for the SAM2 image predictor on the composite (the composite is in frame k's coordinates, so prompts
   computed at frame k need no warping):
   gt_box      GT box + its centre (oracle reference)
   blob_box    blob tracker's merged box at k (merged_center) + its centre;  blob_point  merged centre only
   t1_box      bbox of T1's mask at k + interior point;  t1_point  interior point only
   (interior point = distance-transform peak of T1's mask after a 15 px closing)
Arms: alignment ablation with gt_box; prompt ablation for each GT-free alignment.
Selection: the fully GT-free arm with the best mean ΔJ vs BRUSH; reported vs POLYGONS.
PASS: chosen GT-free arm ΔJ(V̂_T1 − T1) vs polygons, paired bootstrap CI > 0.
Frames: seq1 hand frames inside a trial window with local index j ≥ 6 (history inside the window).
Strata: window T1 init quality = mean J(T1, SAM2-ceiling) over local frames 1..5 (≥ 0.5 'good init').
"""
import json
import os

import cv2
import numpy as np
import torch
from scipy.stats import theilslopes

from benchmark.nfo_vos import a4, run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, SEQ, box, stats
from benchmark.nfo_vos.diag_composite_diff import box_centre, shift
from benchmark.nfo_vos.diag_fine_add import zpix
from tracking.core.blob_tracker import merged_center

TAU = 6.31
OUT = f'{TR.RES}/factorised_gtfree'
IMG = 'images/benchmark/rr_checks/composite_tests'
ARMS = [('gt', 'gt_box')] + [(a, p) for a in ('blob', 't1') for p in ('gt_box', 'blob_box', 'blob_point', 't1_box', 't1_point')]


def theil_at(ts, xs, t_eval):
    if len(ts) < 2:
        return {t: (xs[-1] if xs else np.nan) for t in t_eval}
    s, b, _, _ = theilslopes(xs, ts)
    return {t: s * t + b for t in t_eval}


def interior_point(m):
    mc = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    d = cv2.distanceTransform(mc, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(d), d.shape)
    return float(x), float(y)


def composite(seq, k, shifts, bg):
    """GT-free cleaned composite of k-6..k-1 onto k (diag_fine_add.composites, 'gtfree' arm) with given shifts."""
    H = list(range(k - a4.N + 1, k))
    return composite_frames([V.native(i, seq).astype(np.float32) for i in H],
                            [np.float32([[1, 0, shifts[i][0]], [0, 1, shifts[i][1]]]) for i in H], bg)


def composite_frames(imgs, Ms, bg):
    """The same composite for given history frames (oldest first) and 2×3 affines Ms onto the target frame
    (translation-only Ms reproduce `composite`; the layers baselines pass box similarity transforms)."""
    w = np.exp(-np.arange(len(imgs) - 1, -1, -1) / a4.TAU_AGE); w = (w / w.sum())[:, None, None]
    wa = lambda x, M: cv2.warpAffine(x.astype(np.float32), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    A = np.stack([wa(img, M) for img, M in zip(imgs, Ms)])
    G = np.stack([wa(1 / (1 + np.exp(-(zpix(img, bg) - 3))), M) for img, M in zip(imgs, Ms)])
    raw = (w * A).sum(0)
    return np.where((w * G).sum(0) > 1e-6, (w * G * A).sum(0) / np.maximum((w * G).sum(0), 1e-6), raw)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    tag = lambda s: sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith(f'_{s}.png'))
    hand = {k: ('handpoly',) for k in tag('handpoly')}
    for k in tag('handmask_raw'):
        hand[k] = hand.get(k, ()) + ('handmask_raw',)
    os.makedirs(OUT, exist_ok=True)
    pred, rows, align_err, trk = R.image_predictor(), [], {'blob': [], 't1': []}, {}
    for k in sorted(hand):
        win = [t for t in T if k in t['frames']]
        if not win:
            continue
        t = win[0]; j = t['frames'].index(k)
        if j < 6:
            print(f'{k}: local index {j} < 6, history outside the window, skipped'); continue
        if t['id'] not in trk:
            trk[t['id']] = R.track(t)
        Tk = trk[t['id']]
        M1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks'].astype(bool)
        cer = lambda i: cv2.imread(f'{GT}/{i:05d}_sammask_native.png', 0) > 127
        init_q = np.mean([stats(M1[i], cer(t['frames'][i]))[0] for i in range(1, 6)])
        w0, w1 = t['warmup']
        bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
        H = list(range(k - 6, k)); loc = lambda i: t['frames'].index(i)
        span = [i for i in range(max(0, j - 12), j + 1)]
        # alignment shifts {history frame: (dx, dy)} native px
        sh = {}
        g0 = box_centre(SEQ, k); sh['gt'] = {i: (g0[0] - box_centre(SEQ, t['frames'][loc(i)])[0],
                                                 g0[1] - box_centre(SEQ, t['frames'][loc(i)])[1]) for i in H}
        cs = [l for l in span if l in Tk['chain']]
        fx = theil_at(cs, [TR.to_native(*Tk['chain'][l])[0] for l in cs], [j] + [loc(i) for i in H])
        sh['blob'] = {i: (fx[j] - fx[loc(i)], 0.0) for i in H}
        cm = [(l, *np.argwhere(M1[l])[:, ::-1].mean(0)) for l in span if M1[l].any()]
        fx = theil_at([c[0] for c in cm], [c[1] for c in cm], [j] + [loc(i) for i in H])
        fy = theil_at([c[0] for c in cm], [c[2] for c in cm], [j] + [loc(i) for i in H])
        sh['t1'] = {i: (fx[j] - fx[loc(i)], fy[j] - fy[loc(i)]) for i in H}
        for a in ('blob', 't1'):
            align_err[a].append(np.median([abs(sh[a][i][0] - sh['gt'][i][0]) for i in H]))
        # prompts at frame k (native)
        anchor = Tk['chain'].get(j) or tuple(R.readout_line(Tk['chain'], j)[:2])   # chain gap: tracker's own prediction
        cx, cy, bb = merged_center(Tk['dets'][j], *anchor, Tk['kw']['merge_radius'], return_box=True)
        P = {'gt_box': (box(k), None)}
        if bb is not None:
            (bx0, by0), (bx1, by1) = TR.to_native(bb[0], bb[1]), TR.to_native(bb[2], bb[3])
            P['blob_box'] = ([bx0, by0, bx1, by1], TR.to_native(cx, cy))
        P['blob_point'] = (None, TR.to_native(cx, cy))
        if M1[j].any():
            ys, xs = np.nonzero(M1[j]); ip = interior_point(M1[j])
            P['t1_box'] = ([xs.min(), ys.min(), xs.max(), ys.max()], ip); P['t1_point'] = (None, ip)
        Rk = V.native(k, SEQ).astype(np.float32)
        change = np.log(cv2.boxFilter((Rk - bg) ** 2, -1, (5, 5)) + 1) > TAU
        Cs, r = {}, {'k': k, 'id': t['id'], 'good_init': init_q >= 0.5}
        for a, p in ARMS:
            if p not in P:
                continue
            if a not in Cs:
                Cs[a] = composite(SEQ, k, sh[a], bg)
            bx_, pt = P[p] if p != 'gt_box' else (P[p][0], ((P[p][0][0] + P[p][0][2]) / 2, (P[p][0][1] + P[p][0][3]) / 2))
            with torch.inference_mode():
                pred.set_image(np.stack([np.clip(Cs[a], 0, 255).astype(np.uint8)] * 3, -1))
                m, _, _ = pred.predict(point_coords=np.array([pt]), point_labels=np.array([1]),
                                       box=None if bx_ is None else np.array(bx_, dtype=np.float32), multimask_output=False)
            MC = m[0] > 0
            np.save(f'{OUT}/MC_{k:05d}_{a}_{p}.npy', MC)
            for suf in hand[k]:
                h = cv2.imread(f'{GT}/{k:05d}_{suf}.png', 0) == 1
                r[(suf, a, p)] = stats((MC & change) | M1[j], h)[0]
                r[(suf, 'T1')] = stats(M1[j], h)[0]
        rows.append(r)
        print(f'{k} ({t["id"]}, j={j}, init {"good" if r["good_init"] else "bad"}) done', flush=True)

    rng = np.random.default_rng(0)
    print(f'\nalignment error vs GT shifts (median |dx| per frame, native px): blob {np.median(align_err["blob"]):.1f}, '
          f'T1 {np.median(align_err["t1"]):.1f}  (n={len(align_err["t1"])} frames)')
    def delta(suf, a, p, sub=None):
        rr = [r for r in rows if (suf, a, p) in r and (sub is None or r['good_init'] == sub)]
        return np.array([r[(suf, a, p)] - r[(suf, 'T1')] for r in rr])
    def ci(d):
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        return np.percentile(bs, [2.5, 97.5])
    print('\nΔJ(V̂_T1 − T1), mean [95% CI] (n)')
    print(f'  {"alignment":6s} {"prompt":10s}   vs brush                      vs polygons')
    for a, p in ARMS:
        db, dp = delta('handmask_raw', a, p), delta('handpoly', a, p)
        fmt = lambda d: f'{d.mean():+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}] ({len(d)})' if len(d) > 1 else 'n/a'
        print(f'  {a:6s} {p:10s}   {fmt(db):28s}  {fmt(dp)}')
    free = [(a, p) for a, p in ARMS if a != 'gt' and p != 'gt_box']
    best = max(free, key=lambda ap: delta('handmask_raw', *ap).mean() if len(delta('handmask_raw', *ap)) else -9)
    d = delta('handpoly', *best); lo, hi = ci(d)
    print(f'\nchosen GT-free arm (on brush): {best}; vs polygons {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], '
          f'better on {(d > 0).sum()}/{len(d)} -> {"PASS" if lo > 0 else "FAIL"}')
    for sub, name in ((True, 'good T1 init'), (False, 'bad T1 init')):
        for a, p in (('gt', 'gt_box'), best, ('t1', best[1])):
            dd = np.concatenate([delta('handpoly', a, p, sub), delta('handmask_raw', a, p, sub)])
            print(f'  stratum {name:13s} {a:4s}/{p:10s} ΔJ pooled polygon+brush {dd.mean() if len(dd) else np.nan:+.3f} (n={len(dd)})')
    np.save(f'{OUT}/rows.npy', rows, allow_pickle=True)


if __name__ == '__main__':
    main()
