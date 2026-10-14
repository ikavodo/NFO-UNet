"""Two gating tests for the composite strategies (S1 re-prompting, S2 attention prior, S3 stride-4 adapter), 2026-10-14.
Written before running; pass bars fixed here.

Test 1 (fine evidence, gates S3): on the 20 seq1 seg-4 brush frames, candidates = GT box (dilated 10 px) minus the
  pseudo-GT M; positive = brush person pixel (void 2 excluded), negative = the rest. Scores as in diag_fine_add.py
  (2026-10-12, then scored against the coarse pseudo-GT): empty-scene cues z16, z4, pix_bg and layered / correspondence
  scores against the raw, GT-free and oracle composites at pixels and stride-4 features. Report AUC pooled, and on the
  H2 class (positives in brush components < 1 stride-16 cell, i.e. sub-cell pieces, vs all negatives).
  PASS: best GT-free score AUC ≥ 0.80 on the sub-cell class.
Test 2 (extent, gates S1/S2): SAM2 image predictor (sam2.1 b+) on the GT-free composite C of frame k (history
  k-6..k-1, aligned onto k by GT box centres = oracle alignment), prompted like the pseudo-GT generator (GT box +
  its centre point), single-mask output → M_C. Scored vs polygons (17) and brush (20): M_C, M ∪ M_C, and the S1 prompt
  candidate M_C ∧ changed cells (stride-16 static z ≥ 2, descriptive, untuned).
  PASS: mean J(M ∪ M_C) > mean J(M) vs polygons with paired bootstrap CI > 0.
Window (warm-up / empty scene) for frame k = the seq1 trial window nearest to k.
"""
import json
import os

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.diag_composite_diff import feats, static_z
from benchmark.nfo_vos.diag_fine_add import composites, corr_max, min_ssd, up
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

SEQ, GT = 'seq1', 'data/nfo_processed/seq1_gt'
CELL = 800 / 64 * 600 / 64
OUT = f'{TR.RES}/composite_tests'
IMG = 'images/benchmark/rr_checks/composite_tests'


def window_for(k, T):
    return min(T, key=lambda t: 0 if t['frames'][0] <= k <= t['frames'][-1] else min(abs(k - t['frames'][0]), abs(k - t['frames'][-1])))


def box(k):
    b = TR.bbs(SEQ)[k][0]
    return [int(round(v)) for v in gt_to_native(b, TR.NATIVE_W, TR.NATIVE_H)]


def auc(s, y):
    from scipy.stats import rankdata   # AUC = Mann-Whitney U / (n+ n-)
    n1 = y.sum(); n0 = len(y) - n1
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def stats(a, h):
    tp = (a & h).sum()
    return np.array([tp / max((a | h).sum(), 1), tp / max(a.sum(), 1), tp / max(h.sum(), 1)])


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    tag = lambda t: sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith(f'_{t}.png'))
    brush, poly = tag('handmask_raw'), tag('handpoly')
    os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    pred, bgs = R.image_predictor(), {}
    S1, rows2, keep = {}, [], {}
    with torch.inference_mode():
        for k in sorted(set(brush) | set(poly)):
            t = window_for(k, T)
            if tuple(t['warmup']) not in bgs:
                w0, w1 = t['warmup']
                bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
                bgs[tuple(t['warmup'])] = (bg, feats(bg))
            bg, Fbg = bgs[tuple(t['warmup'])]
            C = composites(SEQ, k, bg)
            M = cv2.imread(f'{GT}/{k:05d}_sammask_native.png', 0) > 127
            if C is None:
                print(f'{k}: composite incomplete, skipped'); continue
            Rk = V.native(k, SEQ).astype(np.float32)
            FR = feats(Rk)
            z16 = up(static_z(FR[16], Fbg[16]))
            # Test 2: segment the GT-free composite
            x0, y0, x1, y1 = box(k)
            pred.set_image(np.stack([np.clip(C['gtfree'], 0, 255).astype(np.uint8)] * 3, -1))
            mc, _, _ = pred.predict(point_coords=np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]), point_labels=np.array([1]),
                                    box=np.array([x0, y0, x1, y1]), multimask_output=False)
            MC = mc[0] > 0
            cand = {'M': M, 'M_C': MC, 'M|M_C': M | MC, 'S1 prompt M_C&changed': MC & (z16 >= 2)}
            for ref, ks in (('poly', poly), ('brush', brush)):
                if k in ks:
                    h = cv2.imread(f'{GT}/{k:05d}_{"handpoly" if ref == "poly" else "handmask_raw"}.png', 0) == 1
                    rows2.append(dict(k=k, ref=ref, **{n: stats(a, h) for n, a in cand.items()}))
            keep[k] = (C['gtfree'], MC, M)
            # Test 1: fine evidence on brush frames
            if k in brush:
                Bm = cv2.imread(f'{GT}/{k:05d}_handmask_raw.png', 0)
                E = np.zeros((600, 800), bool); E[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = True
                candm = E & ~M & (Bm != 2)
                sl = (slice(max(0, y0 - 20), min(600, y1 + 20)), slice(max(0, x0 - 20), min(800, x1 + 20)))
                ssd_bg = cv2.boxFilter((Rk[sl] - bg[sl]) ** 2, -1, (5, 5))
                cbg = up(torch.nn.functional.cosine_similarity(FR[4].float(), Fbg[4].float(), dim=0))[sl]
                Sc = {'z16': z16[sl], 'z4': up(static_z(FR[4], Fbg[4]))[sl], 'pix_bg': np.log(ssd_bg + 1)}
                for c, Cc in C.items():
                    Sc[f'layer_pix_{c}'] = np.log(ssd_bg + 1) - np.log(min_ssd(Rk, Cc, sl) + 1)
                    cr = up(corr_max(FR[4], feats(Cc)[4]))[sl]
                    Sc[f'corr4_{c}'] = cr
                    Sc[f'layer_f4_{c}'] = np.log(1 - cbg + 1e-3) - np.log(1 - cr + 1e-3)
                n, lab, st, _ = cv2.connectedComponentsWithStats((Bm == 1).astype(np.uint8), connectivity=8)
                small = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < CELL])
                c_ = candm[sl]
                S1[k] = dict(y=(Bm == 1)[sl][c_], sub=small[sl][c_], **{a: v[c_] for a, v in Sc.items()})
            print(f'{k} done', flush=True)

    print('\nTEST 1: AUC, candidates = GT box minus pseudo-GT; pos = brush person (20 seg-4 frames)')
    arms = [a for a in next(iter(S1.values())) if a not in ('y', 'sub')]
    y = np.concatenate([d['y'] for d in S1.values()]); sub = np.concatenate([d['sub'] for d in S1.values()])
    print(f'  {len(y)} candidates, {y.sum()} positive, {(y & sub).sum()} in sub-cell components')
    best = (None, 0)
    for a in arms:
        s = np.concatenate([d[a] for d in S1.values()]).astype(np.float32)
        sel = ~y | sub
        a_all, a_sub = auc(s, y), auc(s[sel], (y & sub)[sel])
        print(f'  {a:20s} all {a_all:.3f}   sub-cell {a_sub:.3f}')
        if not a.endswith('oracle') and a_sub > best[1]:
            best = (a, a_sub)
    print(f'  best GT-free on sub-cell: {best[0]} {best[1]:.3f} -> {"PASS" if best[1] >= 0.80 else "FAIL"} (bar 0.80)')

    print('\nTEST 2: SAM2 image predictor on the GT-free composite, J / P / R')
    rng = np.random.default_rng(0)
    for ref in ('poly', 'brush'):
        rr = [r for r in rows2 if r['ref'] == ref]
        print(f'  vs {ref} (n={len(rr)})')
        for n in ('M', 'M_C', 'M|M_C', 'S1 prompt M_C&changed'):
            print(f'    {n:22s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([r[n] for r in rr], 0)))
        d = np.array([r['M|M_C'][0] - r['M'][0] for r in rr])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        verdict = '' if ref != 'poly' else (' -> PASS' if np.percentile(bs, 2.5) > 0 else ' -> FAIL')
        print(f'    ΔJ (M|M_C − M) {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]{verdict}')
    np.save(f'{OUT}/test2_rows.npy', rows2, allow_pickle=True)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = [k for k in poly if k in keep]
    fig, ax = plt.subplots(2, len(ks), figsize=(1.0 * len(ks) + 2, 5))
    for j, k in enumerate(ks):
        Cg, MC, M = keep[k]
        h = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1
        ys, xs = np.nonzero(h | MC | M)
        y0, y1, x0, x1 = max(0, ys.min() - 10), min(600, ys.max() + 10), max(0, xs.min() - 10), min(800, xs.max() + 10)
        for i, (img, ttl) in enumerate(((V.native(k, SEQ), str(k)), (Cg, 'C'))):
            ax[i, j].imshow(img[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255)
            for mm, col in ((h, 'cyan'), (M, 'lime'), (MC, 'magenta')):
                ax[i, j].contour(mm[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=col, linewidths=0.6)
            ax[i, j].set_title(ttl, fontsize=7); ax[i, j].axis('off')
    fig.suptitle('top: frame k, bottom: GT-free composite C. cyan = polygon, green = pseudo-GT, magenta = SAM2 on C',
                 fontsize=9)
    fig.tight_layout(); fig.savefig(f'{IMG}/composite_extent.png', dpi=75)


if __name__ == '__main__':
    main()
