"""Two follow-ups to factorised_gtfree.py (2026-10-14), CPU only, reuse cached M_C masks.

1. Prompt fallback: blob_box where the blob tracker has a merged box at k, else blob_point (alignment t1 and blob).
2. Transferable threshold: replace pix_bg > 6.31 (raw grey-level units, seq1-specific) by a per-frame robust z of
   pix_bg (median / 1.4826 MAD over the whole frame) > τ_z, τ_z chosen leave-one-frame-out on the brush frames
   (max mean ΔJ), evaluated vs polygons. Arm: t1 alignment + fallback prompt. Equivalence bar (not a pass/fail on
   the method): |J(z) − J(raw)| ≤ 0.01 vs polygons → use z for other sequences.
"""
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, SEQ, stats
from benchmark.nfo_vos.factorised_gtfree import OUT

TAU = 6.31


def pixz(p):
    med = np.median(p); mad = 1.4826 * np.median(np.abs(p - med))
    return (p - med) / max(mad, 1e-6)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    rows = list(np.load(f'{OUT}/rows.npy', allow_pickle=True))
    rng = np.random.default_rng(0)
    D = {}
    for r in rows:
        k = r['k']; t = next(x for x in T if x['id'] == r['id']); j = t['frames'].index(k)
        w0, w1 = t['warmup']
        bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
        pb = np.log(cv2.boxFilter((V.native(k, SEQ).astype(np.float32) - bg) ** 2, -1, (5, 5)) + 1)
        T1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks'][j].astype(bool)
        mc = {}
        for a in ('t1', 'blob'):
            p = f'{OUT}/MC_{k:05d}_{a}_blob_box.npy'
            mc[a] = np.load(p) if os.path.exists(p) else np.load(f'{OUT}/MC_{k:05d}_{a}_blob_point.npy')
        mc['blob_pt'] = np.load(f'{OUT}/MC_{k:05d}_blob_blob_point.npy')   # corrected chosen arm
        H = {s: cv2.imread(f'{GT}/{k:05d}_{s}.png', 0) == 1 for s in ('handpoly', 'handmask_raw')
             if os.path.exists(f'{GT}/{k:05d}_{s}.png')}
        D[k] = dict(pb=pb, z=pixz(pb), T1=T1, mc=mc, H=H)

    def dJ(suf, a, key, tau, ks=None):
        ks = [k for k in D if suf in D[k]['H'] and (ks is None or k in ks)]
        return np.array([stats((D[k]['mc'][a] & (D[k][key] > tau)) | D[k]['T1'], D[k]['H'][suf])[0]
                         - stats(D[k]['T1'], D[k]['H'][suf])[0] for k in ks])

    def ci(d):
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        return np.percentile(bs, [2.5, 97.5])

    print('1. fallback prompt (blob_box else blob_point), raw τ = 6.31: ΔJ(V̂_T1 − T1)')
    for a in ('t1', 'blob'):
        for suf in ('handmask_raw', 'handpoly'):
            d = dJ(suf, a, 'pb', TAU); lo, hi = ci(d)
            print(f'  align {a:4s} vs {suf:12s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], better on {(d > 0).sum()}/{len(d)}')

    print('\n2. robust-z threshold (chosen arm: blob alignment, blob_point prompt)')
    grid = np.arange(0.0, 12.01, 0.25)
    bk = [k for k in D if 'handmask_raw' in D[k]['H']]
    tz_all = grid[np.argmax([dJ('handmask_raw', 'blob_pt', 'z', g).mean() for g in grid])]
    print(f'  τ_z chosen on all brush frames: {tz_all:.2f}; raw τ 6.31 corresponds to z of '
          f'{np.median([np.interp(TAU, np.sort(D[k]["pb"].ravel()), np.sort(D[k]["z"].ravel())) for k in D]):.2f} (median over frames)')
    out = []
    for k in D:
        if 'handpoly' not in D[k]['H']:
            continue
        tz = tz_all if k not in bk else grid[np.argmax([dJ('handmask_raw', 'blob_pt', 'z', g, [x for x in bk if x != k]).mean() for g in grid])]
        h = D[k]['H']['handpoly']
        out.append((stats((D[k]['mc']['blob_pt'] & (D[k]['z'] > tz)) | D[k]['T1'], h)[0],
                    stats((D[k]['mc']['blob_pt'] & (D[k]['pb'] > TAU)) | D[k]['T1'], h)[0], stats(D[k]['T1'], h)[0]))
    o = np.array(out)
    lo, hi = ci(o[:, 0] - o[:, 2])
    print(f'  vs polygons (n={len(o)}): J T1 {o[:, 2].mean():.3f}, raw τ {o[:, 1].mean():.3f}, robust z {o[:, 0].mean():.3f}; '
          f'ΔJ(z − T1) {np.mean(o[:, 0] - o[:, 2]):+.3f} [{lo:+.3f}, {hi:+.3f}]; '
          f'|J(z) − J(raw)| {abs(o[:, 0].mean() - o[:, 1].mean()):.3f} -> {"use z" if abs(o[:, 0].mean() - o[:, 1].mean()) <= 0.01 else "not equivalent"}')


if __name__ == '__main__':
    main()
