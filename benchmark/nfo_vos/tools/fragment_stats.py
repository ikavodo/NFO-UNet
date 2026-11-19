"""Fragmentation statistics of the visible person: NFO hand labels vs the synthetic occluder modes (is the training
occlusion model compatible with NFO?). CPU only. All statistics are scale-free (person height h = GT / silhouette box):
  vis     |V ∩ box| / |box|                                   ncomp   components of V in the box (area ≥ (0.02 h)²)
  comp    median component area / h²                         per     boundary pixels of V · h / |V|
  gap     median distance from a gap pixel to V, / h; gaps = closing(V, 0.15 h disk) ∧ ¬V (envelope proxy, no S needed)
  speed   |Δ box centre x| / h per frame (how fast history reveals new parts)
    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.fragment_stats
"""
import numpy as np
import cv2

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter import dataset as D
from benchmark.nfo_vos.adapter.fusion_data import BG_SPLIT
from benchmark.nfo_vos.synth import gpjatk_foliage as GF

OUT = 'results/benchmark/rr/feat_integral/fragment_stats.csv'
KEYS = ('vis', 'ncomp', 'comp', 'per', 'gap')


def stats(V, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h = max(y1 - y0, 1)
    v = V[max(y0, 0):y1, max(x0, 0):x1].astype(np.uint8)
    if v.sum() < 10:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(v, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]; areas = areas[areas >= (0.02 * h) ** 2]
    bnd = v & ~cv2.erode(v, np.ones((3, 3), np.uint8))
    r = max(int(0.075 * h), 1)
    env = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    gaps = env.astype(bool) & ~v.astype(bool)
    dt = cv2.distanceTransform((1 - v).astype(np.uint8), cv2.DIST_L2, 3)
    return dict(vis=v.sum() / v.size, ncomp=len(areas), comp=float(np.median(areas)) / h ** 2 if len(areas) else 0.0,
                per=bnd.sum() * h / v.sum(), gap=float(np.median(dt[gaps])) / h if gaps.any() else 0.0)


def nfo():
    rows = []
    for seq in ('seq1', 'seq2', 'seq3', 'seq4'):
        bb = TR.bbs(seq)
        for k in TR.hand_frames(seq):
            V = TR.hand_label(seq, k)[0]
            b = bb[k][0]
            s = max(TR.NATIVE_W, TR.NATIVE_H)
            x0, y0 = b.x * s - (s - TR.NATIVE_W) / 2, b.y * s - (s - TR.NATIVE_H) / 2
            r = stats(V, (x0, y0, x0 + b.w * s, y0 + b.h * s))
            if r:
                c = [bb[j][0].x + bb[j][0].w / 2 for j in (k - 1, k + 1) if j in bb]
                r['speed'] = abs(c[-1] - c[0]) / 2 / b.h if len(c) == 2 else np.nan          # frames k−1 → k+1
                rows.append(dict(src='NFO', **r))
    return rows


def synth(mode, clump=24, n_clips=12):
    GF.set_bg_pool(BG_SPLIT['val'])
    rows = []
    for seq, cam, rho, seed in D.clip_specs('val')[:n_clips]:
        *_, L = GF.make_layered_clip(seq, cam, rho, seed, mode, None, n=12, clump=clump, save=False, layers=True)
        S, V = L['S'], L['V']
        cs = []
        for t in (3, 6, 9):
            if not S[t].any():
                continue
            ys, xs = np.nonzero(S[t]); box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            r = stats(V[t], box)
            if r:
                c0 = [np.nonzero(S[j])[1].mean() for j in (t - 1, t + 1) if S[j].any()]
                r['speed'] = abs(c0[-1] - c0[0]) / 2 / (box[3] - box[1]) if len(c0) == 2 else np.nan
                rows.append(dict(src=f'{mode}{"" if mode != "perlin" else "-" + str(clump)}', **r))
    return rows


def main():
    import csv
    rows = nfo()
    for m, c in (('perlin', 12), ('perlin', 24), ('perlin', (12, 24)), ('stack', 24), ('depth', 24)):
        try:
            rows += synth(m, c)
        except Exception as e:                       # a mode whose assets are missing: report, keep going
            print(f'{m} {c}: {type(e).__name__}: {e}')
    srcs = list(dict.fromkeys(r['src'] for r in rows))
    print(f"{'source':>14} {'n':>3} | " + ' '.join(f'{k:>12}' for k in KEYS + ('speed',)) + '   (median [p25, p75])')
    for s in srcs:
        R = [r for r in rows if r['src'] == s]
        q = lambda k: np.nanpercentile([r[k] for r in R], [25, 50, 75])
        print(f'{s:>14} {len(R):>3} | ' + ' '.join(f'{q(k)[1]:5.3f}[{q(k)[0]:.2f},{q(k)[2]:.2f}]'[:12].rjust(12) if k != 'ncomp'
                                                 else f'{q(k)[1]:4.0f} [{q(k)[0]:.0f},{q(k)[2]:.0f}]'.rjust(12) for k in KEYS + ('speed',)))
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['src', *KEYS, 'speed']); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
