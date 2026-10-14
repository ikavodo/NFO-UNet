"""seq2–4 tier A re-scored with the annotator's void regions ignored (labels `handAv`: 1 person, 2 void; 2026-10-14).

Arms (all GT-free except the SAM2-ceiling, all cached, nothing re-run):
  T1 SAMURAI, B0 SAM2.1, SAM2-ceiling, V̂ anchored (u* = 0.85, fixed after the LOSO fit in learned_thresh.py),
  Diffusion-VAS amodal (SAM2-ceiling input, seed 23, results/benchmark/rr/amodal_dvas_<seq>).
Void pixels are dropped from union, intersection, P and R alike (DAVIS/VOC convention for ignore regions).
Also: share of the label that is void, recall of the labelled person by component size (stride-16 cells), and
where T1's missed (non-void) area lies.
"""
import cv2
import numpy as np

from benchmark.nfo_vos.learned_thresh import CELL, add_anchored, frames, load

U = 0.85


def lab(seq, k):
    v = cv2.imread(f'data/nfo_processed/{seq}_gt/{k:05d}_handAv.png', 0)
    return v == 1, v != 2


def st(a, h, keep):
    tp = (a & h & keep).sum()
    return np.array([tp / max(((a | h) & keep).sum(), 1), tp / max((a & keep).sum(), 1), tp / max((h & keep).sum(), 1)])


def main():
    F = [(s, k) for s, k, *_ in frames() if s != 'seq1']
    arms = ('T1', 'B0', 'ceiling', 'V̂', 'amodal')
    S = {a: [] for a in arms}
    void, bins, miss = [], {a: [] for a in arms}, []
    for s, k in F:
        d, (h, keep) = load(s, k), lab(s, k)
        void.append((~keep).sum() / max((h | ~keep).sum(), 1))
        M = {'T1': d['T1'], 'B0': d['B0'], 'ceiling': d['ceil'], 'V̂': d['T1'] | add_anchored(d, U),
             'amodal': np.load(f'results/benchmark/rr/amodal_dvas_{s}/{k:05d}.npz')['amodal'][0]}
        for a in arms:
            S[a].append(st(M[a], h, keep))
        n, L, cs, _ = cv2.connectedComponentsWithStats(h.astype(np.uint8), connectivity=8)
        for c in range(1, n):
            cc = L == c
            for a in arms:
                bins[a].append((cs[c, cv2.CC_STAT_AREA] / CELL ** 2, (M[a] & cc).sum() / cc.sum()))
        fn = h & keep & ~d['T1']
        near = cv2.dilate(d['T1'].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        u = (d['pb'] - d['mb']) / max(d['mp'] - d['mb'], 1e-6)
        miss.append([fn.sum() / max((h & keep).sum(), 1), (fn & near).sum() / max((h & keep).sum(), 1),
                     (fn & (u > U)).sum() / max(fn.sum(), 1)])
    print(f'n = {len(F)} frames; void = {np.mean(void) * 100:.1f}% of (person ∪ void) area on average')
    print('J / P / R, void ignored')
    for a in arms:
        print(f'  {a:8s} ' + ' / '.join(f'{v:.3f}' for v in np.mean(S[a], 0)))
    rng = np.random.default_rng(0)
    for a in ('B0', 'ceiling', 'V̂', 'amodal'):
        dj = np.array(S[a])[:, 0] - np.array(S['T1'])[:, 0]
        bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(5000)]
        per = '  '.join(f'{q}: {dj[[i for i, f in enumerate(F) if f[0] == q]].mean():+.3f}' for q in ('seq2', 'seq3', 'seq4'))
        print(f'  ΔJ {a:8s}− T1 {dj.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] '
              f'better {(dj > 0).sum()}/{len(dj)}   {per}')
    m = np.mean(miss, 0)
    print(f'T1 misses {m[0] * 100:.1f}% of non-void person area; {m[1] * 100:.1f}% lies within 3 px of T1; '
          f'{m[2] * 100:.0f}% of the missed area shows change u > {U}')
    print('recall of person components by size (cells): n / ' + ' / '.join(arms))
    for lo, hi in ((0, .5), (.5, 1), (1, 4), (4, 1e9)):
        v = {a: [r for z, r in bins[a] if lo <= z < hi] for a in arms}
        print(f'  [{lo}, {hi if hi < 1e9 else "inf"}): {len(v["T1"])} / ' + ' / '.join(
            f'{np.mean(v[a]):.2f}' if v[a] else 'n/a' for a in arms))


if __name__ == '__main__':
    main()
