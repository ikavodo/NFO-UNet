"""V̂ and baselines on ALL seq1 windows, scored against the automatic silhouette label (2026-10-14).

Label (frozen 2026-10-13 by the pre-registered F2 test in extent_label_test.py, PASS on 8 held-out polygon frames:
F2 0.798 vs SAM2-ceiling 0.706): L = signed-distance blend of the 3 best head-complete clear-frame SAM2-ceiling
silhouettes (retrieved by coverage of the SAM2-ceiling P minus overlap with "background seen through a gap",
4-band sideways alignment), no depth cleaning (θ = none), ∪ P.  Cached in results/benchmark/rr/labels_sil_frozen/.
Caveat: L ⊇ P (the SAM2-ceiling), so the SAM2-ceiling arm has precision 1 by construction; it is listed only as a
reference.
Arms: T1 (SAMURAI), B0 (SAM2.1), V̂_z (frozen robust-z rule, masks/t1-vhat), V̂_anch (u* = 0.85, learned_thresh.py
anchored rule, fixed after LOSO), SAM2-ceiling.
Frames: local index 6 .. W-2 with non-empty P (V̂ needs 6 history frames). Window-level bootstrap (15 windows).
"""
import json
import os

import numpy as np

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos import exemplar_gt as EG, extent_label_test as ELT
from benchmark.nfo_vos.extent_label_test import bank, silhouette
from benchmark.nfo_vos.learned_thresh import add_anchored, anchors, features
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

LDIR = f'{TR.RES}/labels_sil_frozen'
FDIR = f'{TR.RES}/vhat_sil_feats'
U = 0.85
# 2026-10-16: the same frozen rule on seq2–4 (NFO_SEQ=seq2 ...): the exemplar bank, evidence and depth are the
# sequence's own (depth_seq.py); the label modules read SEQ as a module global, so it is set here.
SEQ = ELT.SEQ = EG.SEQ = os.environ.get('NFO_SEQ', 'seq1')
DEPTH = f'{TR.RES}/depth/{SEQ}_bg_disp_1036.npy'


def st(a, h):
    tp = (a & h).sum(); p, r = tp / max(a.sum(), 1), tp / max(h.sum(), 1)
    return np.array([tp / max((a | h).sum(), 1), p, r, 5 * p * r / max(4 * p + r, 1e-9)])


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    os.makedirs(LDIR, exist_ok=True); os.makedirs(FDIR, exist_ok=True)
    d, E, pred = np.load(DEPTH), bank('clear'), None
    W = {a: [] for a in ('T1', 'B0', 'V̂_z', 'V̂_anch', 'ceiling')}
    for t in T:
        n = len(t['frames'])
        js = [j for j in range(6, n - 1) if (P := gt_mask_native(SEQ, t['frames'][j])) is not None and P.any()]
        w0, w1 = t['warmup']
        bg = np.median(np.stack([V.native(k, SEQ) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
        lp = f"{LDIR}/{t['id']}.npz"
        if os.path.exists(lp):
            L = dict(np.load(lp))
        else:
            L = {}
            for j in js:
                S, P = silhouette(t['frames'][j], E, bg, d, align=True)
                L[str(j)] = S | P
            np.savez_compressed(lp, **L)
        fp = f"{FDIR}/{t['id']}.npz"
        M1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks'].astype(bool)
        B0 = np.load(f"{TR.RES}/masks/b0/{t['id']}.npz")['masks'].astype(bool)
        VZ = np.load(f"{TR.RES}/masks/t1-vhat/{t['id']}.npz")['masks'].astype(bool)
        if os.path.exists(fp):
            Fz = np.load(fp); MC, PB = Fz['MC'], Fz['pb'].astype(np.float32)
        else:
            pred = pred or R.image_predictor()
            Tk = R.track(t)
            MC, PB = np.zeros_like(M1), np.zeros(M1.shape, np.float16)
            for j in js:
                f = features(t, j, Tk, bg, pred, M1[j])
                if f is not None:
                    MC[j], PB[j] = f[0], f[1].astype(np.float16)
            np.savez_compressed(fp, MC=MC, pb=PB); PB = PB.astype(np.float32)
        rows = {a: [] for a in W}
        for j in js:
            h = L[str(j)]
            dd = {'T1': M1[j], 'MC': MC[j], 'pb': PB[j], 'cand': MC[j] & ~M1[j]}
            dd['mb'], dd['sb'], dd['mp'] = anchors(MC[j], M1[j], PB[j])
            arms = {'T1': M1[j], 'B0': B0[j], 'V̂_z': VZ[j], 'V̂_anch': M1[j] | add_anchored(dd, U),
                    'ceiling': gt_mask_native(SEQ, t['frames'][j])}
            for a, m in arms.items():
                rows[a].append(st(m, h))
        for a in W:
            W[a].append(np.mean(rows[a], 0))
        print(f"{t['id']}: {len(js)} frames; J T1 {W['T1'][-1][0]:.3f}  V̂_anch {W['V̂_anch'][-1][0]:.3f}", flush=True)
    rng = np.random.default_rng(0)
    print(f'\n{SEQ}, {len(T)} windows, vs frozen silhouette label: mean over windows J / P / R / F2')
    for a in W:
        print(f'  {a:8s} ' + ' / '.join(f'{v:.3f}' for v in np.mean(W[a], 0)))
    for a in ('B0', 'V̂_z', 'V̂_anch'):
        for i, m in ((0, 'J'), (2, 'R'), (3, 'F2')):
            dj = np.array(W[a])[:, i] - np.array(W['T1'])[:, i]
            bs = [dj[rng.integers(0, len(dj), len(dj))].mean() for _ in range(5000)]
            print(f'  Δ{m:2s} {a:8s}− T1 {dj.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] '
                  f'better {(dj > 0).sum()}/{len(dj)}')


if __name__ == '__main__':
    main()
