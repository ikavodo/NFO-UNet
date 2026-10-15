"""Can missed visible person pixels be ADDED to T1's mask from frame-t evidence compared against two
references, the empty scene (static background layer) and a person-aligned cleaned composite of t-6..t-1
(moving person layer)? Layered framing: Wang & Adelson, "Representing moving images with layers", IEEE TIP
1994. Pre-registered in docs/vos_benchmark/nfo_vos_pilot_findings.md (2026-10-12).

    python -m benchmark.nfo_vos.diag_fine_add run       # GPU: ring scores per frame -> results/.../fine_add/<trial>.npz
    python -m benchmark.nfo_vos.diag_fine_add report    # CPU: precision at 50% reach, AUC, tau on seq1, dJ on seq2-4
    python -m benchmark.nfo_vos.diag_fine_add gallery   # visual trace of ring candidates and additions

Per window, frames t0+6, +9, ..., +36; candidates = 10 px ring around T1's mask inside the GT box at t
(dilated 10 px); label = SAM2-ceiling visible. Alignment = GT box centres (oracle).
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

from benchmark.nfo_vos import a4, trials as TR, velocity as V
from benchmark.nfo_vos.diag_composite_diff import box_centre, feats, mask_native, shift, static_z

OUT = f'{TR.RES}/fine_add'
RELS = list(range(6, 37, 3))
RING, DX, DY, DF = 10, 6, 3, 3          # ring px; pixel search +-DX, +-DY; feature search +-DF stride-4 cells


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def centre(seq, k):
    c = box_centre(seq, k)
    if c is None:
        m, _ = mask_native(seq, k)
        if m is None or not m.any():
            return None
        ys, xs = np.nonzero(m); c = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    return c


def zpix(img, bg):
    d = np.abs(img - bg); med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med))
    return (d - med) / max(mad, 1e-6)


def composites(seq, f, bg):
    """raw, GT-free cleaned, oracle-cleaned composites of f-6..f-1 aligned onto f; None if incomplete."""
    H = list(range(f - a4.N + 1, f))
    c0, cs = centre(seq, f), [centre(seq, k) for k in H]
    ms = [mask_native(seq, k)[0] for k in H]
    if c0 is None or any(c is None for c in cs) or any(m is None for m in ms):
        return None
    w = np.exp(-np.arange(len(H) - 1, -1, -1) / a4.TAU_AGE); w = (w / w.sum())[:, None, None]
    A, Vm, G = [], [], []
    for k, c, m in zip(H, cs, ms):
        dx, dy = c0[0] - c[0], c0[1] - c[1]
        img = V.native(k, seq).astype(np.float32)
        A.append(shift(img, dx, dy)); Vm.append(shift(m.astype(np.float32), dx, dy, cv2.INTER_NEAREST))
        G.append(shift(1 / (1 + np.exp(-(zpix(img, bg) - 3))), dx, dy))
    A, Vm, G = np.stack(A), np.stack(Vm), np.stack(G)
    raw = (w * A).sum(0)
    clean = lambda W_: np.where((w * W_).sum(0) > 1e-6, (w * W_ * A).sum(0) / np.maximum((w * W_).sum(0), 1e-6), raw)
    return {'raw': raw, 'gtfree': clean(G), 'oracle': clean(Vm)}


def min_ssd(R, C, sl):
    """min over shifts of the 5x5 SSD between R and shifted C, inside crop sl."""
    best = np.full(R[sl].shape, np.inf, np.float32)
    for dy in range(-DY, DY + 1):
        for dx in range(-DX, DX + 1):
            Cs = np.roll(C, (dy, dx), (0, 1))
            best = np.minimum(best, cv2.boxFilter((R[sl] - Cs[sl]) ** 2, -1, (5, 5)))
    return best


def corr_max(Ft, Fc):
    """max over a (2DF+1)^2 window of cos(Ft(x), Fc(x+d)); [C,H,W] -> [H,W]."""
    Ft = torch.nn.functional.normalize(Ft.float(), dim=0); Fc = torch.nn.functional.normalize(Fc.float(), dim=0)
    best = torch.full(Ft.shape[1:], -1.0, device=Ft.device)
    for dy in range(-DF, DF + 1):
        for dx in range(-DF, DF + 1):
            best = torch.maximum(best, (Ft * torch.roll(Fc, (dy, dx), (1, 2))).sum(0))
    return best


def up(x):
    return torch.nn.functional.interpolate(x[None, None].float(), size=(600, 800), mode='bilinear',
                                           align_corners=False)[0, 0].cpu().numpy()


def run():
    os.makedirs(OUT, exist_ok=True)
    T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    with torch.inference_mode():
        for i, t in enumerate(T):
            path = f"{OUT}/{t['id']}.npz"
            if os.path.exists(path):
                continue
            seq = t['seq']
            w0, w1 = t['warmup']
            bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
            Fbg = feats(bg)
            M1 = np.load(f"{TR.RES}/masks/t1/{t['id']}.npz")['masks']
            Z16 = np.load(f"{TR.RES}/static_z/{t['id']}.npz")['z']
            rec = {}
            for rel in RELS:
                f = t['frames'][rel]
                C = composites(seq, f, bg)
                G, _ = mask_native(seq, f)
                b = TR.bbs(seq).get(f)
                if C is None or G is None or not b or b[0].x < 0:
                    continue
                from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
                x0, y0, x1, y1 = (int(round(v)) for v in gt_to_native(b[0], TR.NATIVE_W, TR.NATIVE_H))
                E = np.zeros((600, 800), bool); E[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = True
                M = M1[rel]
                ring = (cv2.dilate(M.astype(np.uint8), disk(RING)) > 0) & ~M & E
                if not ring.any():
                    continue
                R = V.native(f, seq).astype(np.float32)
                sl = (slice(max(0, y0 - 20), min(600, y1 + 20)), slice(max(0, x0 - 20), min(800, x1 + 20)))
                ssd_bg = cv2.boxFilter((R[sl] - bg[sl]) ** 2, -1, (5, 5))
                S = {'z16': cv2.resize(Z16[rel].astype(np.float32), (800, 600))[sl],
                     'pix_bg': np.log(ssd_bg + 1)}
                FR = feats(R)
                S['z4'] = up(static_z(FR[4], Fbg[4]))[sl]
                cbg = up(torch.nn.functional.cosine_similarity(FR[4].float(), Fbg[4].float(), dim=0))[sl]
                for c, Cc in C.items():
                    S[f'layer_pix_{c}'] = np.log(ssd_bg + 1) - np.log(min_ssd(R, Cc, sl) + 1)
                    cr = up(corr_max(FR[4], feats(Cc)[4]))[sl]
                    S[f'corr4_{c}'] = cr
                    S[f'layer_f4_{c}'] = np.log(1 - cbg + 1e-3) - np.log(1 - cr + 1e-3)
                rg = ring[sl]
                rec[rel] = dict(y=G[sl][rg], I=int((M & G).sum()), nM=int(M.sum()), nG=int(G.sum()),
                                **{k: v[rg].astype(np.float16) for k, v in S.items()})
            np.savez_compressed(path, rels=np.array(sorted(rec)),
                                **{f'{r}__{k}': v for r, d in rec.items() for k, v in d.items()})
            print(f"{i + 1}/{len(T)} {t['id']} frames {len(rec)}", flush=True)


ARMS = ['z16', 'z4', 'pix_bg', 'corr4_raw', 'corr4_gtfree', 'corr4_oracle', 'layer_f4_raw', 'layer_f4_gtfree',
        'layer_f4_oracle', 'layer_pix_raw', 'layer_pix_gtfree', 'layer_pix_oracle']
GTFREE = [a for a in ARMS if not a.endswith('oracle')]


def load():
    T = {t['id']: t for t in json.load(open(TR.OUT)) if t['admissible']}
    frames = []
    for tid, t in T.items():
        z = np.load(f'{OUT}/{tid}.npz')
        for r in z['rels']:
            d = {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(f'{r}__')}
            frames.append(dict(trial=tid, seq=t['seq'], grp=f"{t['seq']}_{t['dir']}_{t['gait']}", **d))
    return frames


def prec_at_reach(s, y, reach=0.5):
    o = np.argsort(-s); tp = np.cumsum(y[o]); n = np.arange(1, len(y) + 1)
    k = np.searchsorted(tp / max(y.sum(), 1), reach)
    return float(tp[k] / n[k]) if k < len(y) else np.nan


def report():
    from scipy.stats import mannwhitneyu
    from benchmark.nfo_vos.analyze import cluster_bootstrap_ci
    pt = pd.read_csv(f'{TR.RES}/per_trial.csv')
    hard = set(pt[pt.method.isin(['b0', 't1']) & pt.init_fail].trial)
    F = load()
    y = np.concatenate([f['y'] for f in F])
    print(f'{len(F)} frames, {len(y)} ring pixels, positives (missed visible) {y.mean():.3f}')
    rows = []
    for a in ARMS:
        s = np.concatenate([f[a].astype(np.float32) for f in F])
        auc = mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())
        rows.append(dict(arm=a, AUC=auc, P_at_R50=prec_at_reach(s, y)))
    print(pd.DataFrame(rows).set_index('arm').round(3).to_string())

    def Jf(f, a, tau):
        add = f[a].astype(np.float32) > tau
        tp, fp = int((add & f['y']).sum()), int((add & ~f['y']).sum())
        return (f['I'] + tp) / (f['nM'] + f['nG'] - f['I'] + fp)
    J0 = lambda f: f['I'] / (f['nM'] + f['nG'] - f['I'])
    print('\ntau tuned on seq1 (mean J over frames), dJ on seq2-4 frames (per-window mean, cluster bootstrap):')
    for a in ARMS:
        s1 = [f for f in F if f['seq'] == 'seq1']
        cand = np.quantile(np.concatenate([f[a].astype(np.float32) for f in s1]), [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999])
        tau = max(cand, key=lambda tt: np.mean([Jf(f, a, tt) for f in s1]))
        d = pd.DataFrame([dict(trial=f['trial'], seq=f['seq'], grp=f['grp'], d=Jf(f, a, tau) - J0(f)) for f in F if f['seq'] != 'seq1'])
        w = d.groupby(['trial', 'seq', 'grp'], as_index=False).d.mean()
        ci = cluster_bootstrap_ci(w, 'd', 'grp', strata='seq')
        hh = w[w.trial.isin(hard)].d.mean(); ee = w[~w.trial.isin(hard)].d.mean()
        print(f'{a:18s} tau {tau:+.3f}  seq1 dJ {np.mean([Jf(f, a, tau) - J0(f) for f in s1]):+.4f}  '
              f'seq2-4 dJ {w.d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  hard {hh:+.4f} easy {ee:+.4f}')




def gallery(ids=('seq2_s1_t315', 'seq4_s1_t313', 'seq3_s5_t1283', 'seq1_s4_t1194', 'seq2_s3_t1103', 'seq4_s7_t1428'), rel=18):
    """Visual trace of the ring test (2026-10-12), one row per window at frame t0+rel. Tiles:
    frame + GT (green) + T1 edge (magenta) | ring candidates: yellow = missed visible (GT, not T1),
    red = ring not GT | additions at 50% reach by layer_pix_gtfree: green = correct, red = wrong |
    GT-free cleaned composite | Vbar = aligned SAM2-CEILING history visibility (oracle input, not a cue)."""
    out = f'{TR.IMG}_checks/fine_add'; os.makedirs(out, exist_ok=True)
    T = {t['id']: t for t in json.load(open(TR.OUT)) if t['admissible']}
    rows = []
    for tid in ids:
        t = T[tid]; seq = t['seq']; f = t['frames'][rel]
        w0, w1 = t['warmup']
        bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
        C = composites(seq, f, bg)
        G, _ = mask_native(seq, f)
        M = np.load(f"{TR.RES}/masks/t1/{tid}.npz")['masks'][rel]
        from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
        x0, y0, x1, y1 = (int(round(v)) for v in gt_to_native(TR.bbs(seq)[f][0], TR.NATIVE_W, TR.NATIVE_H))
        E = np.zeros((600, 800), bool); E[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = True
        ring = (cv2.dilate(M.astype(np.uint8), disk(RING)) > 0) & ~M & E
        R = V.native(f, seq).astype(np.float32)
        s = np.full((600, 800), -np.inf, np.float32)
        full = (slice(0, 600), slice(0, 800))
        s[ring] = (np.log(cv2.boxFilter((R - bg) ** 2, -1, (5, 5)) + 1) - np.log(min_ssd(R, C['gtfree'], full) + 1))[ring]
        yr = G[ring]; sv = s[ring]; o = np.argsort(-sv); tp = np.cumsum(yr[o])
        k = min(np.searchsorted(tp / max(yr.sum(), 1), 0.5), len(o) - 1); thr = sv[o][k]
        add = ring & (s >= thr)
        # Vbar from the aligned SAM2-ceiling history (as composites() oracle weights)
        H = list(range(f - a4.N + 1, f)); c0 = centre(seq, f)
        w = np.exp(-np.arange(len(H) - 1, -1, -1) / a4.TAU_AGE); w /= w.sum()
        vbar = sum(wi * shift(mask_native(seq, h)[0].astype(np.float32), c0[0] - centre(seq, h)[0], c0[1] - centre(seq, h)[1], cv2.INTER_NEAREST)
                   for wi, h in zip(w, H))
        sl = (slice(max(0, y0 - 30), min(600, y1 + 30)), slice(max(0, x0 - 30), min(800, x1 + 30)))
        def base(img):
            return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        def outline(im, m, col):
            cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cnt, -1, col, 1)
        a = base(R); outline(a, G, (0, 255, 0)); outline(a, M, (255, 0, 255))
        b = base(R); b[ring & G] = (0, 255, 255); b[ring & ~G] = (0, 0, 200)
        c = base(R); c[add & G] = (0, 255, 0); c[add & ~G] = (0, 0, 255); outline(c, M, (255, 0, 255))
        d = base(C['gtfree']); e = base(vbar * 255); outline(e, G, (0, 255, 0))
        tiles = []
        for im, name in ((a, 'GT green, T1 magenta'), (b, 'ring: yellow missed, red not GT'),
                         (c, f'adds @50% reach: prec {(add & G).sum() / max(add.sum(), 1):.2f}'),
                         (d, 'GT-free clean C'), (e, 'Vbar = SAM2-CEILING history')):
            im = cv2.resize(im[sl], (190, 300), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
            tiles.append(im)
        row = np.hstack(tiles); cv2.putText(row, f'{tid} t0+{rel}', (3, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        rows.append(row)
    cv2.imwrite(f'{out}/ring_candidates.png', np.vstack(rows))
    print(f'-> {out}/ring_candidates.png')


if __name__ == '__main__':
    {'run': run, 'report': report, 'gallery': gallery}[sys.argv[1]]()
