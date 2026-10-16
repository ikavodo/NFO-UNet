"""S0 on NFO hard starts: inside the person's extent at t0, does "frame t0 differs from the
person-aligned composite of its history" separate visible person pixels from occluder/background? (2026-10-12)

    python -m benchmark.nfo_vos.diag_composite_diff     # local GPU (SAM2 image encoder for the feature arms)
    LAYERED=1 python -m benchmark.nfo_vos.diag_composite_diff   # layered-memory plan, option C step 1

Setup per hard start (B0 or T1 init failure) with a full 7-frame window t0-6..t0:
  history H = t0-6..t0-1 (t0 itself is NOT in the composite), aligned onto t0 by GT box centres
  (seq4_s2_t466: no GT boxes before t0 -> centres of the _sammask_ext masks, flagged 'ext');
  recency weights exp(-age/3.5) (a4.recency_weights), normalised over H;
  per-pixel history visibility Vbar = sum_k w_k * aligned visible mask_k (SAM2-ceiling, or _ext).
Composites of H:
  raw    sum_k w_k A_k                       (the A/4 rule, history only)
  median per-pixel median of the A_k
  clean  sum_k w_k v_k A_k / sum_k w_k v_k   ORACLE cleaning (SAM2-ceiling visibility); falls back to
         raw where no history frame saw the person (Vbar = 0)
Scores inside the extent E = GT (amodal) box at t0; label = SAM2-ceiling visible mask at t0:
  pix_<c>  -|R_t0 - C| (3x3 mean)      feat<s>_<c>  cosine(SAM2 features of R_t0, of C) at stride s
  bgdiff   |R_t0 - warm-up background| (the cue that failed on 2026-10-08), as the control
Reported: pixel AUC per start (mean over starts) and pooled by Vbar bin. Derived from the mixing
model R = vP + (1-v)O, C = Vbar P + (1-Vbar) Obar: the raw composite should separate only at high
Vbar; the clean one everywhere.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from benchmark.nfo_vos import a4, run_t4 as R, trials as TR, velocity as V
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

OUT_CSV = f'{TR.RES}/diag_composite_diff.csv'
TRACE = f'{TR.IMG}_checks/composite_diff'
BINS = [(0, 0.2), (0.2, 0.5), (0.5, 1.01)]


def mask_native(seq, idx):
    """SAM2-ceiling visible mask, else the _ext mask; (mask, kind) or (None, None). Native files
    (*_native.png, 2026-10-12) are preferred over the upsampled 224 ones."""
    for kind, suf in (('gt', 'sammask_native'), ('gt', 'sammask'), ('ext', 'sammask_ext_native'), ('ext', 'sammask_ext')):
        p = f'data/nfo_processed/{seq}_gt/{idx:05d}_{suf}.png'
        if os.path.exists(p) and suf.endswith('native'):
            return cv2.imread(p, 0) > 127, kind
        if os.path.exists(p):
            up = cv2.resize(cv2.imread(p, 0), (800, 800), interpolation=cv2.INTER_LINEAR) > 127
            return up[100:700, :], kind
    return None, None


def box_centre(seq, k):
    b = TR.bbs(seq)
    if k not in b or not b[k] or b[k][0].x < 0:
        return None
    x0, y0, x1, y1 = gt_to_native(b[k][0], TR.NATIVE_W, TR.NATIVE_H)
    return (x0 + x1) / 2, (y0 + y1) / 2


def shift(img, dx, dy, interp=cv2.INTER_LINEAR):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img.astype(np.float32), M, (img.shape[1], img.shape[0]), flags=interp,
                          borderMode=cv2.BORDER_REPLICATE)


def window(t):
    """Aligned history frames, masks, weights; None if the 7-frame window is incomplete."""
    seq, f = t['seq'], t['t0']
    H = list(range(f - a4.N + 1, f))
    masks = [mask_native(seq, k) for k in H]
    if any(m is None for m, _ in masks):
        return None
    kinds = {kd for _, kd in masks}
    g0 = box_centre(seq, f)
    cs = [box_centre(seq, k) for k in H]
    if any(c is None for c in cs):        # no GT boxes in the history: centres of the (ext) masks' boxes
        def mbox(m):
            ys, xs = np.nonzero(m); return (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        cs = [mbox(m) for m, _ in masks]
        G0, _ = mask_native(seq, f)
        g0 = mbox(G0)
        kinds.add('ext-align')
    A = [shift(V.native(k, seq), g0[0] - c[0], g0[1] - c[1]) for k, c in zip(H, cs)]
    Vm = [shift(m.astype(np.float32), g0[0] - c[0], g0[1] - c[1], cv2.INTER_NEAREST) for (m, _), c in zip(masks, cs)]
    w = np.exp(-np.arange(len(H) - 1, -1, -1) / a4.TAU_AGE); w = w / w.sum()
    return dict(A=np.stack(A), V=np.stack(Vm), w=w, kind='+'.join(sorted(kinds)))


def composites(W):
    A, Vm, w = W['A'], W['V'], W['w'][:, None, None]
    raw = (w * A).sum(0)
    vw = (w * Vm).sum(0)
    clean = np.where(vw > 1e-6, (w * Vm * A).sum(0) / np.maximum(vw, 1e-6), raw)
    return dict(raw=raw, median=np.median(A, 0), clean=clean), vw


def feats(img):
    pred = R.image_predictor()
    pred.set_image(np.stack([np.clip(img, 0, 255).astype(np.uint8)] * 3, -1))
    hr = pred._features['high_res_feats']
    return {4: hr[0][0], 8: hr[1][0], 16: pred._features['image_embed'][0]}


def cosine_map(Fa, Fb, shape):
    c = torch.nn.functional.cosine_similarity(Fa.float(), Fb.float(), dim=0)[None, None]
    # SAM2 resizes the native frame to 1024x1024 (no padding): stretch the map back to the native frame
    return torch.nn.functional.interpolate(c, size=shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def auc(s, y):
    if y.all() or not y.any():
        return np.nan
    return mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum())


def main():
    T = {x['id']: x for x in json.load(open(TR.OUT)) if x['admissible']}
    pt = pd.read_csv(f'{TR.RES}/per_trial.csv')
    hard = sorted(set(pt[pt.method.isin(['b0', 't1']) & pt.init_fail].trial))
    os.makedirs(TRACE, exist_ok=True)
    rows, pooled = [], []
    with torch.inference_mode():
        for tid in hard:
            t = T[tid]; seq, f = t['seq'], t['t0']
            W = window(t)
            if W is None:
                print(f'{tid}: skipped (incomplete 7-frame window)'); continue
            Rt = V.native(f, seq).astype(np.float32)
            G0, _ = mask_native(seq, f)
            C, vbar = composites(W)
            w0, w1 = t['warmup']
            bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
            x0, y0, x1, y1 = (int(round(v)) for v in t['box_native'])
            E = np.zeros(Rt.shape, bool); E[max(0, y0):y1, max(0, x0):x1] = True
            S = {f'pix_{c}': -cv2.blur(np.abs(Rt - Cc), (3, 3)) for c, Cc in C.items()}
            S['bgdiff'] = cv2.blur(np.abs(Rt - bg), (3, 3))
            FR = feats(Rt)
            for c in ('raw', 'clean'):
                FC = feats(C[c])
                for s in (4, 16):
                    S[f'feat{s}_{c}'] = cosine_map(FR[s], FC[s], Rt.shape)
            y = G0[E]
            r = dict(trial=tid, kind=W['kind'], v=t['v'], vis_frac=y.mean(), vbar_mean=vbar[E].mean(),
                     vbar_on_vis=vbar[E & G0].mean())
            r.update({k: auc(s[E], y) for k, s in S.items()})
            rows.append(r)
            pooled.append(pd.DataFrame({'y': y, 'vbar': vbar[E], **{k: s[E] for k, s in S.items()}}))
            print(f"{tid} [{W['kind']}] vis {r['vis_frac']:.2f} Vbar|vis {r['vbar_on_vis']:.2f}  AUC raw {r['pix_raw']:.2f} "
                  f"median {r['pix_median']:.2f} clean {r['pix_clean']:.2f} f4clean {r['feat4_clean']:.2f} "
                  f"f16clean {r['feat16_clean']:.2f} bgdiff {r['bgdiff']:.2f}", flush=True)
            # trace: frame + GT outline | raw C | clean C | -|R-raw| | -|R-clean| | feat4 clean | Vbar
            sl = (slice(max(0, y0 - 30), min(600, y1 + 30)), slice(max(0, x0 - 30), min(800, x1 + 30)))
            def tile(img, name, norm=False):
                a = img[sl].astype(np.float32)
                if norm:
                    a = 255 * (a - a.min()) / max(np.ptp(a), 1e-6)
                im = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                cs, _ = cv2.findContours(G0[sl].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
                im = cv2.resize(im, (150, 240)); cv2.putText(im, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
                return im
            cv2.imwrite(f'{TRACE}/{tid}.png', np.hstack([
                tile(Rt, 'R t0 + GT'), tile(C['raw'], 'raw C(hist)'), tile(C['clean'], 'clean C (oracle)'),
                tile(S['pix_raw'], '-|R-raw|', True), tile(S['pix_clean'], '-|R-clean|', True),
                tile(S['feat4_clean'], 'cos s4 clean', True), tile(vbar * 255, 'Vbar')]))
    d = pd.DataFrame(rows); d.to_csv(OUT_CSV, index=False)
    arms = [c for c in d.columns if c.startswith(('pix_', 'feat', 'bgdiff'))]
    print(f'\nper-start AUC (visible vs not, inside the GT box), mean over {len(d)} starts:')
    print(d[arms].mean().round(3).to_string())
    P = pd.concat(pooled)
    print('\npooled AUC by Vbar bin (pixels, all starts):')
    tab = {f'[{lo},{hi:.1f})': {a: auc(P[a].values[(P.vbar >= lo) & (P.vbar < hi)], P.y.values[(P.vbar >= lo) & (P.vbar < hi)])
                                for a in arms} for lo, hi in BINS}
    tab = pd.DataFrame(tab)
    tab.loc['n_pixels'] = [int(((P.vbar >= lo) & (P.vbar < hi)).sum()) for lo, hi in BINS]
    tab.loc['visible_frac'] = [P.y[(P.vbar >= lo) & (P.vbar < hi)].mean() for lo, hi in BINS]
    print(tab.round(3).to_string())
    print(f'-> {OUT_CSV}, traces {TRACE}/')


if __name__ == '__main__' and not any(os.environ.get(k) for k in ('LAYERED', 'MULTI', 'GALLERY')):
    main()


def static_z(F, Sbg):
    """Per-cell z-score of 'differs from the empty scene': (median - cos) / (1.4826 MAD) of
    cos(F, S_bg) over the frame's 64x64 cells (most cells are background). F: [..., C, H, W]."""
    c = torch.nn.functional.cosine_similarity(F.float(), Sbg.float(), dim=-3)
    flat = c.flatten(-2)
    med = flat.median(-1).values[..., None, None]
    mad = (flat - flat.median(-1).values[..., None]).abs().median(-1).values[..., None, None] * 1.4826
    return (med - c) / mad.clamp(min=1e-6)


def layered_memory_check():
    """Layered-memory plan (docs/layer_sam2/design/layered_memory_test_plan.md), option C, step 1 on NFO hard
    starts, frozen SAM2 stride-16 backbone features of the history t0-6..t0-1, GT-shift alignment:
      D       mean of the history features warped onto t0 (the plan's dynamic layer)
      S       mean of the unwarped history features (the plan's static layer)
      S_bg    features of the person-free warm-up median (NFO's clean static layer)
      D_clean like D, but each history cell weighted by g_k = sigmoid(z_k - 3), z_k = static_z of that
              frame vs S_bg (GT-free 'something new was visible here'), recency weights w_k
      r       -cos(f_t, S_bg) + cos(f_t, D_clean)   (refinement score: new now AND looks like the person)
    AUC visible vs not inside the GT box, precision at recall 0.9, best J (per-start oracle threshold)."""
    T = {x['id']: x for x in json.load(open(TR.OUT)) if x['admissible']}
    pt = pd.read_csv(f'{TR.RES}/per_trial.csv')
    hard = sorted(set(pt[pt.method.isin(['b0', 't1']) & pt.init_fail].trial))
    rows, pooled = [], []
    cell = 800 / 64                                  # native px per stride-16 cell in x (1024 input = stretched 800)
    fn = torch.nn.functional
    with torch.inference_mode():
        for tid in hard:
            t = T[tid]; seq, f = t['seq'], t['t0']
            W = window(t)
            if W is None:
                continue
            H = list(range(f - a4.N + 1, f))
            g0 = box_centre(seq, f); cs = [box_centre(seq, k) for k in H]
            if any(c is None for c in cs) or g0 is None:
                continue                              # needs GT shifts (drops the ext-aligned start)
            Ft = feats(V.native(f, seq).astype(np.float32))[16].float()
            Fh = torch.stack([feats(V.native(k, seq).astype(np.float32))[16] for k in H]).float()   # [N,C,64,64]
            w0, w1 = t['warmup']
            bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
            Sbg = feats(bg)[16].float()
            dx = torch.tensor([(g0[0] - c[0]) / cell for c in cs], device=Fh.device)
            dy = torch.tensor([(g0[1] - c[1]) / (600 / 64) for c in cs], device=Fh.device)
            N, C, Hh, Ww = Fh.shape
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, Hh, device=Fh.device), torch.linspace(-1, 1, Ww, device=Fh.device), indexing='ij')
            grid = torch.stack([xs[None] - (2 * dx / (Ww - 1))[:, None, None], ys[None] - (2 * dy / (Hh - 1))[:, None, None]], -1).float()
            warp = lambda X: fn.grid_sample(X, grid, align_corners=True)
            valid = warp(torch.ones_like(Fh[:, :1]))
            D = (warp(Fh) * valid).sum(0) / valid.sum(0).clamp(min=1e-6)
            S = Fh.mean(0)
            w = torch.tensor(W['w'], device=Fh.device, dtype=torch.float32)[:, None, None, None]
            g = torch.sigmoid(static_z(Fh, Sbg) - 3)[:, None]                 # [N,1,64,64]
            den = (warp(w * g) * valid).sum(0)
            Dc = (warp(w * g * Fh) * valid).sum(0) / den.clamp(min=1e-6)
            shape = (600, 800)
            cD, cS, cB, cDc = (cosine_map(Ft, X, shape) for X in (D, S, Sbg, Dc))
            cov = fn.interpolate(den[None], size=shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy() > 0.05
            G0, _ = mask_native(seq, f)
            x0, y0, x1, y1 = (int(round(v)) for v in t['box_native'])
            E = np.zeros(shape, bool); E[max(0, y0):y1, max(0, x0):x1] = True
            sc = {'cosD': cD, 'D-S': cD - cS, 'D-Sbg': cD - cB, '-cosSbg': -cB,
                  'cosDclean': np.where(cov, cDc, -1.0), 'r': -cB + np.where(cov, cDc, 0.0),
                  'bgdiff_pix': cv2.blur(np.abs(V.native(f, seq).astype(np.float32) - bg), (3, 3))}
            y = G0[E]
            r = dict(trial=tid, motion_cells=float(abs(dx).max()), Dclean_cov=float(cov[E].mean()),
                     **{k: auc(v[E], y) for k, v in sc.items()})
            for k in ('-cosSbg', 'r', 'cosDclean', 'bgdiff_pix'):
                sv = sc[k][E]; o = np.argsort(-sv); yy = y[o]; tp = np.cumsum(yy); n = np.arange(1, len(yy) + 1)
                rec, prec = tp / max(y.sum(), 1), tp / n
                r[f'P@R.9_{k}'] = float(prec[np.searchsorted(rec, 0.9)]) if rec[-1] >= 0.9 else np.nan
                r[f'Jbest_{k}'] = float((tp / (n + y.sum() - tp)).max())
            rows.append(r); pooled.append(pd.DataFrame({'y': y, **{k: v[E] for k, v in sc.items()}}))
            print(f"{tid}: motion {r['motion_cells']:.1f} cells, D_clean coverage {r['Dclean_cov']:.2f}  AUC -cosSbg {r['-cosSbg']:.2f}  "
                  f"cosDclean {r['cosDclean']:.2f}  r {r['r']:.2f}  cosD {r['cosD']:.2f}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{TR.RES}/diag_layered_memory_c.csv', index=False)
    P = pd.concat(pooled)
    print(f'\nmean over {len(d)} starts:'); print(d.drop(columns='trial').mean().round(3).to_string())
    print('pooled AUC:', {k: round(auc(P[k].values, P.y.values), 3) for k in ['cosD', 'D-S', 'D-Sbg', '-cosSbg', 'cosDclean', 'r', 'bgdiff_pix']})


def multilevel_check():
    """Background cue at each SAM2 decoder level (2026-10-12): z of cos(f_t, S_bg) at strides 4, 8 (the
    encoder's high-res features the mask decoder adds during upsampling) and 16 (image embedding), plus the
    pixel background difference. All 17 hard starts (needs only frame t0 and the warm-up background).
    Inside the GT box: AUC visible vs not, precision at recall 0.9, best J (per-start oracle threshold)."""
    T = {x['id']: x for x in json.load(open(TR.OUT)) if x['admissible']}
    pt = pd.read_csv(f'{TR.RES}/per_trial.csv')
    hard = sorted(set(pt[pt.method.isin(['b0', 't1']) & pt.init_fail].trial))
    rows = []
    with torch.inference_mode():
        for tid in hard:
            t = T[tid]; seq, f = t['seq'], t['t0']
            w0, w1 = t['warmup']
            bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
            Rt = V.native(f, seq).astype(np.float32)
            FR, FB = feats(Rt), feats(bg)
            sc = {}
            for s in (4, 8, 16):
                z = static_z(FR[s], FB[s])[None, None]
                sc[f'z{s}'] = torch.nn.functional.interpolate(z, size=(600, 800), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            sc['z4+z16'] = sc['z4'] + sc['z16']
            sc['pix'] = cv2.blur(np.abs(Rt - bg), (3, 3))
            G0, _ = mask_native(seq, f)
            x0, y0, x1, y1 = (int(round(v)) for v in t['box_native'])
            E = np.zeros((600, 800), bool); E[max(0, y0):y1, max(0, x0):x1] = True
            y = G0[E]; r = dict(trial=tid)
            for k, v in sc.items():
                sv = v[E]; r[f'AUC_{k}'] = auc(sv, y)
                o = np.argsort(-sv); yy = y[o]; tp = np.cumsum(yy); n = np.arange(1, len(yy) + 1); rec = tp / max(y.sum(), 1)
                r[f'P@R.9_{k}'] = float((tp / n)[np.searchsorted(rec, 0.9)]) if rec[-1] >= 0.9 else np.nan
                r[f'Jbest_{k}'] = float((tp / (n + y.sum() - tp)).max())
            rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(f'{TR.RES}/diag_multilevel_bg.csv', index=False)
    m = d.drop(columns='trial').mean()
    print(pd.DataFrame({k: [m[f'AUC_{k}'], m[f'P@R.9_{k}'], m[f'Jbest_{k}']] for k in ['z4', 'z8', 'z16', 'z4+z16', 'pix']},
                       index=['AUC', 'P@R.9', 'Jbest']).round(3).to_string())
    print(f'({len(d)} hard starts)')


def gallery():
    """Composite gallery for eyeballing 'good' composites (2026-10-12): every admissible window whose
    7-frame history at t0 has masks (GT, else _ext). Tiles: R_t0 + GT outline | raw recency composite of
    the history | GT-FREE cleaned (each view's pixel weighted by sigmoid(z_pix - 3), z_pix = per-frame
    robust z of |view - warm-up background|, computed before alignment) | ORACLE-cleaned (SAM2-ceiling
    visibility weights). Alignment = GT box centres (oracle). Also a contact sheet of the GT-free column."""
    T = [x for x in json.load(open(TR.OUT)) if x['admissible']]
    out = f'{TR.IMG}_checks/composite_gallery'; os.makedirs(out, exist_ok=True)
    sheet = []
    for t in T:
        W = window(t)
        if W is None:
            continue
        seq, f = t['seq'], t['t0']
        H = list(range(f - a4.N + 1, f))
        g0 = box_centre(seq, f); cs = [box_centre(seq, k) for k in H]
        if g0 is None or any(c is None for c in cs):
            continue
        w0, w1 = t['warmup']
        bg = np.median(np.stack([V.native(k, seq) for k in range(w0, w1 + 1)]), 0).astype(np.float32)
        G = []
        for k, c in zip(H, cs):
            d = np.abs(V.native(k, seq).astype(np.float32) - bg)
            med = np.median(d); mad = 1.4826 * np.median(np.abs(d - med))
            zk = (d - med) / max(mad, 1e-6)
            G.append(shift(1 / (1 + np.exp(-(zk - 3))), g0[0] - c[0], g0[1] - c[1]))
        G = np.stack(G); w = W['w'][:, None, None]
        zc = (w * G * W['A']).sum(0) / np.maximum((w * G).sum(0), 1e-6)
        C, _ = composites(W)
        G0, _ = mask_native(seq, f)
        x0, y0, x1, y1 = (int(round(v)) for v in t['box_native'])
        sl = (slice(max(0, y0 - 30), min(600, y1 + 30)), slice(max(0, x0 - 30), min(800, x1 + 30)))
        def tile(img, name, outline=True):
            im = cv2.cvtColor(np.clip(img[sl], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if outline:
                cnt, _ = cv2.findContours(G0[sl].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnt, -1, (0, 255, 0), 1)
            im = cv2.resize(im, (150, 240)); cv2.putText(im, name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
            return im
        tiles = [tile(V.native(f, seq).astype(np.float32), 'R t0 + GT'), tile(C['raw'], 'raw C', False),
                 tile(zc, 'GT-free clean C', False), tile(C['clean'], 'oracle clean C', False)]
        cv2.imwrite(f"{out}/{t['id']}.png", np.hstack(tiles))
        s_ = tiles[2].copy(); cv2.putText(s_, t['id'], (3, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        sheet.append(s_)
    rows = [np.hstack(sheet[i:i + 10] + [np.zeros_like(sheet[0])] * (10 - len(sheet[i:i + 10]))) for i in range(0, len(sheet), 10)]
    cv2.imwrite(f'{out}/_contact_sheet_gtfree.png', np.vstack(rows))
    print(f'{len(sheet)} windows -> {out}/')


if __name__ == '__main__' and os.environ.get('GALLERY'):
    gallery()

if __name__ == '__main__' and os.environ.get('MULTI'):
    multilevel_check()

if __name__ == '__main__' and os.environ.get('LAYERED'):
    layered_memory_check()
