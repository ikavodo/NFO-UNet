"""S0 on NFO hard starts: inside the person's extent at t0, does "frame t0 differs from the
person-aligned composite of its history" separate visible person pixels from occluder/background? (2026-10-12)

    python -m benchmark.nfo_vos.diag_composite_diff     # local GPU (SAM2 image encoder for the feature arms)

Setup per hard start (B0 or T1 init failure) with a full 7-frame window t0-6..t0:
  history H = t0-6..t0-1 (t0 itself is NOT in the composite), aligned onto t0 by GT box centres
  (seq4_s2_t466: no GT boxes before t0 -> centres of the _sammask_ext masks, flagged 'ext');
  recency weights exp(-age/3.5) (a4.recency_weights), normalised over H;
  per-pixel history visibility Vbar = sum_k w_k * aligned visible mask_k (pseudo-GT, or _ext).
Composites of H:
  raw    sum_k w_k A_k                       (the A/4 rule, history only)
  median per-pixel median of the A_k
  clean  sum_k w_k v_k A_k / sum_k w_k v_k   ORACLE cleaning (pseudo-GT visibility); falls back to
         raw where no history frame saw the person (Vbar = 0)
Scores inside the extent E = GT (amodal) box at t0; label = pseudo-GT visible mask at t0:
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
    """Pseudo-GT visible mask, else the _ext mask; (mask, kind) or (None, None)."""
    for kind, suf in (('gt', 'sammask'), ('ext', 'sammask_ext')):
        p = f'data/nfo_processed/{seq}_gt/{idx:05d}_{suf}.png'
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
    return {4: hr[0][0], 16: pred._features['image_embed'][0]}


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


if __name__ == '__main__':
    main()
