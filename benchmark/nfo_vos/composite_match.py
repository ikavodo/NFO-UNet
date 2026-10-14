"""Gate for the λ₂ term of S2 (2026-10-14, fixed before running): inside the composite envelope M_C, does frame t
MATCH the composite where the person is visible, and not where an occluder hides it?

Pixels: inside M_C (cached by composite_envelope.py), 18 seq1 seg-4 brush frames; label = brush person (void
excluded). Scores (upsampled to native):
  z16          frame vs empty scene, stride 16 (reference: what the static model already gives)
  match_s      max over ±1 cell of cos(f_t^s(x), f_C^s(x + d)), s ∈ {16, 8, 4}, C = GT-free composite (oracle aligned)
  far          empty-scene disparity ≤ 0.2 (the depth filter of test 2b)
  z16+match16  mean of within-frame ranks of z16 and match16 (untuned combination)
PASS (λ₂ worth building): AUC(match16) ≥ 0.70 AND AUC(z16+match16) ≥ AUC(z16) + 0.02.
"""
import json
import os

import cv2
import numpy as np
import torch
from scipy.stats import rankdata

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, OUT, SEQ, auc, window_for
from benchmark.nfo_vos.diag_composite_diff import feats, static_z
from benchmark.nfo_vos.diag_fine_add import composites, up

FAR = np.load('results/benchmark/rr/depth/seq1_bg_disp_1036.npy') <= 0.2


def match(Ft, Fc):
    Ft = torch.nn.functional.normalize(Ft.float(), dim=0); Fc = torch.nn.functional.normalize(Fc.float(), dim=0)
    best = torch.full(Ft.shape[1:], -1.0, device=Ft.device)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            best = torch.maximum(best, (Ft * torch.roll(Fc, (dy, dx), (1, 2))).sum(0))
    return best


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]
    ks = sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith('_handmask_raw.png'))
    bgs, S = {}, {}
    with torch.inference_mode():
        for k in ks:
            p = f'{OUT}/MC_{k:05d}.npy'
            if not os.path.exists(p):
                continue
            MC = np.load(p)
            t = window_for(k, T)
            if tuple(t['warmup']) not in bgs:
                w0, w1 = t['warmup']
                bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
                bgs[tuple(t['warmup'])] = (bg, feats(bg))
            bg, Fbg = bgs[tuple(t['warmup'])]
            C = composites(SEQ, k, bg)['gtfree']
            FR, FC = feats(V.native(k, SEQ).astype(np.float32)), feats(C)
            Bm = cv2.imread(f'{GT}/{k:05d}_handmask_raw.png', 0)
            sel = MC & (Bm != 2)
            sc = {'z16': up(static_z(FR[16], Fbg[16])), 'far': FAR.astype(np.float32),
                  **{f'match{s}': up(match(FR[s], FC[s])) for s in (16, 8, 4)}}
            r = {a: v[sel] for a, v in sc.items()}
            r['z16+match16'] = (rankdata(r['z16']) + rankdata(r['match16'])) / 2
            S[k] = dict(y=(Bm == 1)[sel], **r)
    y = np.concatenate([d['y'] for d in S.values()])
    print(f'{len(S)} frames, {len(y)} pixels inside M_C, {y.mean():.2f} visible person (brush)')
    res = {}
    for a in ('z16', 'match16', 'match8', 'match4', 'far', 'z16+match16'):
        res[a] = auc(np.concatenate([d[a] for d in S.values()]).astype(np.float64), y)
        per = np.nanmean([auc(d[a].astype(np.float64), d['y']) for d in S.values()])
        print(f'  {a:12s} pooled AUC {res[a]:.3f}   mean per-frame AUC {per:.3f}')
    ok = res['match16'] >= 0.70 and res['z16+match16'] >= res['z16'] + 0.02
    # fairer combination: logistic regression on standardised [z16, match16(, far)], leave-one-frame-out
    from scipy.optimize import minimize
    def fit(X, yy):
        f = lambda w: np.logaddexp(0, -(2 * yy - 1) * (X @ w[1:] + w[0])).mean() + 1e-4 * (w[1:] ** 2).sum()
        return minimize(f, np.zeros(X.shape[1] + 1), method='L-BFGS-B').x
    for feats_ in (['z16'], ['z16', 'match16'], ['z16', 'far'], ['z16', 'far', 'match16']):
        sc_, ys_ = [], []
        for k in S:
            tr = [j for j in S if j != k]
            Xtr = np.vstack([np.column_stack([S[j][a] for a in feats_]) for j in tr]).astype(np.float64)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            w = fit((Xtr - mu) / sd, np.concatenate([S[j]['y'] for j in tr]).astype(np.float64))
            Xk = (np.column_stack([S[k][a] for a in feats_]).astype(np.float64) - mu) / sd
            sc_.append(Xk @ w[1:] + w[0]); ys_.append(S[k]['y'])
        print(f'  LOFO logistic {"+".join(feats_):22s} pooled AUC {auc(np.concatenate(sc_), np.concatenate(ys_)):.3f}')
    print(f'gate: match16 {res["match16"]:.3f} (≥ 0.70), z16+match16 − z16 = {res["z16+match16"] - res["z16"]:+.3f} '
          f'(≥ +0.02) -> {"PASS" if ok else "FAIL"}')


if __name__ == '__main__':
    main()
