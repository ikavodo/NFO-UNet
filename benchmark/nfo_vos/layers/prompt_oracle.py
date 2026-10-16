"""O1 of docs/layer_sam2/training/plan_fusion_prompt_adapter.md (2026-10-16, fixed before running): can SAM2's dense
mask-prompt channel carry the information the adapter would learn? No training.

Clips: 30 fixed val clips (dataset.clip_specs('val')[::4]), flutter nuisance (noise.FLUTTER_AMP), noisy frames X;
frames dataset.valid_targets (t ≥ 6, stride 3). Frozen SAM2.1 b+ image predictor, memory-free, per frame.
Sparse prompt (all arms): tracker-like point = centre of the jittered GT box (σ_a = 3 px; can land on foliage, like
the blob tracker's merged centre). Composite: factorised_gtfree.composite_frames of X[t-6..t-1] aligned by the
jittered-box similarity (dataset.sim), bg = warm-up median.  C̃ = a·C + (1 − a)·bg, a = clip(Σ w G) (plan §2).
Arms (dense prompt = mask_input, 256×256 logits; g = 1):
  pt        point only (the control)
  box       point + jittered box
  hand_C    logit(σ(l_C)·(1 − σ(l_B))), l_X = SAM2 low-res logits with the point alone on X ∈ {C, bg}
  hand_Ct   the same on C̃ (does background-consistency remove the leak?)
  gt_S      ±10 logits of S_t (extent)          gt_SO  S_t ∧ O_t (observable extent)       gt_V  V_t (modal bound)
J / P / R vs V_t, mean per clip; Δ vs pt, clip bootstrap; also on the bad-init stratum (frames with J(pt) < 0.5).
PRE-REGISTERED KILL: max(ΔJ gt_S, ΔJ gt_SO) ≤ +0.02 → the dense-prompt channel cannot carry the fix; stop the plan.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.prompt_oracle
"""
import json
import os

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import a4, run_t4 as R
from benchmark.nfo_vos.diag_fine_add import zpix
from benchmark.nfo_vos.layers import dataset as D
from benchmark.nfo_vos.synth import noise

OUT = 'results/benchmark/rr/layers_prompt_oracle'
IMG = 'images/benchmark/gpjatk_foliage/prompt_oracle'
ARMS = ('pt', 'box', 'hand_C', 'hand_Ct', 'gt_S', 'gt_SO', 'gt_V', 'box+hand_Ct', 'box+gt_SO', 'box+gt_V')
# box+* arms added after the 1-clip smoke test (box beat gt_V): the dense prompt on top of the standard hand prompt.
# They do not change the pre-registered kill rule.
SIGMA_A = 3.0


def composite_and_mass(imgs, Ms, bg):
    """factorised_gtfree.composite_frames (factorised_gtfree.py:62-71) plus the gate mass a = Σ w G."""
    w = np.exp(-np.arange(len(imgs) - 1, -1, -1) / a4.TAU_AGE); w = (w / w.sum())[:, None, None]
    wa = lambda x, M: cv2.warpAffine(x.astype(np.float32), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
    A = np.stack([wa(img, M) for img, M in zip(imgs, Ms)])
    G = np.stack([wa(1 / (1 + np.exp(-(zpix(img, bg) - 3))), M) for img, M in zip(imgs, Ms)])
    raw, mass = (w * A).sum(0), (w * G).sum(0)
    C = np.where(mass > 1e-6, (w * G * A).sum(0) / np.maximum(mass, 1e-6), raw)
    return C, np.clip(mass, 0, 1)


def lowres(m):
    """Binary native mask -> ±10 logits on SAM2's 256×256 mask-input grid (1024² resize, no aspect kept)."""
    return 20 * cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) - 10


def run_clip(pred, spec, rng, keep):
    d = D.make_clip(*spec, flutter=noise.FLUTTER_AMP)
    bxj = D.jitter(d['boxes'], SIGMA_A, rng)
    rgb = lambda x: np.stack([np.clip(x, 0, 255).astype(np.uint8)] * 3, -1)
    rows, trace = [], None
    with torch.inference_mode():
        pred.set_image(rgb(d['bg']))
        Fbg = (pred._features, pred._orig_hw)
        for t in D.valid_targets(d['boxes']):
            V, S = d['V'][t], d['S'][t]
            if not V.any():
                continue
            O = D.observable(d['S'], d['V'], d['boxes'], t)
            cx, cy, h = bxj[t]
            pt = np.array([[cx, cy]], np.float32)
            ys, xs = np.nonzero(S)
            w = h * (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)      # GT aspect, jittered centre/height
            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)
            Ms = [D.sim(bxj[i], bxj[t]) for i in range(t - D.N_HIST + 1, t)]
            C, a = composite_and_mass([d['X'][i] for i in range(t - D.N_HIST + 1, t)], Ms, d['bg'])
            Ct = a * C + (1 - a) * d['bg']
            kw = dict(point_coords=pt, point_labels=np.array([1]), multimask_output=False)
            lo = {}
            pred._features, pred._orig_hw = Fbg; pred._is_image_set = True
            lo['B'] = pred.predict(**kw, return_logits=True)[2][0]
            for name, img in (('C', C), ('Ct', Ct)):
                pred.set_image(rgb(img))
                lo[name] = pred.predict(**kw, return_logits=True)[2][0]
            sig = lambda x: 1 / (1 + np.exp(-x))
            hand = lambda l: np.log(np.clip(sig(l) * (1 - sig(lo['B'])), 1e-4, 1 - 1e-4) /
                                    np.clip(1 - sig(l) * (1 - sig(lo['B'])), 1e-4, 1))
            dense = dict(pt=None, box=None, hand_C=hand(lo['C']), hand_Ct=hand(lo['Ct']), gt_S=lowres(S),
                         gt_SO=lowres(S & O), gt_V=lowres(V))
            for k in ('hand_Ct', 'gt_SO', 'gt_V'):
                dense[f'box+{k}'] = dense[k]
            pred.set_image(rgb(d['X'][t]))
            r, out = {}, {}
            for arm in ARMS:
                m = pred.predict(**kw, box=box if arm.startswith('box') else None,
                                 mask_input=None if dense[arm] is None else dense[arm][None])[0][0] > 0
                tp = (m & V).sum()
                r[arm] = (tp / max((m | V).sum(), 1), tp / max(m.sum(), 1), tp / max(V.sum(), 1))
                out[arm] = m
            rows.append(r)
            if keep and trace is None and r['pt'][0] < 0.5:
                trace = dict(X=d['X'][t], C=C, Ct=Ct, V=V, hand=sig(dense['hand_Ct']), **out)
    return d['cid'], rows, trace


def montage(traces, path):
    g = lambda x: cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rows = []
    for tr in traces:
        def ov(m):
            v = g(tr['X']); v[m & tr['V']] = (0, 200, 0); v[m & ~tr['V']] = (0, 0, 255); v[~m & tr['V']] = (255, 0, 0)
            return v
        hand = cv2.applyColorMap(cv2.resize((tr['hand'] * 255).astype(np.uint8), tr['X'].shape[::-1]), cv2.COLORMAP_JET)
        panes = [g(tr['X']), g(tr['C']), g(tr['Ct']), hand, ov(tr['pt']), ov(tr['hand_Ct']), ov(tr['gt_SO'])]
        for v, txt in zip(panes, ('X_t', 'C', 'C~', 'hand prompt (C~)', 'pt', 'hand_Ct', 'gt_SO')):
            cv2.putText(v, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        rows.append(cv2.resize(np.hstack(panes), None, fx=0.18, fy=0.18))
    cv2.imwrite(path, np.vstack(rows))


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(IMG), exist_ok=True)
    pred, rng = R.image_predictor(), np.random.default_rng(0)
    specs = D.clip_specs('val')[::4][:int(os.environ.get('O1_LIMIT', 30))]
    per_clip, bad, traces = {a: [] for a in ARMS}, {a: [] for a in ARMS}, []
    for n, spec in enumerate(specs):
        cid, rows, tr = run_clip(pred, spec, rng, keep=len(traces) < 6)
        if tr is not None:
            traces.append(tr)
        for a in ARMS:
            per_clip[a].append(np.mean([r[a] for r in rows], 0))
            bad[a] += [r[a] for r in rows if r['pt'][0] < 0.5]
        print(f"{n + 1}/{len(specs)} {cid}: J " + ' '.join(f"{a} {per_clip[a][-1][0]:.3f}" for a in ARMS), flush=True)
    montage(traces, f'{IMG}.png')
    boot = np.random.default_rng(1)
    res = {}
    print(f'\n{len(specs)} val clips; mean per clip J / P / R vs V_t; Δ vs pt [95% clip bootstrap]')
    for a in ARMS:
        v = np.array(per_clip[a]); dj = v[:, 0] - np.array(per_clip['pt'])[:, 0]
        bs = [dj[boot.integers(0, len(dj), len(dj))].mean() for _ in range(5000)]
        res[a] = dict(J=v[:, 0].mean(), P=v[:, 1].mean(), R=v[:, 2].mean(), dJ=dj.mean(),
                      ci=[np.percentile(bs, 2.5), np.percentile(bs, 97.5)], bad_J=float(np.mean([r[0] for r in bad[a]])))
        print(f"  {a:8s} {res[a]['J']:.3f} / {res[a]['P']:.3f} / {res[a]['R']:.3f}   ΔJ {dj.mean():+.3f} "
              f"[{res[a]['ci'][0]:+.3f}, {res[a]['ci'][1]:+.3f}]   bad-init stratum J {res[a]['bad_J']:.3f} "
              f"(n {len(bad[a])})")
    best = max(res['gt_S']['dJ'], res['gt_SO']['dJ'])
    print(f"KILL rule (max ΔJ gt_S, gt_SO ≤ +0.02): {'KILL' if best <= 0.02 else 'PASS'} ({best:+.3f})")
    json.dump(res, open(f'{OUT}/o1.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
