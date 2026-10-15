"""End-to-end oracle test (2026-10-15, user-approved option B): do perfect feature layers improve SAMURAI's modal mask?

Clips: 30 fixed val clips (dataset.clip_specs('val')[::4]), flutter arm (noise.FLUTTER_AMP), noisy frames.
T1 = SAMURAI video predictor, prompted at frame 0 with the S box + V's distance-transform peak (as
layered_experiment.run), logits at native 540×960.
Bias: logits' = logits + λ·s, s = up(cos(F_t, F_obj) − cos(F_t, F_bg)) at s4 (g1_baselines), F_bg = Enc(warm-up median).
  oracle  F_obj = Enc(clean_t)   (perfect composite: the ceiling of the learned-alignment idea)
  none    F_obj = F_t            (s = 1 − cos(F_t, F_bg): static background features only, no composite)
λ chosen on the first 10 clips (max mean J), reported on the other 20. J vs V (modal), frames 1…39, mean per clip.
PRE-REGISTERED KILL: oracle ΔJ (vs T1, paired clip bootstrap) ≤ +0.02 or CI ∋ 0 → stop the direction.
'none' answers whether any gain needs the composite at all (static F_bg alone).
"""
import os
import shutil
import tempfile

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R
from benchmark.nfo_vos.layers import dataset as D
from benchmark.nfo_vos.run_sam2_video import build_predictor
from benchmark.nfo_vos.synth import noise
from benchmark.nfo_vos.synth.oracle_visibility import cosmap, feats, up

OUT = 'results/benchmark/rr/layers_oracle_bias'
IMG = 'images/benchmark/gpjatk_foliage/oracle_bias.png'
LAMBDAS = (0, 1, 2, 4, 8, 16, 32, 64)
ARMS = ('oracle', 'none')
N_FIT = 10


def prompt(S0, V0):
    ys, xs = np.nonzero(S0)
    dt = cv2.distanceTransform(V0.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(dt), dt.shape)
    return np.array([[px, py]], np.float32), np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


def t1_logits(vp, X, S0, V0):
    stage = tempfile.mkdtemp(prefix='gpjb_')
    try:
        for i, x in enumerate(X):
            cv2.imwrite(f'{stage}/{i}.jpg', np.stack([np.clip(x, 0, 255).astype(np.uint8)] * 3, -1),
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
        pt, bx = prompt(S0, V0)
        L = np.zeros((len(X),) + X.shape[1:], np.float32)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            st = vp.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            vp.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=pt, labels=np.array([1], np.int32), box=bx)
            for fi, _, lg in vp.propagate_in_video(st):
                L[fi] = lg[0, 0].float().cpu().numpy()
    finally:
        shutil.rmtree(stage)
    return L


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def run_clip(vp, ip, spec, keep):
    d = D.make_clip(*spec, flutter=noise.FLUTTER_AMP)
    L = t1_logits(vp, d['X'], d['S'][0], d['V'][0])
    J = {a: np.full((len(LAMBDAS), D.N), np.nan) for a in ARMS}
    trace = None
    with torch.inference_mode():
        Fbg = feats(ip, d['bg'])[2]
        for t in range(1, D.N):
            if not d['V'][t].any():
                continue
            Ft, Fc = feats(ip, d['X'][t])[2], feats(ip, d['clean'][t])[2]
            cb = up(cosmap(Ft, Fbg))
            s = dict(oracle=up(cosmap(Ft, Fc)) - cb, none=1 - cb)
            for a in ARMS:
                for k, lam in enumerate(LAMBDAS):
                    J[a][k, t] = iou(L[t] + lam * s[a] > 0, d['V'][t])
            if keep and t == 20:
                trace = dict(X=d['X'][t], V=d['V'][t], L=L[t], **{f's_{a}': s[a] for a in ARMS})
    return d['cid'], {a: np.nanmean(J[a], 1) for a in ARMS}, trace


def boot(x, rng, n=2000):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def montage(traces, lam, path=IMG):
    rows = []
    for tr in traces:
        ys, xs = np.nonzero(tr['V']); cy, cx, c = int(ys.mean()), int(xs.mean()), 200
        y0, x0 = np.clip(cy - c // 2, 0, 540 - c), np.clip(cx - c // 2, 0, 960 - c)
        cr = lambda im: im[y0:y0 + c, x0:x0 + c]
        g = np.clip(cr(tr['X']), 0, 255).astype(np.uint8)

        def ov(m, lab):
            o = np.stack([g] * 3, -1).copy(); v = cr(tr['V'])
            o[m & v] = (0.5 * o[m & v] + [0, 127, 0]).astype(np.uint8)        # TP green
            o[m & ~v] = (0.5 * o[m & ~v] + [0, 0, 127]).astype(np.uint8)      # FP red
            o[~m & v] = (0.5 * o[~m & v] + [127, 0, 0]).astype(np.uint8)      # FN blue
            cv2.putText(o, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            return o
        tiles = [np.stack([g] * 3, -1), ov(cr(tr['L']) > 0, 'T1')]
        tiles += [ov(cr(tr['L'] + lam[a] * tr[f's_{a}']) > 0, f'T1+{lam[a]}s {a}') for a in ARMS]
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(path, np.concatenate(rows, 0))


def main():
    os.makedirs(OUT, exist_ok=True)
    specs = D.clip_specs('val')[::4]
    vp, ip = build_predictor('t1'), R.image_predictor()
    res, traces = [], []
    for k, spec in enumerate(specs):
        f = f'{OUT}/clip_{k:02d}.npz'
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); res.append(z['J'].item())
            if 'trace' in z.files and z['trace'].item() is not None:
                traces.append(z['trace'].item())
            continue
        cid, J, tr = run_clip(vp, ip, spec, keep=k % 10 == 0)
        np.savez_compressed(f, J=J, trace=tr, cid=cid); res.append(J)
        if tr is not None:
            traces.append(tr)
        print(f'{k + 1}/{len(specs)} {cid} T1 J {J["oracle"][0]:.3f}', flush=True)
    rng = np.random.default_rng(0)
    fit, ev = res[:N_FIT], res[N_FIT:]
    lam, lines = {}, [f'Oracle logit bias on SAMURAI, flutter, val: λ fit on {len(fit)} clips, reported on {len(ev)}']
    for a in ARMS:
        mf = np.mean([r[a] for r in fit], 0)
        k = int(np.argmax(mf)); lam[a] = LAMBDAS[k]
        lines.append(f'  fit {a:6s} mean J by λ: ' + ' '.join(f'{l}:{m:.3f}' for l, m in zip(LAMBDAS, mf)))
        j0, j1 = np.array([r[a][0] for r in ev]), np.array([r[a][k] for r in ev])
        lo, hi = boot(j1 - j0, rng)
        lines.append(f'  eval {a:6s} λ={lam[a]}: T1 J {j0.mean():.3f} → {j1.mean():.3f}, ΔJ {np.mean(j1 - j0):+.3f} [{lo:+.3f}, {hi:+.3f}]')
    ko, kn = LAMBDAS.index(lam['oracle']), LAMBDAS.index(lam['none'])
    dd = np.array([r['oracle'][ko] - r['none'][kn] for r in ev]); lo, hi = boot(dd, rng)
    lines.append(f'  eval oracle − none: {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')
    print('\n'.join(lines)); open(f'{OUT}/table.txt', 'w').write('\n'.join(lines) + '\n')
    if traces:
        montage(traces, lam)


if __name__ == '__main__':
    main()
