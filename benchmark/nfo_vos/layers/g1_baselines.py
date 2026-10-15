"""G1 table before any training (plan_feature_layers_training.md §4, step 3): val subjects, noisy inputs, jittered boxes.

Score at level ℓ ∈ {s8, s4}: s = cos(F_t, F_obj) − cos(F_t, F_bg), bilinearly upsampled to 540×960 (oracle_visibility).
F_t = Enc(noisy X_t); F_bg = Enc(pixel median of the noisy warm-up) for all arms except 'oracle-clean'.
Arms (F_obj):
  none        F_t                                    (do-nothing control: s = 1 − cos(F_t, F_bg))
  hand        Enc(factorised_gtfree.composite_frames of noisy X_{t−6..t−1}, box similarity warps, gate vs bg)
  warp        Σ_{i=t−6..t} w_i · warp(F_i),  w_i ∝ exp(−(t−i)/3.5) (a4.TAU_AGE), FGFA-style, no learning
  warp-t      the same over t−6..t−1 (diagnostic: isolates including F_t, which 'hand' does not)
  oracle      Enc(clean_t), noisy F_t and F_bg
  oracle-clean Enc(clean_t), F_t = Enc(X_t before photometric noise; sway kept), F_bg = Enc(B)
Nuisance arms (one variable each vs 'base' = the step-3 table, same 120 clip specs and seeds):
  flutter  + non-rigid flutter field (noise.flutter_fields, RMS 2.1 px, ρ 0.97, calibrated to NFO seq2's residual std)
  chunks   occluder = union of perlin clump 12 and 24 (each at density 1 − √(1 − ρ)), instead of clump 24 alone
Boxes: GT silhouette boxes jittered with σ_a = 3 px (scale × (1 + N(0, 0.03))). Targets t = 6, 9, …, 39.
AUC: visible V by fragment-size bin (true SAM2 cells) vs occluded S ∖ V. Pixels capped at CAP per clip × bin;
CIs: clip bootstrap (B = 500), paired across arms (same resampled clips).
"""
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn

from benchmark.nfo_vos import run_t4 as R
from benchmark.nfo_vos.factorised_gtfree import composite_frames
from benchmark.nfo_vos.layers import dataset as D
from benchmark.nfo_vos.synth import noise
from benchmark.nfo_vos.synth.oracle_visibility import BINS, bin_ids, cosmap, feats, up

NUISANCE = dict(base={}, flutter=dict(flutter=noise.FLUTTER_AMP), chunks=dict(clump=(12, 24)))
OUT = 'results/benchmark/rr/layers_g1'                 # + '_<arm>' for arms other than base
IMG = 'images/benchmark/gpjatk_foliage/g1_smaps.png'   # likewise
ARMS = ('none', 'hand', 'warp', 'warp-t', 'oracle', 'oracle-clean')
LEVELS = {'s8': 1, 's4': 2}                 # index into feats() = [stride 16, 8, 4]
SIGMA_A, CAP, NBOOT, TAU_AGE = 3.0, 1500, 500, 3.5


def warp_feat(F, bi, bt):
    """Feature map F [C, h, w] of frame i resampled onto frame t's grid by the box similarity A_{i→t}."""
    C, h, w = F.shape
    xs = (torch.arange(w, device=F.device) + 0.5) * W_ / w - 0.5          # target cell centres, native px
    ys = (torch.arange(h, device=F.device) + 0.5) * H_ / h - 0.5
    s = bt[2] / bi[2]
    sx, sy = bi[0] + (xs - bt[0]) / s, bi[1] + (ys - bt[1]) / s           # inverse map into frame i
    gx, gy = (2 * sx + 1) / W_ - 1, (2 * sy + 1) / H_ - 1                   # align_corners=False normalisation
    grid = torch.stack(torch.meshgrid(gy, gx, indexing='ij')[::-1], -1)[None].float()
    return Fn.grid_sample(F[None].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)[0]


W_, H_ = D.W, D.H


def run_clip(pred, spec, rng, **cfg):
    d = D.make_clip(*spec, **cfg)
    jb = D.jitter(d['boxes'], SIGMA_A, rng)
    ts = D.valid_targets(d['boxes'])
    res = {(a, l, b): [] for a in ARMS for l in LEVELS for b in range(len(BINS))}
    res.update({(a, l, 'neg'): [] for a in ARMS for l in LEVELS})
    smap = None
    with torch.inference_mode():
        lv = lambda f: [f[i] for i in LEVELS.values()]
        Fx = {i: lv(feats(pred, d['X'][i])) for i in range(min(ts) - D.N_HIST + 1, max(ts) + 1)}
        Fbg, FB = lv(feats(pred, d['bg'])), lv(feats(pred, d['B']))
        for t in ts:
            H = range(t - D.N_HIST + 1, t + 1)
            comp = composite_frames([d['X'][i] for i in H][:-1], [D.sim(jb[i], jb[t]) for i in H][:-1], d['bg'])
            Fh, Fc, F0 = lv(feats(pred, comp)), lv(feats(pred, d['clean'][t])), lv(feats(pred, d['X0'][t]))
            w = np.exp(-(t - np.array(H)) / TAU_AGE)
            occl, bid = d['S'][t] & ~d['V'][t], bin_ids(d['V'][t])
            for li, l in enumerate(LEVELS):
                Wp = [warp_feat(Fx[i][li], jb[i], jb[t]) for i in H]
                fw = sum(wi * f for wi, f in zip(w / w.sum(), Wp))
                fwt = sum(wi * f for wi, f in zip(w[:-1] / w[:-1].sum(), Wp[:-1]))
                Ft = Fx[t][li]
                objs = dict(none=(Ft, Ft, Fbg[li]), hand=(Ft, Fh[li], Fbg[li]), warp=(Ft, fw, Fbg[li]),
                            **{'warp-t': (Ft, fwt, Fbg[li])}, oracle=(Ft, Fc[li], Fbg[li]),
                            **{'oracle-clean': (F0[li], Fc[li], FB[li])})
                for a, (ft, fo, fb) in objs.items():
                    s = up(cosmap(ft, fo)) - up(cosmap(ft, fb))
                    res[(a, l, 'neg')].append(s[occl])
                    for b in range(len(BINS)):
                        res[(a, l, b)].append(s[bid == b])
                    if l == 's4' and smap is None and a == 'none':
                        smap = dict(t=t, X=d['X'][t], V=d['V'][t], S=d['S'][t])
                    if l == 's4' and smap is not None and smap['t'] == t:
                        smap[a] = s
    out = {}
    for k, v in res.items():
        v = np.concatenate(v).astype(np.float32)
        out[k] = rng.choice(v, CAP, replace=False) if len(v) > CAP else v
    return d['cid'], out, smap


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ns = np.sort(neg)
    return float(((np.searchsorted(ns, pos, 'left') + np.searchsorted(ns, pos, 'right')) / 2).mean() / len(ns))


def table(clips, rng):
    keys = [b for b in range(len(BINS))] + ['all']
    pool = lambda idx, a, l, b: np.concatenate([clips[i][(a, l, b)] for i in idx] if b != 'all' else
                                               [clips[i][(a, l, bb)] for i in idx for bb in range(len(BINS))])
    neg = lambda idx, a, l: np.concatenate([clips[i][(a, l, 'neg')] for i in idx])
    allidx = np.arange(len(clips))
    point = {(a, l, b): auc(pool(allidx, a, l, b), neg(allidx, a, l)) for a in ARMS for l in LEVELS for b in keys}
    boot = {(a, l, b): [] for a in ARMS for l in LEVELS for b in (0, 'all')}
    for _ in range(NBOOT):
        idx = rng.integers(0, len(clips), len(clips))
        for a in ARMS:
            for l in LEVELS:
                ng = neg(idx, a, l)
                for b in (0, 'all'):
                    boot[(a, l, b)].append(auc(pool(idx, a, l, b), ng))
    return point, {k: np.array(v) for k, v in boot.items()}


def save_smaps(smaps, path):
    rows = []
    for m in smaps[:3]:
        ys, xs = np.nonzero(m['S']); cy, cx, c = int(ys.mean()), int(xs.mean()), 200
        y0, x0 = np.clip(cy - c // 2, 0, H_ - c), np.clip(cx - c // 2, 0, W_ - c)
        cr = lambda im: im[y0:y0 + c, x0:x0 + c]
        g = np.clip(cr(m['X']), 0, 255).astype(np.uint8)
        vis = np.stack([g] * 3, -1).copy(); vv, oc = cr(m['V']), cr(m['S'] & ~m['V'])
        vis[vv] = (0.5 * vis[vv] + [0, 127, 0]).astype(np.uint8); vis[oc] = (0.5 * vis[oc] + [0, 0, 127]).astype(np.uint8)
        tiles = [vis]
        for a in ARMS:
            s = cr(m[a]); s = cv2.applyColorMap(np.clip((s + 1) * 127.5, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.putText(s, a, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1); tiles.append(s)
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(path, np.concatenate(rows, 0))


def main(arm='base', limit=None):
    out, img = (OUT, IMG) if arm == 'base' else (f'{OUT}_{arm}', IMG.replace('.png', f'_{arm}.png'))
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    pred = R.image_predictor()
    specs = D.clip_specs('val')[:limit]
    clips, smaps, t0 = [], [], time.time()
    for k, spec in enumerate(specs):
        f = f'{out}/clip_{k:03d}.npz'
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); clips.append(z['res'].item()); continue
        cid, res, sm = run_clip(pred, spec, rng, **NUISANCE[arm])
        np.savez(f, res=res, cid=cid); clips.append(res)
        if sm is not None and len(smaps) < 3 and k % 40 == 0:
            smaps.append(sm)
        print(f'{k + 1}/{len(specs)} {cid} {time.time() - t0:.0f}s', flush=True)
    if smaps:
        save_smaps(smaps, img)
    point, boot = table(clips, np.random.default_rng(1))
    names = [f'[{lo:g},{hi:g})' for lo, hi in BINS]
    lines = [f'G1 baselines [{arm}], val p21–p26, {len(clips)} clips, σ_a = {SIGMA_A} px, noise U[1,5]',
             'level arm          all   ' + '  '.join(f'{n:>8s}' for n in names) + '   sub-cell 95% CI (clip bootstrap)']
    for l in LEVELS:
        for a in ARMS:
            ci = np.nanpercentile(boot[(a, l, 0)], [2.5, 97.5])
            lines.append(f'{l:5s} {a:12s} {point[(a, l, "all")]:.3f} ' + '  '.join(f'{point[(a, l, b)]:8.3f}' for b in range(len(BINS)))
                         + f'   [{ci[0]:.3f}, {ci[1]:.3f}]')
    lines.append('paired Δ sub-cell AUC vs none (95% CI):')
    for l in LEVELS:
        for a in ARMS[1:]:
            dlt = boot[(a, l, 0)] - boot[('none', l, 0)]
            lines.append(f'  {l} {a:12s} {point[(a, l, 0)] - point[("none", l, 0)]:+.3f} [{np.nanpercentile(dlt, 2.5):+.3f}, {np.nanpercentile(dlt, 97.5):+.3f}]')
    for a, b in (('warp', 'hand'), ('warp', 'warp-t'), ('oracle-clean', 'oracle')):
        dlt = boot[(a, 's4', 0)] - boot[(b, 's4', 0)]
        lines.append(f'  s4 {a} − {b}: {point[(a, "s4", 0)] - point[(b, "s4", 0)]:+.3f} [{np.nanpercentile(dlt, 2.5):+.3f}, {np.nanpercentile(dlt, 97.5):+.3f}]')
    print('\n'.join(lines))
    open(f'{out}/g1_table.txt', 'w').write('\n'.join(lines) + '\n')
    json.dump({f'{a}|{l}|{b}': v for (a, l, b), v in point.items()}, open(f'{out}/g1_point.json', 'w'), indent=1)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='base', choices=list(NUISANCE))
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    main(a.arm, a.limit)
