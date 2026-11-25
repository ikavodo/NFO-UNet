"""The feat_probe question on real NFO footage: does the learned feature integral Ĉ carry, on NFO, the person information
the encoded pixel integral enc(C) carries? (On synthetic data it does: findings §10.x item 7; the adapters' use of it
differs only on NFO, item 8.)

Inputs are taken from the deployed inference path (inference.masks_from_arrays, `probe` hook) on the 8 seq1 full-pass
crossings, frames with history (j ≥ 6) and a GT box. Sets (s16, 256 ch each): none = [F_t, B]; + pixel enc(C) (token_seed0
it200); + learned Ĉ (feat it700 aggregator, joint-trained); + stage0c Ĉ (the reconstruction-pretrained aggregator,
never joint-trained). Labels from the GT box only (no amodal masks on NFO): person = cell centre in the box core (central
50% width, 80% height); background = cell centre outside the box by ≥ half a cell, within the box expanded by 50% per side;
other cells dropped. 'occl' subset: person cells whose current-frame change 1 − cos(F_t, B) is in the lowest third of the
crossing's person cells (they look like the empty scene now, i.e. occluded), vs all background; defined from inputs shared
by every set. Reader: standardised logistic regression (and the 1-hidden-layer MLP) from feat_probe, leave-one-crossing-out
over the 8 crossings; per-crossing AUC, paired differences with a crossing bootstrap. seq1 only (selection data).

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_probe_nfo
"""
import json
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
from benchmark.nfo_vos.adapter.inference import masks_from_arrays
from benchmark.nfo_vos.adapter.tracker import gt_box
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_probe import fit, fit_mlp

A = 'results/benchmark/rr/layers_fusion_adapter'
NETS = {'pixel': f'{A}/token_seed0/fusion_it200.pt', 'learned': f'{A}/feat_seed0/feat_seed0/fusion_it700.pt',
        'stage0c': f'{A}/feat_seed0/feat_seed0/fusion_it700.pt'}           # Fusion unused here; aggregator swapped below
STAGE0C = 'results/benchmark/rr/feat_integral/stage0c_seed0.pt'
SETS = ('none', 'pixel', 'learned', 'stage0c')
OUT = 'results/benchmark/rr/feat_integral/probe_nfo'
G = 64                                                    # s16 grid of SAM2's 1024² input


def labels(box):
    """GT box (native xyxy) -> (keep, y) on the 64×64 grid; None if < 4 person cells."""
    cx = (np.arange(G) + 0.5) * TR.NATIVE_W / G; cy = (np.arange(G) + 0.5) * TR.NATIVE_H / G
    X, Y = np.meshgrid(cx, cy)
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0; mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    core = (abs(X - mx) <= 0.25 * w) & (abs(Y - my) <= 0.4 * h)
    hx, hy = TR.NATIVE_W / G / 2, TR.NATIVE_H / G / 2
    out = (X < x0 - hx) | (X > x1 + hx) | (Y < y0 - hy) | (Y > y1 + hy)
    reg = (abs(X - mx) <= w) & (abs(Y - my) <= h)
    if core.sum() < 4:
        return None
    return core | (out & reg), core


@torch.no_grad()
def crossing(pred, nets, t):
    """{set: (cells, 256·k) fp16}, y, change, for one crossing; one masks_from_arrays run per net (same tracker)."""
    seq, fr = t['seq'], t['frames']
    w0, w1 = t['warmup']
    bg = np.median(np.stack([TR.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    Xn = [TR.native(k, seq) for k in fr]
    lab = {j: labels(b) for j, k in enumerate(fr) if (b := gt_box(seq, k)) is not None}
    lab = {j: v for j, v in lab.items() if v is not None}
    rows = {s: {} for s in SETS}; ys, ch = {}, {}
    for nm, net in nets.items():
        def hook(j, f):
            if j not in lab:
                return
            keep, y = lab[j]; m = torch.from_numpy(keep).cuda()
            rows[nm][j] = f['C16'][0][:, m].T.half().cpu()
            if nm == 'pixel':                                 # shared inputs: once
                rows['none'][j] = torch.cat([f['X16'][0], f['B16'][0]], 0)[:, m].T.half().cpu()
                ys[j] = y[keep]
                ch[j] = (1 - torch.nn.functional.cosine_similarity(f['X16'][0].float(), f['B16'][0].float(), dim=0))[m].cpu().numpy()
        masks_from_arrays(pred, net, Xn, bg, gt_box(seq, fr[0]), probe=hook)
    js = sorted(set.intersection(*[set(r) for r in rows.values()]))
    j_vis = js[len(js) // 2]                                  # visual trace: label cells on the middle frame
    keep, y = lab[j_vis]; im = cv2.cvtColor(Xn[j_vis], cv2.COLOR_GRAY2BGR)
    ov = np.zeros((G, G, 3), np.uint8); ov[keep & y] = (0, 0, 255); ov[keep & ~y] = (255, 0, 0)
    ov = cv2.resize(ov, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f'{OUT}/labels_{t["id"]}.jpg', np.where(ov.any(2, keepdims=True), (0.6 * im + 0.4 * ov).astype(np.uint8), im))
    y = np.concatenate([ys[j] for j in js]); c = np.concatenate([ch[j] for j in js])
    occl = y & (c <= np.percentile(c[y], 33))
    return {s: torch.cat([rows[s][j] for j in js]) for s in SETS}, y, occl


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = f'{OUT}/cells_seq1.pt'
    trials = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    if os.path.exists(cache):
        data = torch.load(cache, weights_only=False)
    else:
        pred = image_predictor()
        nets = {nm: load_net(pred, c) for nm, c in NETS.items()}
        nets['stage0c'].agg.load_compat(torch.load(STAGE0C, map_location='cuda'))
        data = []
        for t in trials:
            data.append(crossing(pred, nets, t))
            print(f"{t['id']}: {len(data[-1][1])} cells, person {data[-1][1].mean():.2f}, occl {data[-1][2].sum()}", flush=True)
        torch.save(data, cache)
    res, rng = {}, np.random.default_rng(0)
    for reader in ('linear', 'mlp'):
        sc = {s: np.full((len(data), 2), np.nan) for s in SETS}       # per held-out crossing: (all-AUC, occl-AUC)
        for s in SETS:
            for k in range(len(data)):
                tr = [d for i, d in enumerate(data) if i != k]
                X = torch.cat([torch.cat([d[0]['none'], d[0][s]], 1) if s != 'none' else d[0]['none'] for d in tr])
                yt = np.concatenate([d[1] for d in tr])
                if reader == 'mlp':
                    f_ = fit_mlp(X, yt)
                else:
                    mu, sd, w, b = fit(X, yt); f_ = lambda Z, mu=mu, sd=sd, w=w, b=b: ((Z.cuda().float() - mu) / sd) @ w + b
                c, y, oc = data[k]
                Z = torch.cat([c['none'], c[s]], 1) if s != 'none' else c['none']
                p = f_(Z).cpu().numpy(); sel = oc | ~y
                sc[s][k] = roc_auc_score(y, p), roc_auc_score(y[sel], p[sel])
                del X
        for s in SETS:
            print(f"{reader} {s:8s} AUC all {np.mean(sc[s][:, 0]):.3f} occl {np.mean(sc[s][:, 1]):.3f}", flush=True)
        for a, b in (('pixel', 'none'), ('learned', 'none'), ('stage0c', 'none'), ('learned', 'pixel'), ('stage0c', 'pixel'),
                     ('learned', 'stage0c')):
            for j, nm in enumerate(('all', 'occl')):
                d = sc[a][:, j] - sc[b][:, j]
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                ci = np.percentile(bs, [2.5, 97.5])
                res[f'{reader}_{a}-{b}_{nm}'] = [float(d.mean()), *map(float, ci), d.tolist()]
                print(f"{reader} {a} − {b} {nm}: {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  per crossing "
                      + ' '.join(f'{x:+.3f}' for x in d), flush=True)
    json.dump(res, open(f'{OUT}/probe_nfo_seq1.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
