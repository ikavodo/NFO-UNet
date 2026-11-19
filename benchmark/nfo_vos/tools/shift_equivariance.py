"""G0 (plan docs/superpowers/plans/2026-11-19-layer-sam-feature-integral.md, Task 4): are SAM2.1-b+ features shift
equivariant under SUBJECT motion over a static scene, closely enough to align past frames' features by the tracker shift?

For 30 synthetic val clips (middle frame): P0 = person over the empty scene B; P_Δ = the same person translated by Δ in x
over the same B. Compare shift_Δ(E(P0)) (roll / bilinear, `feat_integral.shift_keys`) with E(P_Δ) on the person cells of
P_Δ at each stride:  ε = 1 − cos (centred at stride 16);  reference contrast δ = 1 − cos(E(P_Δ), E(B)) on the same cells
(what the tracker and gate rely on); do-nothing baseline ε0 = 1 − cos(E(P0), E(P_Δ)) unshifted.
Pass (pre-registered): p90 ε16 < ½ median δ16 at every Δ, for at least one mode.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.shift_equivariance [n_clips]
"""
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

from benchmark.nfo_vos.adapter import dataset as D
from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, BG_SPLIT, _enc, u8
from benchmark.nfo_vos.adapter.feat_integral import ccos, shift_keys
from benchmark.nfo_vos.synth import gpjatk_foliage as GF

CELLS = (0.25, 0.5, 1.0, 2.0, 4.0)           # Δ in stride-16 cells (1 cell = 16·960/1024 = 15 clip px ≈ 12.5 NFO native px)
OUT = 'results/benchmark/rr/feat_integral'
IMG = 'images/benchmark/feat_integral'


def cover(m, hw):
    return cv2.resize(m.astype(np.float32), (hw, hw), interpolation=cv2.INTER_AREA) > 0.5   # SAM2: 1024² anisotropic


@torch.no_grad()
def main(n_clips=30):
    os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    pred = image_predictor()
    mu = pred.model.no_mem_embed.detach().float().reshape(1, -1, 1, 1)
    GF.set_bg_pool(BG_SPLIT['val'])
    H, W = GF.H, GF.W
    rows, t0, ex = [], time.time(), None
    for ci, (seq, cam, rho, seed) in enumerate(D.clip_specs('val')[:n_clips]):
        *_, L = GF.make_layered_clip(seq, cam, rho, seed, 'perlin', None, n=8, save=False, layers=True)
        t = int(np.argmax(L['mf'].sum((1, 2))))
        I, mf, B = L['I'][t].astype(np.float32), L['mf'][t].astype(np.float32), L['B'].astype(np.float32)
        imgs, masks = [B, mf * I + (1 - mf) * B], []
        for c in CELLS:
            M = np.float32([[1, 0, c * 16 * W / 1024], [0, 1, 0]])
            Is = cv2.warpAffine(I, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ms = cv2.warpAffine(mf, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            imgs.append(ms * Is + (1 - ms) * B); masks.append(ms)
        with torch.autocast('cuda', dtype=AMP):
            F16, F8, F4 = _enc(pred, np.stack([u8(x) for x in imgs]))
        Fs = {'16': F16.float(), '8': F8.float(), '4': F4.float()}
        for i, c in enumerate(CELLS):
            dx = torch.tensor([[c * 16.0]], device='cuda')
            for s, Fx in Fs.items():
                hw = Fx.shape[-1]
                m = torch.as_tensor(cover(masks[i], hw), device='cuda')
                mu_s = mu[0] if s == '16' else None          # [C, 1, 1] for [C, h, w] maps
                tgt, src, bg = Fx[2 + i], Fx[1], Fx[0]
                d = (1 - ccos(tgt, bg, mu_s))
                e0 = (1 - ccos(src, tgt, mu_s))
                for mode in ('roll', 'bilinear'):
                    sh, v = shift_keys(src[None, None], dx, int(s), mode)
                    ok = m & v[0, 0, 0]
                    e = 1 - ccos(sh[0, 0], tgt, mu_s)
                    rows += [dict(clip=ci, s=s, cells=c, mode=mode, eps=float(a), eps0=float(b), delta=float(g))
                             for a, b, g in zip(e[ok].tolist(), e0[ok].tolist(), d[ok].tolist())]
                    if ex is None and s == '16' and c == 1.0 and mode == 'bilinear':
                        ex = (imgs[1], imgs[2 + i], (e * ok).cpu().numpy())
        print(f'clip {ci + 1}/{n_clips} {time.time() - t0:.0f}s', flush=True)
    with open(f'{OUT}/g0_shift_cells.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'s':>3} {'Δcells':>6} {'mode':>8} | {'med ε':>7} {'p90 ε':>7} | {'med ε0':>7} | {'med δ':>7}  (n cells)")
    ok_mode = {'roll': True, 'bilinear': True}
    summary = []
    for s in ('16', '8', '4'):
        for c in CELLS:
            for mode in ('roll', 'bilinear'):
                r = [x for x in rows if x['s'] == s and x['cells'] == c and x['mode'] == mode]
                e = np.array([x['eps'] for x in r]); e0 = np.array([x['eps0'] for x in r]); d = np.array([x['delta'] for x in r])
                row = dict(s=s, cells=c, mode=mode, med_eps=np.median(e), p90_eps=np.quantile(e, 0.9),
                           med_eps0=np.median(e0), med_delta=np.median(d), n=len(e))
                summary.append(row)
                print(f"{s:>3} {c:>6} {mode:>8} | {row['med_eps']:7.3f} {row['p90_eps']:7.3f} | {row['med_eps0']:7.3f} | "
                      f"{row['med_delta']:7.3f}  ({len(e)})")
                if s == '16' and not row['p90_eps'] < 0.5 * row['med_delta']:
                    ok_mode[mode] = False
    with open(f'{OUT}/g0_shift_summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    if ex is not None:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.6))
        ax[0].imshow(ex[0], cmap='gray', vmin=0, vmax=255); ax[0].set_title('P0')
        ax[1].imshow(ex[1], cmap='gray', vmin=0, vmax=255); ax[1].set_title('P_Δ, Δ = 1 cell')
        im = ax[2].imshow(ex[2], vmin=0, vmax=0.5); ax[2].set_title('ε at s16 (person cells), bilinear')
        plt.colorbar(im, ax=ax[2]); [a.axis('off') for a in ax]
        fig.savefig(f'{IMG}/g0_example.png', dpi=80, bbox_inches='tight'); plt.close(fig)
    good = [m for m in ('bilinear', 'roll') if ok_mode[m]]
    print(f"\nG0 {'PASS (mode=' + good[0] + ')' if good else 'FAIL'}: rule p90 ε16 < ½ median δ16 at every Δ")


if __name__ == '__main__':
    main(*map(int, sys.argv[1:]))
