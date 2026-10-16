"""Sanity check (2026-10-16, fixed before running): does tracking on SAM2 feature maps beat the current blob tracker on
NFO? All 61 admissible windows, frames 1..W−1, GT boxes as reference (labels only, never inputs beyond t0).

Arms (all causal, all start from the t0 prompt box the benchmark gives every method):
  blob          run_t4.track: MOG2 blobs, Kalman/Hungarian, chain + readout_line on gaps (the current tracker)
  kernel-pix    fixed-size person kernel (w0 × h0 of the t0 box) searched in a window around a Theil–Sen
                constant-velocity prediction (x and y, last ≤ 8 estimates); score = mean of the map inside the
                kernel (integral image) = a matched filter for a box-shaped person on a foreground map
                (mean-shift-style fixed-kernel tracking, Comaniciu et al. CVPR 2000); map = clip(robust z of
                |frame − bg|, 0, 10), bg = warm-up median (diag_fine_add.zpix)
  kernel-f16    same tracker, map = 1 − cos(F_t, F_B), SAM2.1 b+ image_embed (stride 16), bilinear to native
  kernel-f8     same, high_res_feats stride 8
  register      (added 2026-10-16) the 6-frame shift measured DIRECTLY: normalised cross-correlation of the
                foreground difference features Δ = F − F_B (stride 16) of frame k−6 against frame k, inside the person box
                around the kernel-f16 position at k; search ±4 cells in x, ±1 in y around kernel-f16's own shift;
                parabolic sub-cell peak in x (phase/cross-correlation registration; FlowNet's correlation layer,
                arXiv 1504.06852). Only dx6 is defined for it.
One variable between kernel arms: the map. Metrics per window, mean over frames:
  cerr  ‖c − c_GT‖ / h_GT           (centre error in person heights)
  dx6   |Δx_est(k, k−6) − Δx_GT(k, k−6)| native px   (the composite-alignment quantity; k ≥ 6)
Paired window bootstrap vs blob. Trace: images/benchmark/rr_checks/track_check.png (x(t), 4 windows).

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.track_check
"""
import json

import cv2
import numpy as np
import torch
from scipy.stats import theilslopes

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.diag_fine_add import zpix
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native

OUT = f'{TR.RES}/track_check.json'
IMG = 'images/benchmark/rr_checks/track_check.png'
ARMS = ('blob', 'kernel-pix', 'kernel-f16', 'kernel-f8', 'register')
BATCH = int(__import__('os').environ.get('TC_BATCH', 1))


def gt_box(seq, k):
    bb = TR.bbs(seq).get(k)
    if not bb or bb[0].x < 0:
        return None
    return np.array(gt_to_native(bb[0], 800, 600), float)


def kernel_track(maps, c0, w0, h0):
    """maps: [T, H, W] float; c0 = t0 centre. Returns [T, 2] centres."""
    T, H, W = maps.shape
    out = [np.array(c0, float)]
    hw, hh = int(round(w0 / 2)), int(round(h0 / 2))
    for t in range(1, T):
        P = np.array(out[-8:])
        if len(P) >= 3:
            ts = np.arange(len(P))
            pred = np.array([theilslopes(P[:, i], ts)[0] * len(P) + theilslopes(P[:, i], ts)[1] for i in (0, 1)])
            v = np.abs(pred - P[-1])
        else:
            pred, v = P[-1], np.zeros(2)
        rx, ry = int(max(0.5 * w0, 3 * v[0] + 20)), int(max(0.25 * h0, 3 * v[1] + 10))
        I = cv2.integral(maps[t].astype(np.float64))
        best, bc = -np.inf, pred
        for cy in range(int(pred[1]) - ry, int(pred[1]) + ry + 1, 2):
            y0, y1 = np.clip([cy - hh, cy + hh], 0, H)
            if y1 <= y0:
                continue
            xs = np.arange(int(pred[0]) - rx, int(pred[0]) + rx + 1, 2)
            x0, x1 = np.clip(xs - hw, 0, W), np.clip(xs + hw, 0, W)
            area = np.maximum((x1 - x0) * (y1 - y0), 1)
            s = (I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]) / area
            i = int(np.argmax(s))
            if s[i] > best:
                best, bc = s[i], np.array([xs[i], cy], float)
        out.append(bc)
    return np.array(out)


CX, CY = 800 / 64, 600 / 64          # native px per stride-16 cell (SAM2 resizes to 1024² without keeping aspect)


def register_dx(Dt, Di, c, w0, h0, prior_px, R=4):
    """Shift (native px) of the person from frame i to frame t: NCC of Δ_t inside the person box at centre c against Δ_i
    shifted back by dx (cells), dx ∈ prior ± R, dy ∈ {−1, 0, 1}; parabolic sub-cell refinement in x."""
    C, H, W = Dt.shape
    x0, x1 = int(round((c[0] - w0 / 2) / CX)), int(round((c[0] + w0 / 2) / CX)) + 1
    y0, y1 = int(round((c[1] - h0 / 2) / CY)), int(round((c[1] + h0 / 2) / CY)) + 1
    x0, x1, y0, y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return prior_px
    a = Dt[:, y0:y1, x0:x1].astype(np.float32).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    p = int(round(prior_px / CX))
    sc = {}
    for dy in (-1, 0, 1):
        for dx in range(p - R, p + R + 1):
            ys, xs = y0 - dy, x0 - dx
            if ys < 0 or xs < 0 or ys + (y1 - y0) > H or xs + (x1 - x0) > W:
                continue
            b = Di[:, ys:ys + (y1 - y0), xs:xs + (x1 - x0)].astype(np.float32).ravel()
            sc[(dy, dx)] = float(np.dot(a, (b - b.mean()) / (b.std() + 1e-6)) / a.size)
    if not sc:
        return prior_px
    (dy, dx), _ = max(sc.items(), key=lambda kv: kv[1])
    l, m, r = sc.get((dy, dx - 1)), sc[(dy, dx)], sc.get((dy, dx + 1))
    off = 0.0 if l is None or r is None or (l - 2 * m + r) >= 0 else 0.5 * (l - r) / (l - 2 * m + r)
    return (dx + off) * CX


def feat_maps(pred, frames, bg):
    up = lambda c: torch.nn.functional.interpolate(c[None, None], size=frames.shape[1:3], mode='bilinear', align_corners=False)[0, 0]
    rgb = lambda x: np.stack([np.clip(x, 0, 255).astype(np.uint8)] * 3, -1)
    with torch.inference_mode():
        pred.set_image(rgb(bg))
        B16, B8 = pred._features['image_embed'][0].float(), pred._features['high_res_feats'][1][0].float()
        m16, m8, d16 = [], [], []
        for i in range(0, len(frames), BATCH):                 # BATCH 1: shares the GPU with a training run
            pred.set_image_batch([rgb(x) for x in frames[i:i + BATCH]])
            F16, F8 = pred._features['image_embed'].float(), pred._features['high_res_feats'][1].float()
            for a, b in zip(F16, F8):
                m16.append(up(1 - torch.nn.functional.cosine_similarity(a, B16, dim=0)).cpu().numpy())
                m8.append(up(1 - torch.nn.functional.cosine_similarity(b, B8, dim=0)).cpu().numpy())
                d16.append((a - B16).half().cpu().numpy())
    return np.stack(m16), np.stack(m8), np.stack(d16)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    pred = R.image_predictor()
    res, traces = {a: [] for a in ARMS}, []
    for n, t in enumerate(T):
        seq, fr = t['seq'], t['frames']
        G = [gt_box(seq, k) for k in fr]
        if G[0] is None:
            continue
        w0, h0 = G[0][2] - G[0][0], G[0][3] - G[0][1]
        c0 = ((G[0][0] + G[0][2]) / 2, (G[0][1] + G[0][3]) / 2)
        w0_, w1_ = t['warmup']
        bg = np.median(np.stack([V.native(i, seq) for i in range(w0_, w1_ + 1)]), 0).astype(np.float32)
        frames = np.stack([V.native(k, seq).astype(np.float32) for k in fr])
        Tk = R.track(t)
        blob = np.array([TR.to_native(*(Tk['chain'].get(j) or R.readout_line(Tk['chain'], j)[:2])) for j in range(len(fr))])
        blob[0] = c0
        mp = np.stack([np.clip(zpix(x, bg), 0, 10) for x in frames])
        m16, m8, d16 = feat_maps(pred, frames, bg)
        est = {'blob': blob, 'kernel-pix': kernel_track(mp, c0, w0, h0), 'kernel-f16': kernel_track(m16, c0, w0, h0),
               'kernel-f8': kernel_track(m8, c0, w0, h0)}
        k16 = est['kernel-f16']
        est['register'] = np.full_like(k16, np.nan)                 # only the 6-frame shifts are defined
        reg_dx = np.full(len(fr), np.nan)
        from benchmark.nfo_vos.layers.align import shifts as align_shifts
        for k in range(6, len(fr)):
            reg_dx[k] = align_shifts(d16[k - 6:k - 5], d16[k], k16[k], (w0, h0), [k16[k, 0] - k16[k - 6, 0]], (CX, CY))[0][0]
        gc = np.array([((g[0] + g[2]) / 2, (g[1] + g[3]) / 2, g[3] - g[1]) if g is not None else (np.nan,) * 3 for g in G])
        for a, e in est.items():
            cerr = np.linalg.norm(e[1:] - gc[1:, :2], axis=1) / gc[1:, 2]
            dx = np.abs((reg_dx[6:] if a == 'register' else e[6:, 0] - e[:-6, 0]) - (gc[6:, 0] - gc[:-6, 0]))
            res[a].append((np.nanmean(cerr), np.nanmean(dx), t['id'], seq))
        if len(traces) < 4 and n % 15 == 0:
            traces.append((t['id'], gc, est))
        print(f"{n + 1}/{len(T)} {t['id']}: dx6 " + ' '.join(f"{a} {res[a][-1][1]:.1f}" for a in ARMS), flush=True)
    rng = np.random.default_rng(0)
    summary = {}
    print(f'\n{len(res["blob"])} windows: mean over windows of per-window means   cerr (person heights) | dx6 (native px)')
    for a in ARMS:
        c, d = np.array([r[0] for r in res[a]]), np.array([r[1] for r in res[a]])
        summary[a] = dict(cerr=float(np.nanmean(c)), dx6=float(np.nanmean(d)))
        line = f'  {a:11s} {np.nanmean(c):.3f} | {np.nanmean(d):5.1f}'
        if a != 'blob':
            for i, m in ((0, 'cerr'), (1, 'dx6')):
                diff = np.array([r[i] for r in res[a]]) - np.array([r[i] for r in res['blob']])
                diff = diff[np.isfinite(diff)]
                if not len(diff):
                    continue
                bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
                line += f'   Δ{m} vs blob {diff.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]'
                summary[a][f'd_{m}'] = [float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(line)
    for s in ('seq1', 'seq2', 'seq3', 'seq4'):
        print(f'  {s}: dx6 ' + ' '.join(f"{a} {np.nanmean([r[1] for r in res[a] if r[3] == s]):.1f}" for a in ARMS))
    json.dump(summary, open(OUT, 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(traces), figsize=(10, 2.6), dpi=80)
    for k, (tid, gc, est) in enumerate(traces):
        ax[k].plot(gc[:, 0], 'k', lw=2, label='GT')
        for a, e in est.items():
            if a != 'register':
                ax[k].plot(e[:, 0], lw=1, label=a)
        ax[k].set_title(tid, fontsize=8); ax[k].set_xlabel('frame'); ax[k].set_ylabel('x (native px)')
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(IMG)


if __name__ == '__main__':
    main()


def kernel16_chain(t, pred, return_feats=False):
    """The kernel-f16 track of one window as a run_t4-style chain {local j: (x224, y224)} for every frame, so V̂'s
    composite code (learned_thresh.features: Theil–Sen on the chain, anchor for the point prompt) can use it unchanged."""
    seq, fr = t['seq'], t['frames']
    g = gt_box(seq, fr[0])
    w0_, w1_ = t['warmup']
    bg = np.median(np.stack([V.native(i, seq) for i in range(w0_, w1_ + 1)]), 0).astype(np.float32)
    m16, _, d16 = feat_maps(pred, np.stack([V.native(k, seq).astype(np.float32) for k in fr]), bg)
    c = kernel_track(m16, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), g[2] - g[0], g[3] - g[1])
    s = max(TR.NATIVE_W, TR.NATIVE_H)
    chain = {j: ((x + (s - TR.NATIVE_W) / 2) * TR.SIZE / s - 0.5, (y + (s - TR.NATIVE_H) / 2) * TR.SIZE / s - 0.5)
             for j, (x, y) in enumerate(c)}
    return (chain, d16, c, (g[2] - g[0], g[3] - g[1])) if return_feats else chain
