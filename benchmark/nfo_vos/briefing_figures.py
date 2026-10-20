"""Figures for the collaborator briefing (docs/briefing/, 2026-10-20).

  nfo_inputs.jpg      one seq2 / seq3 / seq4 hand frame per row: what the adapter sees (frame + tracker box, empty
                      scene, aligned change-gated composite C, evidence mass a) and what comes out (hand label,
                      SAMURAI, ours), cropped around the GT box
  synth_example.jpg   synthetic GPJATK-behind-foliage clips: frame, full person S, visible V, and the loss weight
                      classes (fusion_data.loss_weights) drawn on the frame

    TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.briefing_figures
"""
import json
import os

import cv2
import numpy as np
import torch

OUT = 'docs/briefing/figs'
PANEL = (220, 290)          # w, h of one cropped panel


def _crop(img, g, scale=(2.6, 1.5)):
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    w, h = max(g[2] - g[0], 60) * scale[0], (g[3] - g[1]) * scale[1]
    H, W = img.shape[:2]
    X0, Y0 = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0))
    return cv2.resize(img[Y0:int(min(cy + h / 2, H)), X0:int(min(cx + w / 2, W))], PANEL, interpolation=cv2.INTER_AREA)


def _rgb(x):
    x = np.clip(x, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x


def _overlay(gray, m, col, keep=None):
    v = _rgb(gray).astype(np.float32)
    v[m] = 0.45 * v[m] + 0.55 * np.array(col, np.float32)
    if keep is not None:                                    # void (ignored) pixels hatched grey
        v[~keep] = 0.5 * v[~keep] + 0.5 * 160
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    v = v.astype(np.uint8)
    cv2.drawContours(v, cs, -1, col, 1)
    return v


def _label(p, text):
    cv2.rectangle(p, (0, 0), (PANEL[0], 18), (255, 255, 255), -1)
    cv2.putText(p, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return p


def nfo_inputs():
    from benchmark.nfo_vos import run_t4 as R
    from benchmark.nfo_vos.downstream import box_metrics as BM, report as RP
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion
    from benchmark.nfo_vos.layers.video_adapter import S1, per_frame_inputs
    from benchmark.nfo_vos.downstream.dump_arms import CKPT
    pred = R.image_predictor()
    net = Fusion(token=True).cuda(); net.init_fallback(pred.model); net.load_state_dict(torch.load(CKPT)); net.eval()
    trials = json.load(open('results/benchmark/rr/trials.json'))
    rows = []
    for seq in ('seq2', 'seq3', 'seq4'):
        t, k = next((t, k) for t in trials if t['seq'] == seq for k in t['frames'][6:] if RP.hand_label(seq, k) is not None)
        j = t['frames'].index(k)
        with torch.no_grad():
            d = per_frame_inputs(pred, net, t)[j]
        it, g = d['it'], BM.gt_boxes(seq)[k]
        X = it['X'][0]
        H, W = X.shape
        C = it['Ct'][0]
        a = cv2.resize(it['a'][0].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        fr = _rgb(X); b = np.round(d['box'] / S1).astype(int)
        cv2.rectangle(fr, tuple(b[:2]), tuple(b[2:]), (255, 0, 255), 2)
        cv2.rectangle(fr, tuple(int(x) for x in g[:2]), tuple(int(x) for x in g[2:]), (0, 200, 0), 1)
        hand, keep = RP.hand_label(seq, k)
        ms = {}
        for arm in ('t1-c2', 'adapter-pf'):
            m, f_ = BM.load_arm(arm, t['id'])
            ms[arm] = m[[int(x) for x in f_].index(k)]
        J = lambda m: ((m & hand & keep).sum() / max(((m | hand) & keep).sum(), 1))
        panes = [(fr, f'{seq} frame {k}: tracker box'), (_rgb(it['bg']), 'empty scene B'),
                 (_rgb(C), 'composite C (t-6..t-1)'), (heat, 'evidence mass a'),
                 (_overlay(X, hand, (0, 200, 0), keep), 'hand label (grey: void)'),
                 (_overlay(X, ms['t1-c2'], (40, 110, 235)), f'SAMURAI  J {J(ms["t1-c2"]):.2f}'),
                 (_overlay(X, ms['adapter-pf'], (214, 120, 42)), f'ours  J {J(ms["adapter-pf"]):.2f}')]
        rows.append(np.hstack([_label(_crop(p, g), s) for p, s in panes]))
    cv2.imwrite(f'{OUT}/nfo_inputs.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def synth_example():
    from benchmark.nfo_vos.layers import fusion_data as FD
    from benchmark.nfo_vos.synth import gpjatk_foliage as GF
    GF.set_bg_pool(FD.BG_SPLIT['val'])
    rows = []
    for k, spec in enumerate(FD.specs('val')[:12]):
        raw = FD.clip_raw(spec, np.random.default_rng([1, k]))
        rho = raw['occ'].get('rho', 0)
        if rho < 0.25 or len(rows) >= 2 and rho < 0.45:
            continue
        T = len(raw['X'])
        t = max(range(T // 3, T), key=lambda i: raw['S'][i].sum() * (raw['V'][i].sum() > 0))
        X, S, V = raw['X'][t], raw['S'][t], raw['V'][t]
        ys, xs = np.nonzero(S)
        g = [xs.min(), ys.min(), xs.max(), ys.max()]
        near = cv2.dilate(V.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FD.NEAR_PX + 1,) * 2)) > 0
        occ = S & ~V
        wv = _rgb(X).astype(np.float32)
        for m, col in ((V, (0, 200, 0)), (occ & near, (0, 215, 255)), (occ & ~near, (60, 60, 230))):
            wv[m] = 0.35 * wv[m] + 0.65 * np.array(col, np.float32)
        panes = [(_rgb(X), f'synthetic frame, rho {rho:.2f}'), (_overlay(X, S, (200, 120, 40)), 'full person S'),
                 (_overlay(X, V, (0, 200, 0)), 'visible V (target)'),
                 (wv.astype(np.uint8), 'W: green 1, yellow 0, red 0.3')]
        rows.append(np.hstack([_label(_crop(p, g, (2.2, 1.35)), s) for p, s in panes]))
        if len(rows) == 3:
            break
    cv2.imwrite(f'{OUT}/synth_example.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    synth_example()
    nfo_inputs()
    print('wrote', OUT)
