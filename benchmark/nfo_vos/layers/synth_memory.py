"""Does SAM2 memory have headroom over the per-frame adapter? Decided on synthetic clips with EXACT visible masks
(docs/vos_benchmark/nfo_vos_pilot_findings.md §8 item 4; user, 2026-10-19). NFO oracles are bounded by label quality;
here the exact visible mask V can be written into memory, a true upper bound for anything memory can deliver.

Clips: the 30 fixed GPJATK + foliage val clips (fusion_data.Clips('val'), held-out subjects, held-out CanaTree
backgrounds), 40 frames, 540×960, inputs built exactly as in training (fusion_data.build: kernel-f16 tracker from the
jittered t0 box, gated composite, encoded features). Scored frames: build's targets (t ≥ max(6, n_h), V non-empty),
J / P / R vs V; clip-level means; paired clip bootstrap.
Arms
  trk            per-frame SAM2, tracker box + point (no adapter, no memory)
  adapter-pf     per-frame adapter v3.4 it200 (no memory)                                          (a)
  vid-prompts    SAM2 video, t0 box + centre, tracker box + point every frame (decoder only)       (b)
  vid-oracle-V   = vid-prompts, exact V written into memory (causal: frame t's own V after t)     (c)  bound
  vid-write-pf   = vid-prompts, adapter-pf's mask written into memory (no training)               (d)
  vid-adapter-pre  = vid-prompts + adapter residual before memory attention (zero-shot read side)
Reading: (c) ≤ (a) → memory has no headroom, video training is pointless. (c) > (a) → (d) shows what a corrected
write path captures without training; the rest is the target of memory-conditioned training.

  TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.synth_memory [n_clips]
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R
from benchmark.nfo_vos.layers import fusion_data as FD
from benchmark.nfo_vos.layers.fusion_adapter import Fusion, decode
from benchmark.nfo_vos.layers.video_adapter import run_window
from benchmark.nfo_vos.run_sam2_video import build_predictor

CKPT = os.environ.get('CKPT', 'results/benchmark/rr/layers_fusion_adapter/token_seed0/fusion_it200.pt')
OUT = 'results/benchmark/rr/synth_memory' + os.environ.get('TAG', '')
IMG = f"images/benchmark/downstream/synth_memory{os.environ.get('TAG', '')}.jpg"
ARMS = ('trk', 'adapter-pf', 'vid-prompts', 'vid-oracle-V', 'vid-write-pf', 'step0-post')


def up(low, shape):
    return cv2.resize(low.float().cpu().numpy(), shape[::-1], interpolation=cv2.INTER_LINEAR) > 0


@torch.no_grad()
def clip_arms(pred, net, vp, raw):
    """{arm: [T, H, W] bool} and the scored frames. Review fixes (2026-10-19): A_w runs on EVERY frame with history
    (no V gate: the bank must not depend on GT); scored frames = those with V non-empty; object score of a written mask
    = +10 if non-empty else −10 (hook). step0-post: prompts-only loop with A_w written into memory, F_t^mem captured,
    A_w's residuals applied AFTER memory and decoded OUTSIDE the loop (its output never enters memory or obj_ptr)."""
    T, H, W = raw['X'].shape
    f, meta = FD.build(pred, raw, all_targets=True)
    ts = meta['ts']
    score_ts = [t for t in ts if raw['V'][t].any()]
    sc = np.array([1024 / W, 1024 / H] * 2, np.float32)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = net(f, sam=pred.model)
        low_ad = decode(pred.model, o, token=o['token'])
        low_trk = decode(pred.model, dict(f, box=f['box_trk']), point=True)
    M = {a: np.zeros((T, H, W), bool) for a in ('trk', 'adapter-pf', 'step0-post')}
    for i, t in enumerate(ts):
        M['trk'][t], M['adapter-pf'][t] = up(low_trk[i, 0], (H, W)), up(low_ad[i, 0], (H, W))
    trk = meta['trk']
    S0 = raw['S'][0] if raw['S'][0].any() else raw['S'][int(np.argmax(raw['S'].sum((1, 2))))]
    ys, xs = np.nonzero(S0); asp0 = (np.ptp(xs) + 1) / (np.ptp(ys) + 1); h0 = raw['bxj'][0][2]
    hb = np.array([asp0 * h0 / 2, h0 / 2], np.float32)
    inputs = [dict(box=(np.concatenate([trk[j] - hb, trk[j] + hb]) * sc).astype(np.float32),
                   pt=(trk[j] * sc[:2]).astype(np.float32), R=None, delta=None) for j in range(T)]
    assert np.allclose(np.stack([inputs[t]['box'] for t in ts]), f['box_trk'].cpu().numpy(), atol=1e-2)   # review 13
    cx0, cy0, _ = raw['bxj'][0]
    box0 = [cx0 - hb[0], cy0 - hb[1], cx0 + hb[0], cy0 + hb[1]]
    tv = dict(frames=list(range(T)), imgs=list(raw['X']), box_native=box0, point_native=[cx0, cy0])
    M['vid-prompts'] = run_window(vp, tv, inputs, 'prompts')
    M['vid-oracle-V'] = run_window(vp, tv, inputs, 'oracle-mem', labels={t: raw['V'][t] for t in range(T)})
    cap = dict(ts=set(ts), F={})
    M['vid-write-pf'] = run_window(vp, tv, inputs, 'oracle-mem', labels={t: M['adapter-pf'][t] for t in ts}, capture=cap)
    idx = [i for i, t in enumerate(ts) if t in cap['F']]
    Fm = torch.cat([cap['F'][ts[i]][0] for i in idx]); h4 = torch.cat([cap['F'][ts[i]][1][0] for i in idx])
    h8 = torch.cat([cap['F'][ts[i]][1][1] for i in idx])
    sel = torch.as_tensor(idx, device='cuda')
    g = {'X16': Fm + (o['X16'] - f['X16'])[sel], 'X8': h8 + (o['X8'] - f['X8'])[sel], 'X4': h4 + (o['X4'] - f['X4'])[sel]}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        low_s0 = decode(pred.model, g, token=o['token'][sel])
    for k, i in enumerate(idx):
        M['step0-post'][ts[i]] = up(low_s0[k, 0], (H, W))
    empty_w = float(np.mean([not M['adapter-pf'][t].any() for t in ts]))
    return M, score_ts, empty_w


def jpr(m, v):
    tp = (m & v).sum()
    return tp / max((m | v).sum(), 1), tp / max(m.sum(), 1), tp / max(v.sum(), 1)


def main(n=30):
    from benchmark.nfo_vos.synth import gpjatk_foliage as GF
    pred = R.image_predictor()
    net = Fusion(token=True).cuda(); net.init_fallback(pred.model); net.load_state_dict(torch.load(CKPT)); net.eval()
    vp = build_predictor('b0')
    GF.set_bg_pool(FD.BG_SPLIT['val'])
    S = FD.specs('val')[:int(n)]
    os.makedirs(OUT, exist_ok=True)
    res, keep = [], []
    for k, spec in enumerate(S):
        raw = FD.clip_raw(spec, np.random.default_rng([1, k]))
        M, ts, empty_w = clip_arms(pred, net, vp, raw)
        pf = {a: [jpr(M[a][t], raw['V'][t]) for t in ts] for a in ARMS}
        row = {a: np.mean(pf[a], 0).tolist() for a in ARMS}
        # review gate: per-frame oracle selector between the memory-free and the memory arm (heuristic bound for A_r)
        row['max(pf,write)'] = [float(np.mean([max(x[0], y[0]) for x, y in zip(pf['adapter-pf'], pf['vid-write-pf'])]))] * 3
        row['max(pf,step0)'] = [float(np.mean([max(x[0], y[0]) for x, y in zip(pf['adapter-pf'], pf['step0-post'])]))] * 3
        row.update(cid=raw['cid'], n=len(ts), rho=float(raw['occ'].get('rho', np.nan)), empty_w=empty_w)
        res.append(row)
        print(k, raw['cid'], ' '.join(f'{a} {row[a][0]:.3f}' for a in ARMS), flush=True)
        if len(keep) < 4:
            t = ts[len(ts) // 2]
            keep.append((raw['X'][t], raw['V'][t], {a: M[a][t] for a in ARMS}, f'{raw["cid"]} t={t}'))
    json.dump(res, open(f'{OUT}/results.json', 'w'))
    report(res)
    montage(keep)


def report(res, B=2000):
    rng = np.random.default_rng(0)
    print(f'\n{len(res)} synthetic val clips, exact visible-mask J / P / R (clip means)')
    for a in ARMS:
        v = np.array([r[a] for r in res])
        print(f'  {a:16s} J {v[:, 0].mean():.3f}  P {v[:, 1].mean():.3f}  R {v[:, 2].mean():.3f}')
    print(f"  empty A_w masks: {np.mean([r['empty_w'] for r in res]):.1%} of written frames")
    for a, b in (('vid-oracle-V', 'adapter-pf'), ('vid-write-pf', 'adapter-pf'), ('vid-write-pf', 'vid-prompts'),
                 ('step0-post', 'adapter-pf'), ('step0-post', 'vid-write-pf'), ('max(pf,write)', 'adapter-pf'),
                 ('max(pf,step0)', 'adapter-pf'), ('adapter-pf', 'trk')):
        d = np.array([r[a][0] - r[b][0] for r in res])
        bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
        print(f'  ΔJ {a} − {b}: {d.mean():+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}]')


def montage(keep):
    rows = []
    for X, V, ms, title in keep:
        panes = []
        for name, m in [('exact V', V)] + list(ms.items()):
            v = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
            v[m] = (0.5 * v[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            cs, _ = cv2.findContours(V.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, cs, -1, (0, 255, 0), 1)
            cv2.putText(v, name if name != 'exact V' else title, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            panes.append(cv2.resize(v, (320, 180)))
        rows.append(np.hstack(panes))
    os.makedirs(os.path.dirname(IMG), exist_ok=True)
    cv2.imwrite(IMG, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main(*sys.argv[1:])
