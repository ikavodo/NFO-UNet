"""Real-data alignment check for the feature integral: does tracker alignment of cached SAM2 features hold on NFO as well as
on the synthetic training clips? No clean frames are needed: on person cells VISIBLE at t (hand label / synthetic V_t),
F_t itself is the truth, so

  c_k  = cos(shift(F_{t−k}), F_t)   per history age k = 1..6 (s16, centred)   — alignment + articulation + history occlusion
  c0_k = the same without the shift (dx = 0)                                  — what the tracker shift buys
  cC   = cos(Ĉ, F_t) for the learned aggregator (checkpoint given)            — the integral on visible cells

NFO: seq1 hand frames only (the selection split; seq2–4 untouched), kernel-f16 track and x-only Theil–Sen as in
fusion_nfo.items. Synthetic: 30 val clips via fusion_data.build(feat=True). A clearly lower NFO c_k than synthetic at the
same age means a real-data alignment gap (then: neighborhood / deformable attention, design §4).

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.align_consistency [agg.pt]
"""
import json
import sys

import numpy as np
import torch

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.feat_integral import STRIDES, FeatIntegral, ccos, shift_keys
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips, _cover, _enc, build, u8
from benchmark.nfo_vos.adapter.fusion_nfo import hand_windows
from utils.stats_utils import theil_at

OUT = 'results/benchmark/rr/feat_integral/align_consistency.json'
NH = 6


def _summ(rows):
    a = np.array([r['c'] for r in rows]); a0 = np.array([r['c0'] for r in rows]) if 'c0' in rows[0] else None
    out = {'n': len(rows), 'c_by_age': np.nanmean(a, 0).round(3).tolist(), 'cC': float(np.nanmean([r['cC'] for r in rows]))}
    if a0 is not None:
        out['c0_by_age'] = np.nanmean(a0, 0).round(3).tolist()
    return out


@torch.no_grad()
def nfo(pred, agg):
    from benchmark.nfo_vos.adapter.tracker import kernel16_track
    mu = pred.model.no_mem_embed.detach().float().reshape(1, -1, 1, 1)
    rows, trk = [], {}
    for seq, k, t, j in hand_windows('select'):
        if t['id'] not in trk:
            w0, w1 = t['warmup']
            kc, wh = kernel16_track(t, pred)
            bg = np.median(np.stack([TR.native(i, seq) for i in range(w0, w1 + 1)]), 0)
            trk[t['id']] = (kc, u8(bg))
        kc, bg = trk[t['id']]
        cs = list(range(max(0, j - 12), j + 1))
        fx = theil_at(cs, [kc[l, 0] for l in cs], [j] + list(range(j - NH, j)))
        frames = [TR.native(t['frames'][l], seq) for l in range(j - NH, j)] + [TR.native(k, seq), bg]
        with torch.autocast('cuda', dtype=AMP):
            F16, F8, F4 = _enc(pred, np.stack(frames))
        Fs = {'16': F16.float(), '8': F8.float(), '4': F4.float()}
        dx = torch.tensor([[(fx[j] - fx[l]) * 1024 / 800 for l in range(j - NH, j)]], device='cuda')
        K, valid = {}, None
        for s in STRIDES:
            K[s], v = shift_keys(Fs[s][None, :NH], dx, int(s))
            valid = v if s == '16' else valid
        V = torch.as_tensor(_cover(TR.hand_label(seq, k)[0], 64), device='cuda')
        Ft = Fs['16'][NH:NH + 1]
        c = [float(ccos(K['16'][0, i], Ft[0], mu[0])[V & valid[0, i, 0]].mean()) for i in range(NH)][::-1]   # age 1..6
        c0 = [float(ccos(Fs['16'][i], Ft[0], mu[0])[V].mean()) for i in range(NH)][::-1]
        with torch.autocast('cuda', dtype=AMP):
            o = agg(K, valid, {s: Fs[s][NH:NH + 1] for s in STRIDES}, {s: Fs[s][NH + 1:] for s in STRIDES},
                    torch.arange(NH, 0, -1, device='cuda')[None], phase=dx / 16 - torch.round(dx / 16))
        cC = float(ccos(o['C16'].float(), Ft, mu)[0][V].mean())
        rows.append(dict(seq=seq, k=k, c=c, c0=c0, cC=cC, px_per_frame=float(abs(fx[j] - fx[j - 1]))))
    return rows


@torch.no_grad()
def synthetic(pred, agg, n_val=30):
    val = list(torch.utils.data.DataLoader(Clips('val', n_val=n_val), batch_size=None, num_workers=6, collate_fn=_ident))
    rows = []
    for i, raw in enumerate(val):
        raw['occ']['nh'] = NH
        f, meta = build(pred, raw, max_targets=8, rng=np.random.default_rng(i), feat=True)
        mu = f['mu16']
        with torch.autocast('cuda', dtype=AMP):
            o = agg({s: f[f'K{s}'] for s in STRIDES}, f['Kvalid16'], {s: f[f'X{s}'] for s in STRIDES},
                    {s: f[f'B{s}'] for s in STRIDES}, f['age'], phase=f['phase'])
        for n in range(len(meta['ts'])):
            V = f['Vis16'][n, 0]
            if V.sum() < 3 or not f['Kvalid16'][n].any():
                continue
            c = [float(ccos(f['K16'][n, q].float(), f['X16'][n], mu[0])[V & f['Kvalid16'][n, q, 0]].mean()) for q in range(NH)][::-1]
            rows.append(dict(c=c, cC=float(ccos(o['C16'][n:n + 1].float(), f['X16'][n:n + 1], mu)[0][V].mean())))
    return rows


def main(ckpt='results/benchmark/rr/feat_integral/stage0c_seed0.pt'):
    pred = image_predictor()
    agg = FeatIntegral(pred.model.no_mem_embed).cuda().eval()
    agg.load_compat(torch.load(ckpt, map_location='cuda'))
    rn, rs = nfo(pred, agg), synthetic(pred, agg)
    res = {'ckpt': ckpt, 'nfo_seq1': _summ(rn), 'synthetic': _summ(rs),
           'nfo_px_per_frame_median': float(np.median([r['px_per_frame'] for r in rn]))}
    print(f"cos(shifted F_t−k, F_t) on visible person cells, s16, by age k = 1..{NH}")
    print(f"  NFO seq1   (n={res['nfo_seq1']['n']:3d}) shifted   {res['nfo_seq1']['c_by_age']}")
    print(f"  NFO seq1               unshifted {res['nfo_seq1']['c0_by_age']}")
    print(f"  synthetic  (n={res['synthetic']['n']:3d}) shifted   {res['synthetic']['c_by_age']}")
    print(f"cos(Ĉ, F_t) on visible cells: NFO {res['nfo_seq1']['cC']:.3f} | synthetic {res['synthetic']['cC']:.3f}")
    print(f"NFO tracker speed, median |Δfx| per frame: {res['nfo_px_per_frame_median']:.1f} native px")
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
