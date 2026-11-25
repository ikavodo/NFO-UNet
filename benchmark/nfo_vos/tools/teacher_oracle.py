"""Which teacher should the one-pass student imitate? (findings §10.x item 20)

Synthetic val (30 clips, 16 targets each), where T16 = SAM2 features of the occluder-free frame exists. Arms (centred
cosine to T16 at stride 16, on person cells (S share ≥ 0.5) and hidden-person cells (S ≥ 0.5, V < 0.1)):
  feat-α     Ĉ = Σ α_k K_k + α_∅ B: the stage-0c learned weights, mixed in FEATURE space (current feature route)
  pixel-α    E(Σ α_k↑ A_k + α_∅↑ bg): the same weights, upsampled bilinearly, mixed in PIXEL space, then encoded
  pixel-hand E(C): the deployed hand-set rule (recency τ × robust-z change gate), = f['P16'] (the current teacher)
  pixel-orc  E(C*): the same formula and recency weights as pixel-hand, gate replaced by the GT visibility V_k
             (person pixel unoccluded in history frame k, warped onto t): the best pixel gate, upper bound of any rule
  floors     B (empty scene), F_t (current frame)
Comparisons (paired per clip, clip bootstrap): pixel-α − feat-α (does the mixing location matter, same weights?),
pixel-α − pixel-hand (does the learned choice beat the hand rule?), pixel-orc − pixel-hand (headroom of a better gate).
stage-0c α: trained only to reconstruct T on synthetic, never joint-trained with an adapter.

    PYTHONPATH=. python -m benchmark.nfo_vos.tools.teacher_oracle
"""
import json

import cv2
import numpy as np
import torch

from benchmark.nfo_vos.adapter import fusion_data as FD
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.feat_integral import ccos
from benchmark.nfo_vos.eval.dump_arms import load_net
from benchmark.nfo_vos.tools.feat_probe import s16
from benchmark.nfo_vos.tools.feat_probe_nfo import NETS, STAGE0C

OUT = 'results/benchmark/rr/feat_integral'
ARMS = ('B', 'F_t', 'feat-α', 'pixel-hand', 'pixel-α', 'pixel-orc')


def warp(x, M):
    return cv2.warpAffine(x.astype(np.float32), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@torch.no_grad()
def main():
    pred = image_predictor()
    agg = load_net(pred, NETS['learned']).agg.eval(); agg.load_compat(torch.load(STAGE0C, map_location='cuda'))
    mu = agg.mu16
    dl = torch.utils.data.DataLoader(FD.Clips('val', n_val=30), batch_size=None, num_workers=6, collate_fn=_ident)
    per = []                                                              # per clip: arm -> (person cos, hidden cos)
    for k, raw in enumerate(dl):
        if k == 30:
            break
        f, meta = FD.build(pred, raw, max_targets=16, rng=np.random.default_rng(k), feat=True, with_pixel=True)
        X, bg, occ = raw['X'], raw['bg'].astype(np.float32), meta['occ']
        H, W = bg.shape
        acc = {a: [[], []] for a in ARMS}
        for i, t in enumerate(meta['ts']):
            if meta['mode'][i] == 2:
                continue
            S, V = s16(raw['S'][t]) >= 0.5, s16(raw['V'][t]) < 0.1
            if S.sum() < 4:
                continue
            hist, Ms = FD.hist_warps(meta['trk'], t, occ['nh'])
            assert len(hist) == f['K16'].shape[1]
            v = f['Kvalid16'][i:i + 1, :, 0].bool()
            with torch.autocast('cuda', dtype=FD.AMP):
                lk, ln = agg.logits_learned(f['K16'][i:i + 1].float(), f['X16'][i:i + 1], f['B16'][i:i + 1], f['age'][i:i + 1],
                                            phase=f['phase'][i:i + 1])
            al = torch.softmax(torch.cat([lk.float().masked_fill(~v, float('-inf')), ln.float()], 1), 1)      # [1, nh+1, 64, 64]
            chat = (al[:, :-1, None] * f['K16'][i:i + 1].float()).sum(1) + al[:, -1:] * f['B16'][i:i + 1].float()
            A = np.stack([warp(X[q], M) for q, M in zip(hist, Ms)])
            au = torch.nn.functional.interpolate(al, size=(H, W), mode='bilinear', align_corners=False)[0].cpu().numpy()
            C_al = (au[:-1] * A).sum(0) + au[-1] * bg
            w = np.exp(-np.arange(len(hist) - 1, -1, -1) / occ['tau']); w = (w / w.sum())[:, None, None]
            G = np.stack([warp(raw['V'][q].astype(np.float32), M) for q, M in zip(hist, Ms)])
            mass = (w * G).sum(0)
            C_or = np.where(mass > 1e-6, (w * G * A).sum(0) / np.maximum(mass, 1e-6), (w * A).sum(0))
            with torch.autocast('cuda', dtype=FD.AMP):
                E16 = FD._enc(pred, np.stack([FD.u8(C_al), FD.u8(C_or)]))[0].float()
            src = {'B': f['B16'][i], 'F_t': f['X16'][i], 'feat-α': chat[0], 'pixel-hand': f['P16'][i], 'pixel-α': E16[0],
                   'pixel-orc': E16[1]}
            m = torch.from_numpy(S).cuda(); h = torch.from_numpy(S & V).cuda()
            for a, x in src.items():
                c = ccos(x.float()[None], f['T16'][i:i + 1].float(), mu)[0]
                acc[a][0].append(c[m].mean().item())
                if h.sum() > 0:
                    acc[a][1].append(c[h].mean().item())
        if acc['B'][0]:
            per.append({a: (np.mean(v[0]), np.mean(v[1]) if v[1] else np.nan) for a, v in acc.items()})
            print(f'clip {k}: ' + ' '.join(f'{a} {per[-1][a][0]:.3f}' for a in ARMS), flush=True)
        del f
    rng = np.random.default_rng(0); res = {}
    for j, nm in enumerate(('person', 'hidden')):
        print(f'== {nm} cells, mean centred cos to T16 over {len(per)} clips: '
              + '  '.join(f'{a} {np.nanmean([p[a][j] for p in per]):.3f}' for a in ARMS), flush=True)
        for a, b in (('pixel-α', 'feat-α'), ('pixel-α', 'pixel-hand'), ('pixel-orc', 'pixel-hand'), ('pixel-hand', 'feat-α')):
            d = np.array([p[a][j] - p[b][j] for p in per]); d = d[~np.isnan(d)]
            ci = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
            res[f'{nm}_{a}-{b}'] = [float(d.mean()), *map(float, ci)]
            print(f'   {a} − {b}: {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]', flush=True)
    json.dump(res, open(f'{OUT}/teacher_oracle.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
