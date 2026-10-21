"""NFO hand frames as adapter inputs, and the evaluation of a checkpoint on them (used during training).
Splits: 'test' = seq2–4 hand labels (handAv: 1 person, 2 void ignored), touched once at the end; 'select' = seq1 brush
masks, used for checkpoint selection. Inputs exactly as at test time: composite of k-6..k-1 aligned by the x-only
Theil–Sen fit of the kernel-f16 track (adapter/tracker.py), its evidence mass a, warm-up median bg.
"""
import os

import numpy as np
import torch

AMP = torch.float16 if os.environ.get('AMP') == 'fp16' else torch.bfloat16   # fp16 on pre-Ampere GPUs

import json

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.fusion_data import AMP, a256, u8
from benchmark.nfo_vos.adapter.composite import composite_and_mass
from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
from utils.stats_utils import theil_at

def hand_windows(split):
    """[(seq, k, trial, j)]: each hand frame k of the split with the first admissible window holding it at local j ≥ 6
    (six frames of history for the composite). 'select': seq1; 'test': seq2–4."""
    T = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    out = []
    for seq in (('seq1',) if split == 'select' else ('seq2', 'seq3', 'seq4')):
        for k in TR.hand_frames(seq):
            win = [(t, t['frames'].index(k)) for t in T if t['seq'] == seq and k in t['frames']]
            win = [(t, j) for t, j in win if j >= 6]
            if win:
                out.append((seq, k, *win[0]))
    return out


def items(split='test', pred=None):
    """One clip-like dict per hand frame (the format fusion_data.encode consumes), plus 'H' / 'keep' and 'box_trk'.
    split 'test': seq2–4 handAv (modal, void ignored), touched once at the end (remedy 2, 2026-10-16).
    split 'select': seq1 brush masks (handmask_raw, visible-leaning, no void), used for checkpoint selection.
    Alignment (2026-10-16 v3, train/test parity with fusion_data.build): kernel-f16 track of the window
    (tracker.kernel16_track) from the t0 prompt box; x-only Theil–Sen over [j-12, j]; composite of j-6..j-1 with the
    test parameters (τ = composite.TAU_AGE, z0 = adaptive). box_trk: the tracker's box at j (GT-free control, 'trk-box' arm)."""
    from benchmark.nfo_vos.adapter.tracker import kernel16_track
    out, trk = [], {}
    for seq, k, t, j in hand_windows(split):
        if t['id'] not in trk:
            w0, w1 = t['warmup']
            kc, wh = kernel16_track(t, pred)
            trk[t['id']] = (kc, wh, np.median(np.stack([TR.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32))
        kc, wh, bg = trk[t['id']]
        cs = list(range(max(0, j - 12), j + 1))
        fx = theil_at(cs, [kc[l, 0] for l in cs], [j] + list(range(j - 6, j)))
        hist = [t['frames'][l] for l in range(j - 6, j)]
        Ms = [np.float32([[1, 0, fx[j] - fx[l]], [0, 1, 0]]) for l in range(j - 6, j)]
        C, a = composite_and_mass([TR.native(i, seq).astype(np.float32) for i in hist], Ms, bg, z0='adaptive')
        H, keep = TR.hand_label(seq, k)
        v = H.astype(np.uint8) if keep is None else np.where(keep, H, 2).astype(np.uint8)   # 1 person, 2 void
        x0, y0, x1, y1 = gt_to_native(TR.bbs(seq)[k][0], 800, 600)
        s1 = np.array([1024 / 800, 1024 / 600] * 2, np.float32)
        box = np.array([x0, y0, x1, y1], np.float32) * s1
        bt = np.array([kc[j, 0] - wh[0] / 2, kc[j, 1] - wh[1] / 2, kc[j, 0] + wh[0] / 2, kc[j, 1] + wh[1] / 2], np.float32) * s1
        out.append(dict(X=TR.native(k, seq)[None], Ct=u8(C)[None], a=a256(a)[None], V=a256(v == 1)[None],
                        Vn=[v == 1], box=box[None], box_trk=bt[None], trk=(kc[j] * s1[:2])[None].astype(np.float32), mode=np.zeros(1, int), bg=u8(bg), cid=f'{seq}_{k:05d}',
                        occ=dict(rho=-1.0, clump='nfo', flutter=0.0), H=v == 1, keep=v != 2))
    return out


@torch.no_grad()
def evaluate(pred, net, its):
    """Frame-mean J / P / R vs the hand label (void ignored). Arms:
      box          SAM2, GT box (oracle prompt)          box+pt   GT box + its centre
      trk-box      SAM2, the kernel tracker's box (the GT-free control the token-mode adapter must beat)
      adapter      the adapter (its own prompt)          adapter-noC  sanity: composite removed (C := B, a := 0);
                                                         if it scores like 'adapter', the composite is not used
    Also the input statistics (median evidence mass Σa, mean a on the labelled person) for train/test parity."""
    from benchmark.nfo_vos.adapter.fusion_adapter import decode
    from benchmark.nfo_vos.adapter.fusion_data import encode
    net.eval()
    arms = ('box', 'box+pt', 'trk-box', 'adapter', 'adapter-noC')
    res, stats = {a: [] for a in arms}, []
    for it in its:
        f = encode(pred, [it])
        fb = dict(f, box=torch.as_tensor(it['box_trk'], device='cuda').float())
        f0 = dict(f, C16=f['B16'], C8=f['B8'], C4=f['B4'], a=torch.zeros_like(f['a']))
        with torch.autocast('cuda', dtype=AMP):
            o, o0 = net(f, sam=pred.model), net(f0, sam=pred.model)
            outs = {'box': decode(pred.model, f), 'box+pt': decode(pred.model, f, point=True),
                    'trk-box': decode(pred.model, fb), 'adapter': decode(pred.model, o, token=o.get('token')),
                    'adapter-noC': decode(pred.model, o0, token=o0.get('token'))}
        H, keep = torch.from_numpy(it['H']).cuda(), torch.from_numpy(it['keep']).cuda()
        for a, low in outs.items():
            pr = torch.nn.functional.interpolate(low.float(), H.shape, mode='bilinear', align_corners=False)[0, 0] > 0
            tp = (pr & H & keep).sum().item()
            res[a].append((tp / max(((pr | H) & keep).sum().item(), 1), tp / max((pr & keep).sum().item(), 1),
                           tp / max((H & keep).sum().item(), 1)))
        a_np = it['a'][0]; on = a256(it['H'])[...] > 0.5
        stats.append((float(a_np.sum()), float(a_np[on].mean()) if on.any() else np.nan))
    net.train()
    out = {a: dict(zip('JPR', np.mean(v, 0))) for a, v in res.items()}
    out['per_frame'] = {a: [r[0] for r in v] for a, v in res.items()}
    out['inputs'] = dict(a_mass=float(np.median([s_[0] for s_ in stats])), a_on=float(np.nanmean([s_[1] for s_ in stats])))
    return out


