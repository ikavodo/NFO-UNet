"""Per-frame adapter inference on NFO windows (the benchmark path; masks dumped by eval/dump_arms.py)."""
import time

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter import fusion_data as FD
from benchmark.nfo_vos.adapter.fusion_adapter import decode
from benchmark.nfo_vos.adapter.tracker import gt_box, kernel_track
from utils.stats_utils import theil_at

S1 = np.array([1024 / 800, 1024 / 600] * 2, np.float32)    # native xyxy -> SAM2's 1024² input


def _gate_gpu(X, B):
    """integral_image.integral_image_and_mass's gate, z0 = 'adaptive', on the GPU: robust z of |X − B| (median, 1.4826·MAD),
    threshold max(3, q90(z)). Depends only on the frame and the empty scene, so it is computed once per frame."""
    d = (X - B).abs()
    med = d.median()
    z = (d - med) / (1.4826 * (d - med).abs().median()).clamp(min=1e-6)
    return torch.sigmoid(z - torch.quantile(z.flatten(), 0.9).clamp(min=3.0))


def box_signals(G, mass, box):
    """Change-gate and evidence statistics inside the tracker box (gate features, eval/dual_expert.py): mean gate,
    connected components of the gated map (≥ 4 px) and the largest one's share of the gated area, mean evidence mass."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    Hh, Ww = G.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Ww, x1), min(Hh, y1)
    if x1 <= x0 or y1 <= y0:
        return dict(gate_box=0.0, n_comp=0, largest=0.0, mass_box=0.0)
    g = (G[y0:y1, x0:x1] > 0.5).cpu().numpy().astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA]; areas = areas[areas >= 4]
    return dict(gate_box=float(g.mean()), n_comp=int(len(areas)), largest=float(areas.max() / areas.sum()) if len(areas) else 0.0,
                mass_box=float(mass[y0:y1, x0:x1].mean()) if mass is not None else 0.0)


def _shift_x(x, dx):
    """[K, H, W] shifted right by dx[k] native px: bilinear, border replicate (= cv2.warpAffine [[1, 0, dx], [0, 1, 0]])."""
    K, H, W = x.shape
    xs = torch.arange(W, device=x.device, dtype=x.dtype)[None] - dx[:, None]               # source column per output column
    gx = (2 * xs / (W - 1) - 1)[:, None, :].expand(K, H, W)
    gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)[None, :, None].expand(K, H, W)
    return torch.nn.functional.grid_sample(x[:, None], torch.stack([gx, gy], -1), mode='bilinear', padding_mode='border',
                                           align_corners=True)[:, 0]


@torch.no_grad()
def fast_masks(pred, net, t, tau=3.5, noC=False, hist=6):
    """Adapter masks of one NFO window. Each frame is encoded ONCE (shared by tracker, adapter and decoder), the empty
    scene once per window, each frame's gate once, the integral image on the GPU (61 ms/frame vs 445 ms for the original
    path that re-encoded frame / integral image / empty scene per frame; same box AUC and hand J, 2026-10-20).
    net None: the 'trk' arm (per-frame SAM2, tracker box + point, every frame). noC: the integral image removed
    (C := empty scene, a := 0). hist: past frames in the integral image (training drew 4..8). Frames j < hist have no history
    and use the tracker prompts alone.
    Returns ([T, H, W] bool, s / frame)."""
    seq, fr = t['seq'], t['frames']
    w0, w1 = t['warmup']
    bg = np.median(np.stack([TR.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    return masks_from_arrays(pred, net, [TR.native(k, seq) for k in fr], bg, gt_box(seq, fr[0]), tau, noC, hist)


def masks_from_arrays(pred, net, Xn, bg, g, tau=3.5, noC=False, hist=6, feats=None, aux=None):
    """fast_masks on arrays: Xn = list of [H, W] uint8 gray frames, bg = empty scene [H, W], g = t0 box (x0, y0, x1, y1) px.
    feats: per-frame (f16, f8, f4) from a caller that already ran the image encoder on the same 1024² input (shares the
    pass, eval/dual_expert.py); None encodes here. aux: a list that receives one dict of per-frame signals (decoder IoU
    head, object-score logit, evidence and change-gate statistics inside the tracker box). Frame I/O is not timed
    (it was before 2026-11-11, when fast_masks read the frames inside the timed section)."""
    torch.cuda.synchronize(); t0 = time.perf_counter()   # ponytail: empty scene = once-per-sequence setup, not timed per frame
    sam = pred.model
    H, W = Xn[0].shape
    S1 = np.array([1024 / W, 1024 / H] * 2, np.float32)  # native xyxy -> SAM2's 1024² input

    def enc(img):
        """pred.set_image on a uint8 gray image, with SAM2's transforms (to-tensor, 1024² resize, normalise) run on the
        GPU instead of the CPU (sam2_image_predictor.py:110-127)."""
        x = torch.as_tensor(img, device='cuda').to(torch.uint8).float().div(255)[None].expand(3, -1, -1)
        _, vf, _, _ = sam._prepare_backbone_features(sam.forward_image(pred._transforms.transforms(x)[None]))
        if sam.directly_add_no_mem_embed:
            vf[-1] = vf[-1] + sam.no_mem_embed
        f4, f8, f16 = [v.permute(1, 2, 0).view(1, -1, *s_) for v, s_ in zip(vf, pred._bb_feat_sizes)]
        return f16.float(), f8.float(), f4.float()

    with torch.autocast('cuda', dtype=FD.AMP):
        Bf = enc(bg)
        Bg = torch.as_tensor(bg, device='cuda')
        F, G, m16 = [], [], []
        for i, x in enumerate(Xn):                                       # one encoder pass per frame
            F.append(enc(x) if feats is None else feats[i])
            m = 1 - torch.nn.functional.cosine_similarity(F[-1][0][0], Bf[0][0], dim=0)
            m16.append(torch.nn.functional.interpolate(m[None, None], size=x.shape, mode='bilinear', align_corners=False)[0, 0])
            G.append(_gate_gpu(torch.as_tensor(x, device='cuda', dtype=torch.float32), Bg))
    kc = kernel_track(torch.stack(m16).cpu().numpy(), ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), g[2] - g[0], g[3] - g[1])
    wh = (g[2] - g[0], g[3] - g[1])
    Xg = torch.as_tensor(np.stack(Xn), device='cuda', dtype=torch.float32)
    Gg = torch.stack(G)
    wk = torch.as_tensor(np.exp(-np.arange(hist - 1, -1, -1) / tau), device='cuda', dtype=torch.float32)
    wk = (wk / wk.sum())[:, None, None]
    out = np.zeros((len(Xn), H, W), bool)
    for j in range(len(Xn)):
        box = np.array([kc[j, 0] - wh[0] / 2, kc[j, 1] - wh[1] / 2, kc[j, 0] + wh[0] / 2, kc[j, 1] + wh[1] / 2], np.float32) * S1
        pt = (kc[j] * S1[:2]).astype(np.float32)
        f = {f'X{l}': F[j][i] for i, l in enumerate(('16', '8', '4'))}
        f['box'] = f['box_trk'] = torch.as_tensor(box[None], device='cuda')
        f['pt'] = f['trk'] = torch.as_tensor(pt[None], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            da = [] if aux is not None else None
            mass = None
            if j < hist or net is None:                                     # no history yet / trk arm: tracker prompts only
                low = decode(pred.model, f, point=True, aux=da)
            else:
                cs = list(range(max(0, j - 12), j + 1))
                fx = theil_at(cs, [kc[l, 0] for l in cs], [j] + list(range(j - hist, j)))
                dx = torch.tensor([fx[j] - fx[l] for l in range(j - hist, j)], device='cuda', dtype=torch.float32)
                A, Gw = _shift_x(Xg[j - hist:j], dx), _shift_x(Gg[j - hist:j], dx)
                mass = (wk * Gw).sum(0)
                C = torch.where(mass > 1e-6, (wk * Gw * A).sum(0) / mass.clamp(min=1e-6), (wk * A).sum(0))
                if noC:
                    Cf, mass = Bf, torch.zeros_like(mass)
                else:
                    with torch.autocast('cuda', dtype=FD.AMP):
                        Cf = enc(C.clamp(0, 255))   # .to(uint8) in enc = FD.u8's truncation, as the benchmark path
                f.update({f'C{l}': Cf[i] for i, l in enumerate(('16', '8', '4'))})
                f.update({f'B{l}': Bf[i] for i, l in enumerate(('16', '8', '4'))})
                f['a'] = torch.nn.functional.interpolate(mass.clamp(0, 1)[None, None], size=(256, 256), mode='area')[0]
                o = net(f, sam=pred.model)
                low = decode(pred.model, o, token=o['token'], aux=da)
        out[j] = cv2.resize(low[0, 0].float().cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR) > 0
        if aux is not None:
            aux.append(dict(j=j, adapter=bool(j >= hist and net is not None), pred_iou=float(da[0][0]), obj=float(da[0][1]),
                            box=(box / S1).tolist(), **box_signals(Gg[j], mass, box / S1)))
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / len(Xn)
