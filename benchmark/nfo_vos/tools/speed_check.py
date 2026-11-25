"""Speed levers for the one-pass (feature-integral) path, findings §10.x 25. Run on an idle GPU.

(A) Tracker: kernel_track (CPU: scipy Theil–Sen + per-row numpy loop over a native-res integral image) vs a GPU port with the
    same arithmetic (cumsum integral image in float64, all (cy, x) candidates at once, Theil–Sen as median pairwise slope).
    Checks identical tracks, times ms/frame over seq1 windows.
(B) Encoder: SAM2.1 b+ image encoder on the full 1024² input vs a 512² crop of the same input centred on the tracker (same
    scale; Hiera interpolates its positional embedding), eager vs torch.compile. Validity of the crop: 1 − cos between the
    crop's features and the full-frame features of the same cells (s16 / s8 / s4, on the 32×32 crop cells), against the
    natural scale 1 − cos(full_t, full_t+1) on the same cells.

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.speed_check
"""
import json
import time

import numpy as np
import torch

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter.fusion_adapter import AMP, image_predictor
from benchmark.nfo_vos.adapter.tracker import kernel_track
from benchmark.nfo_vos.tools.feat_distil import frames


def ts_fit(P):
    """scipy.stats.theilslopes on (arange(n), P[:, i]): slope = median pairwise slope, intercept = median(y) − slope·median(x)."""
    n = len(P); x = np.arange(n, dtype=float)
    i, j = np.triu_indices(n, 1)
    s = np.median((P[j] - P[i]) / (x[j] - x[i])[:, None], 0)
    return s * n + (np.median(P, 0) - s * np.median(x))


def kernel_track_gpu(maps, c0, w0, h0):
    """kernel_track with the search on the GPU: maps [T, H, W] cuda tensor."""
    T, H, W = maps.shape
    I = torch.zeros(T, H + 1, W + 1, device=maps.device, dtype=torch.float64)
    I[:, 1:, 1:] = maps.double().cumsum(1).cumsum(2)                     # = cv2.integral per frame
    out = [np.array(c0, float)]
    hw, hh = int(round(w0 / 2)), int(round(h0 / 2))
    for t in range(1, T):
        P = np.array(out[-8:])
        if len(P) >= 3:
            pred = ts_fit(P); v = np.abs(pred - P[-1])
        else:
            pred, v = P[-1], np.zeros(2)
        rx, ry = int(max(0.5 * w0, 3 * v[0] + 20)), int(max(0.25 * h0, 3 * v[1] + 10))
        cy = torch.arange(int(pred[1]) - ry, int(pred[1]) + ry + 1, 2, device=maps.device)
        xs = torch.arange(int(pred[0]) - rx, int(pred[0]) + rx + 1, 2, device=maps.device)
        y0, y1 = (cy - hh).clamp(0, H), (cy + hh).clamp(0, H)
        x0, x1 = (xs - hw).clamp(0, W), (xs + hw).clamp(0, W)
        It = I[t]
        s = (It[y1[:, None], x1] - It[y0[:, None], x1] - It[y1[:, None], x0] + It[y0[:, None], x0])
        s = s / ((x1 - x0)[None] * (y1 - y0)[:, None]).clamp(min=1)
        s[(y1 <= y0)] = -float('inf')
        k = int(torch.argmax(s))                                          # first max in row-major order = the CPU loop's choice
        out.append(np.array([float(xs[k % len(xs)]), float(cy[k // len(xs)])]))
    return np.array(out)


def main():
    pred = image_predictor(); sam = pred.model
    tr = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1'][:4]
    tx = lambda img: pred._transforms.transforms(torch.as_tensor(img, device='cuda').float().div(255)[None].expand(3, -1, -1))[None]

    def fpn(x):
        o = sam.forward_image(x)['backbone_fpn']                         # [s4, s8, s16]
        return {'4': o[0][0].float(), '8': o[1][0].float(), '16': o[2][0].float()}

    def ms(fn, n=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t0) / n

    # (A) tracker
    ta, tg, dmax, ex = [], [], 0.0, None
    for t in tr:
        Xn, bg, g = frames(t)
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
            B = fpn(tx(bg))['16']
            m = []
            for x in Xn:
                c = 1 - torch.nn.functional.cosine_similarity(fpn(tx(x))['16'], B, dim=0)
                m.append(torch.nn.functional.interpolate(c[None, None], size=x.shape, mode='bilinear', align_corners=False)[0, 0].float())
        M = torch.stack(m); c0, w0, h0 = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), g[2] - g[0], g[3] - g[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        kc = kernel_track(M.cpu().numpy(), c0, w0, h0)                 # as inference.py: includes the device→host copy
        ta.append((time.perf_counter() - t0) / len(Xn))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        kg = kernel_track_gpu(M, c0, w0, h0); torch.cuda.synchronize()
        tg.append((time.perf_counter() - t0) / len(Xn))
        dmax = max(dmax, float(np.abs(kc - kg).max()))
        ex = ex or (Xn, kc)
    print(f'(A) tracker ms/frame: CPU {1000 * np.mean(ta):.2f}, GPU {1000 * np.mean(tg):.2f}; max track difference {dmax:.2f} px '
          f'({len(tr)} seq1 windows)', flush=True)

    # (B) encoder: full vs 512² crop around the tracker
    Xn, kc = ex; H, W = Xn[0].shape
    par = {l: [] for l in ('16', '8', '4')}; nat = {l: [] for l in ('16', '8', '4')}
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
        prev = None
        for j in range(0, len(Xn), 2):
            x = tx(Xn[j]); full = fpn(x)
            cx, cy = kc[j][0] * 1024 / W, kc[j][1] * 1024 / H
            x0 = int(np.clip(round((cx - 256) / 16), 0, 32)) * 16; y0 = int(np.clip(round((cy - 256) / 16), 0, 32)) * 16
            crop = fpn(x[:, :, y0:y0 + 512, x0:x0 + 512].contiguous())
            for l, s in (('16', 16), ('8', 8), ('4', 4)):
                ref = full[l][:, y0 // s:(y0 + 512) // s, x0 // s:(x0 + 512) // s]
                par[l].append(1 - torch.nn.functional.cosine_similarity(crop[l], ref, dim=0).median().item())
                if prev is not None:
                    nat[l].append(1 - torch.nn.functional.cosine_similarity(prev[l][:, y0 // s:(y0 + 512) // s, x0 // s:(x0 + 512) // s],
                                                                            ref, dim=0).median().item())
            prev = full
    print('(B) crop validity, median 1 − cos on the crop cells: ' + ', '.join(
        f's{l} crop-vs-full {np.mean(par[l]):.3f} / frame-to-frame (2 apart) {np.mean(nat[l]):.3f}' for l in par), flush=True)
    xf = tx(Xn[0]); xc = xf[:, :, :512, :512].contiguous()
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
        r = {'full eager': ms(lambda: sam.forward_image(xf)), 'crop eager': ms(lambda: sam.forward_image(xc))}
        enc = sam.image_encoder
        try:
            sam.image_encoder = torch.compile(enc)
            r['full compiled'] = ms(lambda: sam.forward_image(xf)); r['crop compiled'] = ms(lambda: sam.forward_image(xc))
        except Exception as e:                                            # report, don't hide
            print(f'compile failed: {type(e).__name__}: {e}'[:300], flush=True)
        sam.image_encoder = enc
    print('(B) encoder ms/frame: ' + ', '.join(f'{k} {v:.1f}' for k, v in r.items()), flush=True)


if __name__ == '__main__':
    main()
