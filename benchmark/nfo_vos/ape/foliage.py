"""Foliage pool for planted real-foliage occlusion (Stage 1, approach A, 2026-11-13; plan
docs/superpowers/plans/2026-11-13-ape-foliage-pool.md).

Sources: PanAf500 videos (any class) not in the eval subset and not camera-leaked (results/ape/leakage.csv). Per source:
  - window: the 40 consecutive frames whose union of GT ape boxes (dilated 15% per side) covers the least of the frame;
    that union is stored as `nocut` (the loader never cuts foliage there, so the occluder holds no ape);
  - per-frame near layer: Depth Anything V2 Small on all 40 frames (one batch on the 518x294 model grid), D rank-normalised
    per frame (PanAf20K panaf/depth_cues.py:35), alpha = sigmoid((D - 0.8)/0.03), upsampled to the frame, uint8.
Pixels are not stored: the loader re-decodes the source window (cheap at 720x404), so the same patch is cut from
consecutive frames and real sway + compression noise come with it.
Index columns: ape_cov (nocut share), near_cov (share alpha >= 0.5 outside nocut), frag (boundary px / area px of the near
layer, median frame; high = fragmented vegetation, low = compact ground), flicker (mean |alpha_t - alpha_t-1| on pixels near
in either frame; depth temporal stability), ir (mean Lab chroma < 4), cluster, split (90/10 by leakage cluster, seed 0).
    python -m benchmark.nfo_vos.ape.foliage   -> results/ape/foliage/<video>.npz, results/ape/foliage_index.csv,
                                                  images/ape/foliage_pool.jpg
"""
import json, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as Fn
from benchmark.nfo_vos.ape import PANAF, PANAF_ROOT, OUT

N, STRIDE, DIL = 40, 4, 0.15
IN = (518, 294)                                           # model input (w, h), as PanAf20K panaf/depth_cues.py:25
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
NEAR, TEMP = 0.8, 0.03


def near_alpha(D):
    """D: (B, h, w) depth (larger = nearer). Per-frame rank to [0, 1], then sigmoid((r - 0.8)/0.03)."""
    B = D.shape[0]; f = D.reshape(B, -1)
    r = torch.empty_like(f); r.scatter_(1, f.argsort(1), torch.linspace(0, 1, f.shape[1], device=D.device).expand(B, -1))
    return torch.sigmoid((r - NEAR) / TEMP).reshape(D.shape)


def best_window(boxes, n_frames, H, W):
    """boxes: {frame index (0-based): [[x1,y1,x2,y2], ...]}. Returns (start, nocut bool HxW) of the 40-frame window whose
    union of 15%-dilated boxes is smallest (first one on ties)."""
    s = 4                                                   # selection on a 4x coarser grid
    h, w = (H + s - 1) // s, (W + s - 1) // s

    def mask(frames, sc, hh, ww):
        m = np.zeros((hh, ww), bool)
        for k in frames:
            for x1, y1, x2, y2 in boxes.get(k, ()):
                dx, dy = DIL * (x2 - x1), DIL * (y2 - y1)
                m[max(int((y1 - dy) / sc), 0):int(np.ceil((y2 + dy) / sc)), max(int((x1 - dx) / sc), 0):int(np.ceil((x2 + dx) / sc))] = True
        return m
    starts = list(range(0, n_frames - N + 1, STRIDE))
    cov = [mask(range(t, t + N), s, h, w).mean() for t in starts]
    t = starts[int(np.argmin(cov))]
    return t, mask(range(t, t + N), 1, H, W)


def boundary(m):
    m = m.astype(np.uint8); return int((m != cv2.erode(m, np.ones((3, 3), np.uint8))).sum())


def sources():
    sub = set(pd.read_csv(PANAF / 'results/panaf/eval_subset.csv').video)
    L = pd.read_csv(OUT / 'leakage.csv').set_index('video')
    ann = {p.stem: p for p in PANAF_ROOT.glob('annotations/*/*.json')}
    vids = [v for v in sorted(ann) if v not in sub and v in L.index and not L.in_eval[v] and not L.leaked[v]
            and (PANAF_ROOT / 'videos' / f'{v}.mp4').exists()]
    return vids, ann, L


def read_window(path, start):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, start); out = []
    for _ in range(N):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main():
    from transformers import AutoModelForDepthEstimation
    m = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').cuda().eval().half()
    vids, ann, L = sources()
    cl = sorted(L.cluster[vids].unique()); rng = np.random.default_rng(0)
    val_cl = set(rng.choice(cl, max(1, round(0.1 * len(cl))), replace=False))
    (OUT / 'foliage').mkdir(parents=True, exist_ok=True); rows, tiles, t0 = [], [], time.time()
    print(f'{len(vids)} sources, {len(cl)} clusters ({len(val_cl)} val)', flush=True)
    for i, v in enumerate(vids):
        path = PANAF_ROOT / 'videos' / f'{v}.mp4'; cap = cv2.VideoCapture(str(path))
        nf, W, H = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(3)), int(cap.get(4)); cap.release()
        if nf < N:
            print(f'  skip {v}: {nf} frames', flush=True); continue
        boxes = {}
        for a in json.load(open(ann[v]))['annotations']:
            boxes.setdefault(a['frame_id'] - 1, []).extend(d['bbox'] for d in a['detections'])
        start, nocut = best_window(boxes, nf, H, W)
        fr = read_window(path, start)
        if len(fr) < N:
            print(f'  skip {v}: decoded {len(fr)}/{N}', flush=True); continue
        x = np.stack([cv2.resize(f, IN, interpolation=cv2.INTER_AREA)[..., ::-1] for f in fr]) / 255.
        x = torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2)).half().cuda()
        with torch.no_grad():
            a = near_alpha(m(pixel_values=x).predicted_depth.float())                     # (40, h, w) on the model grid
            a = Fn.interpolate(a[:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0]
        A = (a.clamp(0, 1) * 255).round().byte().cpu().numpy()
        np.savez_compressed(OUT / 'foliage' / f'{v}.npz', alpha=A, nocut=nocut, start=start)
        near = A >= 128; free = ~nocut
        either = near[1:] | near[:-1]
        flick = float(np.abs(np.diff(A.astype(np.int16), axis=0))[either].mean() / 255) if either.any() else 0.0
        med = near[N // 2] & free
        lab = cv2.cvtColor(fr[N // 2], cv2.COLOR_BGR2LAB).astype(np.int16)
        chroma = float((np.abs(lab[..., 1] - 128) + np.abs(lab[..., 2] - 128)).mean())
        c = int(L.cluster[v])
        rows.append(dict(video=v, start=start, H=H, W=W, ape_cov=round(float(nocut.mean()), 4),
                         near_cov=round(float(near[:, free].mean()) if free.any() else 0.0, 4),
                         frag=round(boundary(med) / max(med.sum(), 1), 4), flicker=round(flick, 4),
                         ir=int(chroma < 4), cluster=c, split='val' if c in val_cl else 'train'))
        g = cv2.cvtColor(fr[N // 2], cv2.COLOR_BGR2GRAY); t = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
        t[..., 2] += 0.6 * A[N // 2].astype(np.float32); t = t.clip(0, 255).astype(np.uint8)  # red = alpha (mid frame)
        t[nocut] = (0.5 * t[nocut] + (0, 127, 0)).astype(np.uint8)                                 # green = no-cut
        t = cv2.resize(t, (240, 135)); cv2.putText(t, f'{v} fr{rows[-1]["frag"]:.2f} fl{flick:.2f}', (3, 12), 0, 0.35, (255, 255, 255), 1)
        tiles.append((rows[-1]['frag'], t))
        if i % 20 == 0:
            print(f'  {i + 1}/{len(vids)} {v} start={start} ape_cov={nocut.mean():.2f} flicker={flick:.3f} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT / 'foliage_index.csv', index=False)
    print(f'{len(T)} sources -> {OUT / "foliage_index.csv"} ({time.time() - t0:.0f}s); '
          f'split {dict(T.split.value_counts())}, ir {int(T.ir.sum())}')
    print(T[['ape_cov', 'near_cov', 'frag', 'flicker']].quantile([0.1, 0.5, 0.9]).round(3).to_string())
    tiles = [t for _, t in sorted(tiles, key=lambda z: -z[0])]                             # most fragmented first
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    Path('images/ape').mkdir(parents=True, exist_ok=True)
    cv2.imwrite('images/ape/foliage_pool.jpg', np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    print('montage -> images/ape/foliage_pool.jpg (sorted by frag, descending; red = alpha mid frame, green = no-cut)')


if __name__ == '__main__':
    main()
