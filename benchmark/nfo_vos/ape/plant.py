"""Clash search: for every training clip, the real-foliage crops that occlude THIS ape's 40-frame trajectory well.

Why: a static occluder over a moving ape only teaches Layer-SAM something if (i) the ape is partly hidden in most frames,
(ii) the occluder cuts it into fragments, and (iii) parts hidden at t were visible at t-1..t-6, the frames the integral
image averages. Placing foliage blind gives none of these reliably, so we search.

Per clip (results/ape/clip_index.csv, keep): S_t = SBR mask of the ape, cropped to R and resampled so R's longer side is
G = 128 px. Candidates: every train-split pool source (results/ape/foliage_index.csv) with the clip's IR flag, other than
the clip's own video; scale s in {0.7, 1.0, 1.4} x R; NOFF random offsets; the crop of the source's median near-layer
alpha (40 frames), binarised at 0.5 -> O. Rejected if it touches the source's ape no-cut mask (> 1%).
Scores, with |.| = pixel count on the grid:
  h_t     = |S_t & O| / |S_t|                               hidden fraction; valid if h_t in [0.15, 0.75] in >= 34/40 frames
  frag    = median_t |edge(O) & S_t| / sqrt|S_t|             occluder boundary inside the ape (fragmentation)
  recov   = mean_t |H_t & U_t| / |H_t|,  H_t = S_t & O,      share of the hidden ape that the last 6 frames saw:
            U_t = OR_{k=1..6} shift(S_{t-k} & ~O, c_t - c_{t-k})   (rigid box-centre motion, as the integral image's shift)
Per clip and hidden-fraction bin (median h in [.15,.35), [.35,.55), [.55,.75]): the 64 most fragmented valid candidates
get recov; the top K = 8 by recov, one per source, are kept.
    python -m benchmark.nfo_vos.ape.plant  -> results/ape/plants.csv, images/ape/plants.jpg
Planting itself (plant()) composites the per-frame alpha and pixels of the same source crop over R: real sway.
"""
import json, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as Fn
from torchvision.ops import roi_align
from pycocotools import mask as M
from benchmark.nfo_vos.ape import PANAF_ROOT, SBR_ROOT, OUT
from benchmark.nfo_vos.ape.foliage import N

G, SCALES, NOFF, CHUNK = 128, (0.7, 1.0, 1.4), 64, 4096
HB, MIN_IN = (0.15, 0.75), N - 6
BINS = ((0.15, 0.35), (0.35, 0.55), (0.55, 0.75))
TOPF, K, LAG = 64, 8, 6
DEV = 'cuda'


def pool():
    """Median alpha + no-cut of every pool source, padded to one tensor (n, 2, Hm, Wm) on the GPU (cached)."""
    T = pd.read_csv(OUT / 'foliage_index.csv'); c = OUT / 'foliage_med.npz'
    if c.exists():
        a, nc = np.load(c)['a'], np.load(c)['nc']
    else:
        a = np.zeros((len(T), T.H.max(), T.W.max()), np.uint8); nc = np.ones_like(a, bool)
        for i, r in T.iterrows():
            z = np.load(OUT / 'foliage' / f'{r.video}.npz'); a[i, :r.H, :r.W] = np.median(z['alpha'], 0); nc[i, :r.H, :r.W] = z['nocut']
        np.savez_compressed(c, a=a, nc=nc)
    P = torch.stack([torch.from_numpy(a).float() / 255, torch.from_numpy(nc).float()], 1).to(DEV)
    return T, P


def chroma_ir(f):
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.int16); return int((np.abs(lab[..., 1] - 128) + np.abs(lab[..., 2] - 128)).mean() < 4)


def ape_clip(r):
    """S (N,H,W) bool and boxes (N,4) (nan where absent) of ape r.ape_id over frame ids start..start+N-1."""
    js = json.load(open(next(SBR_ROOT.glob(f'annotations*/train/{r.video}.json'))))
    S = np.zeros((N, r.H, r.W), bool); B = np.full((N, 4), np.nan)
    for a in js['annotations']:
        k = a['frame_id'] - r.start
        if 0 <= k < N:
            for d in a['detections']:
                if str(d['ape_id']) == str(r.ape_id) and d['segmentation']:
                    s = d['segmentation']; S[k] = M.decode({'size': s['size'], 'counts': s['counts'].encode()}).astype(bool); B[k] = d['bbox']
    return S, B


def frames(path, start, n=N, gray=True):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, start); out = []
    for _ in range(n):
        ok, f = cap.read(); out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f)
    return np.stack(out)


def shift(x, dy, dx):
    """y[p] = x[p - d], zero fill."""
    y = torch.zeros_like(x); H, W = x.shape[-2:]
    if abs(dy) >= H or abs(dx) >= W:
        return y
    y[..., max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = x[..., max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return y


def edge(O):
    return O * (1 - (-Fn.max_pool2d(-O[:, None], 3, 1, 1))[:, 0])          # O minus its 3x3 erosion


def recov(O, Sg, c):
    """O (m,gh,gw) {0,1}; Sg (N,gh,gw) {0,1}; c (N,2) centres (x,y) on the grid -> (m,) mean recoverable hidden share."""
    num, cnt = torch.zeros(len(O), device=DEV), torch.zeros(len(O), device=DEV)
    for t in range(LAG, N):
        Ht = Sg[t] * O; nh = Ht.flatten(1).sum(1)
        if Sg[t].sum() == 0:
            continue
        U = torch.zeros_like(O)
        for k in range(1, LAG + 1):
            d = np.round(c[t] - c[t - k]).astype(int); U = torch.maximum(U, shift(Sg[t - k] * (1 - O), int(d[1]), int(d[0])))
        ok = nh > 0; num += torch.where(ok, (Ht * U).flatten(1).sum(1) / nh.clamp(min=1), 0); cnt += ok.float()
    return num / cnt.clamp(min=1)


def search(r, T, P, rng):
    x1, y1, x2, y2 = [int(round(v)) for v in json.loads(r.R)]; Rw, Rh = x2 - x1, y2 - y1
    g = G / max(Rw, Rh); gw, gh = max(int(round(Rw * g)), 8), max(int(round(Rh * g)), 8)
    S, B = ape_clip(r)
    Sg = (Fn.interpolate(torch.from_numpy(S[:, y1:y2, x1:x2]).float()[:, None].to(DEV), size=(gh, gw), mode='area')[:, 0] > 0.5).float()
    ok_b = ~np.isnan(B[:, 0]); t_ = np.arange(N)
    c = np.stack([np.interp(t_, t_[ok_b], (B[ok_b, 0] + B[ok_b, 2]) / 2 - x1), np.interp(t_, t_[ok_b], (B[ok_b, 1] + B[ok_b, 3]) / 2 - y1)], 1) * g
    ir = chroma_ir(frames(SBR_ROOT / 'videos' / f'{r.video}.mp4', r.start - 1 + N // 2, 1, gray=False)[0])
    src = T[(T.split == 'train') & (T.ir == ir) & (T.video != r.video)]
    bx = []
    for i, s in src.iterrows():
        for sc in SCALES:
            cw, ch = sc * Rw, sc * Rh
            if cw >= s.W or ch >= s.H:
                continue
            ox, oy = rng.uniform(0, s.W - cw, NOFF), rng.uniform(0, s.H - ch, NOFF)
            bx.append(np.stack([np.full(NOFF, i), ox, oy, ox + cw, oy + ch, np.full(NOFF, sc)], 1))
    bx = np.concatenate(bx); Sf = Sg.flatten(1); area = Sf.sum(1); has = area > 0
    keep, H_, F_ = [], [], []
    for k0 in range(0, len(bx), CHUNK):
        b = torch.from_numpy(bx[k0:k0 + CHUNK, :5]).float().to(DEV)
        z = roi_align(P, b, (gh, gw), 1.0, 2, True); O = (z[:, 0] >= 0.5).float()
        h = (O.flatten(1) @ Sf.T) / area.clamp(min=1)                                          # (c, N)
        inb = ((h >= HB[0]) & (h <= HB[1])) | ~has
        fr = (edge(O).flatten(1) @ Sf.T) / area.clamp(min=1).sqrt()
        good = (inb.sum(1) >= MIN_IN) & (z[:, 1].flatten(1).mean(1) <= 0.01)
        hm = torch.where(has, h, torch.nan).nanmedian(1).values; fm = torch.where(has, fr, torch.nan).nanmedian(1).values
        keep.append(np.arange(k0, k0 + len(b))[good.cpu().numpy()]); H_.append(hm[good].cpu().numpy()); F_.append(fm[good].cpu().numpy())
    keep, H_, F_ = np.concatenate(keep), np.concatenate(H_), np.concatenate(F_)
    rows, stats = [], dict(video=r.video, ape_id=r.ape_id, n_cand=len(bx), n_valid=len(keep), ir=ir)
    for lo, hi in BINS:
        m = (H_ >= lo) & (H_ < hi); stats[f'n_{lo}'] = int(m.sum())
        if not m.any():
            continue
        idx = np.where(m)[0]; idx = idx[np.argsort(-F_[idx])[:TOPF]]
        b = torch.from_numpy(bx[keep[idx], :5]).float().to(DEV)
        O = (roi_align(P, b, (gh, gw), 1.0, 2, True)[:, 0] >= 0.5).float(); rc = recov(O, Sg, c).cpu().numpy()
        stats[f'recov_all_{lo}'] = float(np.median(rc)); used = set()
        for j in np.argsort(-rc):
            si = int(bx[keep[idx[j]], 0])
            if si in used:
                continue
            used.add(si); q = bx[keep[idx[j]]]
            rows.append(dict(video=r.video, ape_id=r.ape_id, start=r.start, R=r.R, bin=lo, src=T.video[si], src_start=int(T.start[si]),
                             sx1=round(q[1], 1), sy1=round(q[2], 1), sx2=round(q[3], 1), sy2=round(q[4], 1), scale=q[5],
                             h=round(float(H_[idx[j]]), 3), frag=round(float(F_[idx[j]]), 3), recov=round(float(rc[j]), 3), rank=len(used) - 1))
            if len(used) == K:
                break
    return rows, stats


def plant(X, R, F, A, crop):
    """X (N,H,W) gray target frames; R (x1,y1,x2,y2); F (N,Hs,Ws) gray source frames; A (N,Hs,Ws) uint8 alpha of the same
    window; crop (sx1,sy1,sx2,sy2) in source pixels. Returns planted X' and the per-frame alpha over the target (N,H,W)."""
    x1, y1, x2, y2 = [int(round(v)) for v in R]; a, b, c_, d = [int(round(v)) for v in crop]
    Xp = X.astype(np.float32).copy(); Af = np.zeros(X.shape, np.float32)
    for t in range(len(X)):
        f = cv2.resize(F[t, b:d, a:c_], (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA).astype(np.float32)
        al = cv2.resize(A[t, b:d, a:c_], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        Xp[t, y1:y2, x1:x2] = al * f + (1 - al) * Xp[t, y1:y2, x1:x2]; Af[t, y1:y2, x1:x2] = al
    return Xp.round().astype(np.uint8), Af


def montage(P_, C, n_clips=4, path='images/ape/plants.jpg'):
    """Per clip: the original (unplanted) row, then the rank-0 plant of n_clips random clips x 3 bins. Left 4 tiles: planted gray frames t = 6, 17, 28, 39 (crop R);
    right 4: labels (white = visible ape, blue = hidden ape, dark gray = occluder alpha >= 0.5)."""
    rng = np.random.default_rng(1); vids = rng.choice(P_.video.unique(), n_clips, replace=False); rows = []
    for v in vids:
        r = C[C.video == v].iloc[0]; S, _ = ape_clip(r); X = frames(SBR_ROOT / 'videos' / f'{v}.mp4', r.start - 1)
        x1, y1, x2, y2 = [int(round(q)) for q in json.loads(r.R)]; ts = [6, 17, 28, 39]
        row = np.hstack([cv2.resize(cv2.cvtColor(X[t, y1:y2, x1:x2], cv2.COLOR_GRAY2BGR), (200, 160)) for t in ts]
                        + [cv2.resize(np.repeat(S[t, y1:y2, x1:x2, None].astype(np.uint8) * 255, 3, 2), (200, 160)) for t in ts])
        cv2.putText(row, f'{v} ORIGINAL (unoccluded)', (4, 14), 0, 0.45, (0, 255, 0), 1); rows.append(row)
        for lo, _ in BINS:
            p = P_[(P_.video == v) & (P_.bin == lo) & (P_['rank'] == 0)]
            if p.empty:
                continue
            p = p.iloc[0]; F = frames(PANAF_ROOT / 'videos' / f'{p.src}.mp4', p.src_start)
            A = np.load(OUT / 'foliage' / f'{p.src}.npz')['alpha']
            Xp, Af = plant(X, (x1, y1, x2, y2), F, A, (p.sx1, p.sy1, p.sx2, p.sy2)); ts = [6, 17, 28, 39]
            ti = [cv2.cvtColor(Xp[t, y1:y2, x1:x2], cv2.COLOR_GRAY2BGR) for t in ts]
            lb = []
            for t in ts:
                o = Af[t, y1:y2, x1:x2] >= 0.5; s = S[t, y1:y2, x1:x2]; L = np.zeros(o.shape + (3,), np.uint8)
                L[o] = (70, 70, 70); L[s & o] = (255, 90, 0); L[s & ~o] = (255, 255, 255); lb.append(L)
            row = np.hstack([cv2.resize(q, (200, 160)) for q in ti + lb])
            cv2.putText(row, f'{v} bin {lo} h {p.h:.2f} frag {p.frag:.2f} recov {p.recov:.2f}', (4, 14), 0, 0.45, (0, 255, 255), 1)
            rows.append(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'montage -> {path}')


def main():
    T, P = pool(); C = pd.read_csv(OUT / 'clip_index.csv'); C = C[C.keep].reset_index(drop=True)
    rng = np.random.default_rng(0); rows, stats, t0 = [], [], time.time()
    for i, r in C.iterrows():
        a, s = search(r, T, P, rng); rows += a; stats.append(s)
        if i % 20 == 0:
            print(f'  {i + 1}/{len(C)} {r.video}: {s["n_valid"]}/{s["n_cand"]} valid, bins {[s.get(f"n_{lo}", 0) for lo, _ in BINS]} ({time.time() - t0:.0f}s)', flush=True)
    P_ = pd.DataFrame(rows); P_.to_csv(OUT / 'plants.csv', index=False); St = pd.DataFrame(stats)
    St.to_csv(OUT / 'plants_stats.csv', index=False)
    print(f'{len(C)} clips, {len(P_)} plants -> {OUT / "plants.csv"} ({time.time() - t0:.0f}s)')
    for lo, _ in BINS:
        q = P_[P_.bin == lo]; n = q.groupby(['video', 'ape_id']).size()
        print(f'bin {lo}: clips with >= 1 plant {len(n)}/{len(C)}, with {K}: {(n == K).sum()}; selected recov median {q.recov.median():.3f} '
              f'(rank 0: {q[q["rank"] == 0].recov.median():.3f}) vs top-frag pool median {St[f"recov_all_{lo}"].median():.3f}; '
              f'frag median {q.frag.median():.3f}')
    montage(P_, C)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['montage']:   # re-render from results/ape/plants.csv without re-searching
        C = pd.read_csv(OUT / 'clip_index.csv'); montage(pd.read_csv(OUT / 'plants.csv'), C[C.keep].reset_index(drop=True))
    else:
        main()
