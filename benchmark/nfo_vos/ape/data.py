"""Ape clip loader (Stage 1): one planted training clip in the raw-clip format of adapter/fusion_data.clip_raw, so
fusion_data.build and training run unchanged.

Per clip (results/ape/clip_index.csv, keep): the ape's 40 gray frames X_t (PanAf-SBR video), its SBR masks S_t, and a
plant from results/ape/plants.csv (ape/plant.py: foliage crops chosen to clash with this ape's trajectory). Sampling:
a clip uniformly; with P_CLEAR = 0.1 no plant (the adapter must learn that nothing needs fixing), else an occlusion level
uniformly (15-35 / 35-55 / 55-75% hidden), then one of its <= 8 plants uniformly.
  - X'_t = alpha_t F_t + (1 - alpha_t) X_t over R, with the source's per-frame alpha and pixels (real sway), plant.plant();
  - B = the whole-video median of the ape video (cached, results/ape/bg/); B' = median_t of the planted B (the static
    part of the foliage is in the empty scene, its sway is not);
  - V_t = S_t & (alpha_t < 0.5). Boxes (cx, cy, h) = tight boxes of S_t (design should-fix 9), NaN where S_t is empty;
  - track = box centres (gaps interpolated) + N(0, (TRK_SIGMA h)^2) per frame, given to build() as raw['track'] (no
    kernel tracker; TRK_SIGMA is assumed, to calibrate on MD-seeded SAM2 boxes); occ['shift_xy'] = True (x + y Theil-Sen).
    python -m benchmark.nfo_vos.ape.data   -> timing + images/ape/loader.jpg (X'_t, integral image C_t, labels)
"""
import json
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from benchmark.nfo_vos.ape import PANAF_ROOT, SBR_ROOT, OUT
from benchmark.nfo_vos.ape.foliage import N
from benchmark.nfo_vos.ape.plant import ape_clip, frames, plant
from benchmark.nfo_vos.adapter import dataset as D
from benchmark.nfo_vos.adapter.fusion_data import P_MODES, SIGMA_A

P_CLEAR, TRK_SIGMA = 0.10, 0.05


def background(video):
    """Whole-video gray median of the SBR video, cached as results/ape/bg/<video>.png."""
    p = OUT / 'bg' / f'{video}.png'
    if not p.exists():
        cap, fr = cv2.VideoCapture(str(SBR_ROOT / 'videos' / f'{video}.mp4')), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        p.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(p), np.median(np.stack(fr), 0).astype(np.uint8))
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def tight_boxes(S):
    out = np.full((len(S), 3), np.nan)
    for t, s in enumerate(S):
        if s.any():
            ys, xs = np.nonzero(s); out[t] = ((xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, ys.max() - ys.min() + 1)
    return out


def fill(b):
    """NaN rows filled by linear interpolation / nearest at the ends."""
    ok = ~np.isnan(b[:, 0]); t = np.arange(len(b))
    return np.stack([np.interp(t, t[ok], b[ok, j]) for j in range(b.shape[1])], 1)


def clip_raw(row, p, rng):
    """row: a clip_index row; p: a plants.csv row or None (unoccluded)."""
    S, _ = ape_clip(row)
    X = frames(SBR_ROOT / 'videos' / f'{row.video}.mp4', row.start - 1)
    bg = background(row.video)
    if p is None:
        A = np.zeros(X.shape, np.float32)
    else:
        R = json.loads(row.R); crop = (p.sx1, p.sy1, p.sx2, p.sy2)
        F = frames(PANAF_ROOT / 'videos' / f'{p.src}.mp4', p.src_start); Al = np.load(OUT / 'foliage' / f'{p.src}.npz')['alpha']
        X, A = plant(X, R, F, Al, crop)
        bg = np.median(plant(np.repeat(bg[None], N, 0), R, F, Al, crop)[0], 0).astype(np.uint8)
    boxes = tight_boxes(S); bf = fill(boxes)
    track = bf[:, :2] + rng.normal(0, 1, (N, 2)) * TRK_SIGMA * bf[:, 2:3]
    occ = dict(nh=int(rng.integers(4, 9)), tau=float(rng.uniform(2, 6)), z0='adaptive', shift_xy=True,
               level=None if p is None else float(p.bin), h=None if p is None else float(p.h))
    return dict(X=X, bg=bg, V=S & (A < 0.5), S=S, boxes=boxes, bxj=D.jitter(bf, SIGMA_A, rng), track=track,
                modes=rng.choice(4, N, p=P_MODES), occ=occ, cid=f'{row.video}:{row.ape_id}:{"clear" if p is None else p.src}')


def index():
    C = pd.read_csv(OUT / 'clip_index.csv'); C = C[C.keep].reset_index(drop=True)
    P = pd.read_csv(OUT / 'plants.csv'); P['ape_id'] = P.ape_id.astype(str); C['ape_id'] = C.ape_id.astype(str)
    return C, {k: g for k, g in P.groupby(['video', 'ape_id'])}


def sample(C, P, rng):
    r = C.iloc[rng.integers(len(C))]; g = P.get((r.video, r.ape_id))
    if g is None or rng.random() < P_CLEAR:
        return r, None
    lv = rng.choice(g.bin.unique()); g = g[g.bin == lv]
    return r, g.iloc[rng.integers(len(g))]


class ApeClips(torch.utils.data.IterableDataset):
    """Endless planted training clips from CPU workers (per-worker seed stream), drop-in for fusion_data.Clips('train')."""

    def __init__(self, seed=0):
        self.seed = seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); wid = wi.id if wi else 0
        rng = np.random.default_rng([self.seed, wid, 7]); C, P = index()
        while True:
            yield clip_raw(*sample(C, P, rng), rng)


def main():
    """Timing of clip_raw, then one GPU build() per occlusion level; montage of X'_t, the integral image C_t and labels."""
    import time
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    from benchmark.nfo_vos.adapter.fusion_data import build, hist_warps
    from benchmark.nfo_vos.adapter.integral_image import integral_image_and_mass
    C, P = index(); rng = np.random.default_rng(0); t0 = time.time()
    for _ in range(5):
        clip_raw(*sample(C, P, rng), rng)
    print(f'clip_raw: {(time.time() - t0) / 5:.2f} s/clip (CPU, bg cache warm after first visit)', flush=True)
    pred, rows = image_predictor(), []
    r = C.iloc[int(rng.integers(len(C)))]; g = P[(r.video, r.ape_id)]
    for lv in [None] + sorted(g.bin.unique()):
        p = None if lv is None else g[(g.bin == lv) & (g['rank'] == 0)].iloc[0]
        raw = clip_raw(r, p, rng); t1 = time.time(); f, meta = build(pred, raw, max_targets=8, rng=rng)
        torch.cuda.synchronize(); bt = time.time() - t1
        t = 20; x1, y1, x2, y2 = [int(round(v)) for v in json.loads(r.R)]
        hist, Ms = hist_warps(raw['track'], t, raw['occ']['nh'], True)
        Cm, _ = integral_image_and_mass([raw['X'][i].astype(np.float32) for i in hist], Ms, raw['bg'].astype(np.float32), raw['occ']['tau'], 'adaptive')
        L = np.zeros(raw['S'][t].shape + (3,), np.uint8); L[raw['S'][t]] = (255, 90, 0); L[raw['V'][t]] = (255, 255, 255)
        tiles = [cv2.cvtColor(q, cv2.COLOR_GRAY2BGR) for q in (raw['X'][t], np.clip(Cm, 0, 255).astype(np.uint8), raw['bg'])] + [L]
        row = np.hstack([cv2.resize(q[y1:y2, x1:x2], (220, 180)) for q in tiles])
        hv = 1 - raw['V'][t].sum() / max(raw['S'][t].sum(), 1)
        cv2.putText(row, f'{"clear" if lv is None else f"level {lv}"} hidden {hv:.2f} | X_t, C_t, B, labels', (4, 14), 0, 0.42, (0, 255, 255), 1)
        rows.append(row)
        print(f'level {lv}: build {bt:.2f} s, {len(meta["ts"])} targets, X16 {tuple(f["X16"].shape)}, hidden at t={t}: {hv:.2f}', flush=True)
    Path('images/ape').mkdir(parents=True, exist_ok=True); cv2.imwrite('images/ape/loader.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'{r.video} ape {r.ape_id} -> images/ape/loader.jpg')


if __name__ == '__main__':
    main()
