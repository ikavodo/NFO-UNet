"""Per-clip online batches for approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11). Nothing is
cached: each clip is regenerated from its seed in a DataLoader worker; SAM2 encodes on the GPU (encode()).

Per clip (sample_occlusion): person/scene scale s ~ U[0.5, 1] (zoom-out; GPJATK person 0.53 of frame height, NFO
0.33), ρ = 0 (10%) or ρ ~ U[0.2, 0.6], clump ∈ {12, 24, (12, 24)}, flutter ~ U[0, 2]·noise.FLUTTER_AMP.
Alignment (2026-10-16, v3): the kernel-f16 tracker (layers/track_check.kernel_track on 1 − cos(F_t, F_B)), the same
path as on NFO (fusion_nfo). It located the person in 100% of synthetic frames (centre error 0.092 person heights vs
0.105 on NFO); the blob tracker of v2 covered only 48% and left a third of the training frames without a composite.
Whole-clip batches (build): each frame is encoded once and reused. Backgrounds: template photo or CanaTree100 (50/50;
every 5th image held out for val). Composite parameters randomised per clip. Mode 2 (5%): composite = empty scene.
Composite: V̂'s gated composite C (factorised_gtfree.composite_frames) and its evidence mass a = Σ w G.
"""
import os

import cv2
import numpy as np
import torch

AMP = torch.float16 if os.environ.get('AMP') == 'fp16' else torch.bfloat16   # fp16 on pre-Ampere GPUs

from benchmark.nfo_vos.layers import dataset as D
from benchmark.nfo_vos.layers.prompt_oracle import SIGMA_A, composite_and_mass
from benchmark.nfo_vos.synth import noise

P_MODES = (0.95, 0.0, 0.05, 0.0)   # modes 1 / 3 (emulated tracker errors) retired: real tracker
CLUMPS = (12, 24, (12, 24))


def specs(split):
    return D.clip_specs('train') if split == 'train' else D.clip_specs('val')[::4]


def rescale_clip(d, s):
    """Zoom out by s ≤ 1 (person scale augmentation): every image layer resized by s and centred; the border is the
    reflected, rescaled empty scene (never the frame, so a person near the edge is not mirrored without a label);
    S / V resized by area > 0.5 and zero-padded; boxes (cx, cy, h) mapped. s = 1 returns d unchanged."""
    if s >= 1.0:
        return d
    H, W = d['bg'].shape
    h, w = int(round(H * s)), int(round(W * s))
    y0, x0 = (H - h) // 2, (W - w) // 2
    rs = lambda img: cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(rs(d['bg']), y0, H - h - y0, x0, W - w - x0, cv2.BORDER_REFLECT)

    def place(img):
        out = canvas.copy(); out[y0:y0 + h, x0:x0 + w] = rs(img); return out

    def place_mask(m):
        out = np.zeros((H, W), bool); out[y0:y0 + h, x0:x0 + w] = rs(m) > 0.5; return out
    r = dict(d)
    r['X'] = np.stack([place(x) for x in d['X']])
    r['bg'] = place(d['bg'])
    if 'warm' in d:
        r['warm'] = np.stack([place(x) for x in d['warm']])
    r['S'], r['V'] = np.stack([place_mask(m) for m in d['S']]), np.stack([place_mask(m) for m in d['V']])
    b = d['boxes'].copy()
    b[:, 0], b[:, 1], b[:, 2] = b[:, 0] * s + x0, b[:, 1] * s + y0, b[:, 2] * s
    r['boxes'] = b
    return r


def sample_occlusion(rng):
    """10% unoccluded clips (ρ = 0), so the adapter learns R ≈ 0 where nothing is hidden."""
    return dict(scale=float(rng.uniform(0.5, 1.0)), rho=0.0 if rng.random() < 0.1 else float(rng.uniform(0.2, 0.6)), clump=CLUMPS[rng.integers(3)],
                flutter=float(rng.uniform(0, 2) * noise.FLUTTER_AMP))


def a256(m):
    return np.clip(cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA), 0, 1)   # area resample overshoots by ~1e-7


def u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def to224(img):
    """Native frame → the 224 padded square the blob tracker runs on (NFO: BORDER_REPLICATE, then resize)."""
    H, W = img.shape
    s = max(H, W)
    sq = cv2.copyMakeBorder(img, (s - H) // 2, s - H - (s - H) // 2, (s - W) // 2, s - W - (s - W) // 2, cv2.BORDER_REPLICATE)
    return cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA)


def tracker_chain(d, box0):
    """Run the real blob tracker on a (rescaled) clip from the t0 box; returns {local t: native x} of its chain."""
    from benchmark.nfo_vos import run_t4 as R
    H, W = d['bg'].shape
    s = max(H, W)
    u = lambda x: to224(u8(x))
    to224c = lambda x, y: ((x + (s - W) / 2) * 224 / s, (y + (s - H) / 2) * 224 / s)
    x0, y0 = to224c(box0[0], box0[1]); x1, y1 = to224c(box0[2], box0[3])
    D_ = R.detect_and_track_frames(np.stack([u(x) for x in d['X']]), np.stack([u(x) for x in d['warm']]), y1 - y0)
    Tk = R.track(dict(frames=list(range(len(d['X']))), warmup=None, box_224=[x0, y0, x1, y1]), D=D_)
    return {t: (p[0] + 0.5) / 224 * s - (s - W) / 2 for t, p in Tk['chain'].items()}


CANATREE = sorted(__import__('glob').glob('data/CanaTree100/images/*.png'))
BG_SPLIT = {'train': [p for i, p in enumerate(CANATREE) if i % 5], 'val': [p for i, p in enumerate(CANATREE) if i % 5 == 0]}


NEAR_PX, FAR_W = 8, 0.3


def loss_weights(S, V):
    """Layer-aware loss weights on SAM2's 256² grid (v3.4, 2026-10-16; the pseudo-GT's three pixel classes, cf. BCNet's
    occluder / occludee layers, arXiv 2103.12340): 1 on the visible person and outside the person (background, incl.
    see-through gaps: leaks are errors); 0 on occluded person pixels within NEAR_PX native px of a visible fragment
    (S \ V near V: the annotators' void / gap-filling convention); FAR_W on occluded person far from any visible
    fragment (modal: hidden parts are mildly penalised, not ignored)."""
    near = cv2.dilate(V.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR_PX + 1, 2 * NEAR_PX + 1))) > 0
    occ = S & ~V
    w = np.ones(S.shape, np.float32)
    w[occ & near] = 0.0
    w[occ & ~near] = FAR_W
    return cv2.resize(w, (256, 256), interpolation=cv2.INTER_AREA)


def clip_raw(spec, rng):
    """One clip on the CPU (DataLoader worker): rescaled frames, labels, jittered boxes (the t0 prompt), per-frame
    corruption mode, and the composite parameters drawn for this clip (randomised: history n_h ∈ 4..8, age τ ∈ [2, 6],
    gate z0 ∈ [2, 4]; NFO uses 6 / 3.5 / 3)."""
    seq, cam, _, seed = spec
    occ = sample_occlusion(rng)
    occ.update(nh=int(rng.integers(4, 9)), tau=float(rng.uniform(2, 6)), z0='adaptive')   # z0 range [2, 4] broke input parity
    d = rescale_clip(D.make_clip(seq, cam, occ['rho'], seed, flutter=occ['flutter'], clump=occ['clump']), occ['scale'])
    return dict(X=u8(d['X']), bg=u8(d['bg']), V=d['V'], S=d['S'], boxes=d['boxes'], bxj=D.jitter(d['boxes'], SIGMA_A, rng),
                modes=rng.choice(4, len(d['X']), p=P_MODES), occ=occ, cid=d['cid'])


def _enc(pred, imgs, chunk=8):
    rgb = lambda a: [np.stack([x] * 3, -1) for x in a]
    out = [[], [], []]
    for i in range(0, len(imgs), chunk):
        pred.set_image_batch(rgb(imgs[i:i + chunk]))
        hr = pred._features['high_res_feats']
        for o, x in zip(out, (pred._features['image_embed'], hr[1], hr[0])):
            o.append(x.half())
    return [torch.cat(o) for o in out]


@torch.no_grad()
def build(pred, raw, max_targets=None, rng=None, dev='cuda', all_targets=False):
    """GPU side of one clip, the test-time path: encode every frame once (+ the empty scene); kernel-f16 tracker
    (layers/track_check.kernel_track on 1 − cos(F_t, F_B)) from the jittered t0 box; for each target t, x-only Theil–Sen
    on the track over [t-12, t] aligns frames t-n_h..t-1 into V̂'s gated composite (composite_and_mass with the clip's
    τ, z0); the composites are encoded. Each frame's F_t is reused as its own target input, so the encode cost per
    sample is ≈ (T + K + 1) / K. Returns (f, meta)."""
    from concurrent.futures import ThreadPoolExecutor
    from benchmark.nfo_vos.factorised_gtfree import theil_at
    from benchmark.nfo_vos.layers.track_check import kernel_track
    X, bg, occ = raw['X'], raw['bg'], raw['occ']
    T, H, W = X.shape
    with torch.autocast('cuda', dtype=AMP):
        F16, F8, F4 = _enc(pred, X)
        B16, B8, B4 = _enc(pred, bg[None])
    dmap = 1 - torch.nn.functional.cosine_similarity(F16.float(), B16.float(), dim=1)
    dmap = torch.nn.functional.interpolate(dmap[:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0].cpu().numpy()
    cx0, cy0, h0 = raw['bxj'][0]
    S0 = raw['S'][0] if raw['S'][0].any() else raw['S'][int(np.argmax(raw['S'].sum((1, 2))))]
    ys, xs = np.nonzero(S0); asp0 = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
    trk = kernel_track(dmap, (cx0, cy0), asp0 * h0, h0)
    nh = occ['nh']
    ok = ~np.isnan(raw['boxes']).any(1)
    # all_targets: every frame with history, no GT gate (the memory bank at test time has no V); training default keeps
    # the visible-person gate for the loss targets
    ts = list(range(max(6, nh), T)) if all_targets else [t for t in range(max(6, nh), T) if ok[t] and raw['V'][t].any()]
    if max_targets and len(ts) > max_targets:
        ts = sorted((rng or np.random.default_rng()).choice(ts, max_targets, replace=False))

    def comp(t):
        if raw['modes'][t] == 2:
            return bg.astype(np.float32), np.zeros((H, W), np.float32)
        hist = list(range(t - nh, t))
        cs = list(range(max(0, t - 12), t + 1))
        fx = theil_at(cs, [trk[l, 0] for l in cs], [t] + hist)
        return composite_and_mass([X[i].astype(np.float32) for i in hist],
                                  [np.float32([[1, 0, fx[t] - fx[i]], [0, 1, 0]]) for i in hist], bg.astype(np.float32),
                                  occ['tau'], occ['z0'])
    with ThreadPoolExecutor(8) as ex:
        CA = list(ex.map(comp, ts))
    with torch.autocast('cuda', dtype=AMP):
        C16, C8, C4 = _enc(pred, np.stack([u8(c) for c, _ in CA]))
    idx = torch.as_tensor(ts, device=dev)
    sc = np.array([1024 / W, 1024 / H] * 2, np.float32)
    boxes = []
    for t in ts:
        if not raw['S'][t].any() or np.isnan(raw['bxj'][t]).any():          # no person / no GT box: tracker box
            boxes.append(np.concatenate([trk[t] - [asp0 * h0 / 2, h0 / 2], trk[t] + [asp0 * h0 / 2, h0 / 2]]).astype(np.float32) * sc)
            continue
        ys, xs = np.nonzero(raw['S'][t]); asp = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
        cx, cy, h = raw['bxj'][t]
        boxes.append(np.array([cx - asp * h / 2, cy - h / 2, cx + asp * h / 2, cy + h / 2], np.float32) * sc)
    n = len(ts)
    f = {'X16': F16[idx].float(), 'X8': F8[idx].float(), 'X4': F4[idx].float(),
         'C16': C16.float(), 'C8': C8.float(), 'C4': C4.float(),
         'B16': B16.float().expand(n, -1, -1, -1), 'B8': B8.float().expand(n, -1, -1, -1), 'B4': B4.float().expand(n, -1, -1, -1),
         'a': torch.as_tensor(np.stack([a256(a) for _, a in CA]), device=dev),
         'V': torch.as_tensor(np.stack([a256(raw['V'][t]) for t in ts]), device=dev),
         'W': torch.as_tensor(np.stack([loss_weights(raw['S'][t], raw['V'][t]) for t in ts]), device=dev),
         'box': torch.as_tensor(np.stack(boxes), device=dev),
         'mode': torch.as_tensor(raw['modes'][ts].astype(np.float32), device=dev)}
    f['pt'] = torch.stack([(f['box'][:, 0] + f['box'][:, 2]) / 2, (f['box'][:, 1] + f['box'][:, 3]) / 2], 1)
    f['trk'] = torch.as_tensor(trk[ts] * sc[:2], device=dev).float()        # tracker position, 1024 frame
    hb = np.array([asp0 * h0 / 2, h0 / 2], np.float32)                     # tracker box: t0 size, tracker centre
    f['box_trk'] = torch.as_tensor(np.concatenate([trk[ts] - hb, trk[ts] + hb], 1) * sc, device=dev).float()
    Vn = [raw['V'][t] for t in ts]
    on = [float(np.asarray(f['a'][i].cpu())[a256(Vn[i]) > 0.5].mean()) if Vn[i].any() else np.nan for i in range(n)]
    meta = dict(Vn=Vn, ts=ts, cid=raw['cid'], occ=occ, mode=raw['modes'][ts], trk=trk, a_on=on,
                a_mass=[float(x) for x in f['a'].sum((1, 2)).cpu()])
    return f, meta


class Clips(torch.utils.data.IterableDataset):
    """Raw clips (clip_raw) from CPU workers. train: endless, per-worker seed stream, CanaTree train pool; val: the fixed
    clip set (first `n_val`), occlusion drawn deterministically per clip, CanaTree held-out pool (every 5th image)."""

    def __init__(self, split, seed=0, n_val=None):
        self.split, self.seed, self.n_val = split, seed, n_val

    def __iter__(self):
        from benchmark.nfo_vos.synth import gpjatk_foliage as GF
        GF.set_bg_pool(BG_SPLIT['train' if self.split == 'train' else 'val'])
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        S = specs(self.split)
        if self.split == 'train':
            rng = np.random.default_rng([self.seed, wid])
            while True:
                yield clip_raw(S[rng.integers(len(S))], rng)
        else:
            for k in range(wid, len(S[:self.n_val]), nw):
                yield clip_raw(S[k], np.random.default_rng([1, k]))


@torch.no_grad()
def encode(pred, clips, dev='cuda'):
    """Frozen SAM2 (bf16) on a list of clip batches: features of every target frame X, its composite C, and each clip's
    empty scene B (encoded once per clip, broadcast to its frames). Returns one flat batch."""
    rgb = lambda a: [np.stack([x] * 3, -1) for x in a]
    f = {}

    def run(imgs):
        pred.set_image_batch(rgb(imgs))
        hr = pred._features['high_res_feats']
        return pred._features['image_embed'].float(), hr[1].float(), hr[0].float()

    with torch.autocast('cuda', dtype=AMP):
        for key, src in (('X', 'X'), ('C', 'Ct')):
            parts = [run(c[src]) for c in clips]
            for i, l in enumerate(('16', '8', '4')):
                f[f'{key}{l}'] = torch.cat([p[i] for p in parts])
        bparts = run([c['bg'] for c in clips])
        idx = torch.cat([torch.full((len(c['X']),), j) for j, c in enumerate(clips)]).to(dev)
        for i, l in enumerate(('16', '8', '4')):
            f[f'B{l}'] = bparts[i][idx]
    for k in ('a', 'V', 'box', 'mode') + tuple(k for k in ('trk', 'box_trk') if all(k in c for c in clips)):
        f[k] = torch.as_tensor(np.concatenate([c[k] for c in clips])).to(dev).float()
    f['pt'] = torch.stack([(f['box'][:, 0] + f['box'][:, 2]) / 2, (f['box'][:, 1] + f['box'][:, 3]) / 2], 1)
    return f
