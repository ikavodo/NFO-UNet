"""Per-clip online batches for approach A (design: docs/layer_sam2/plan_fusion_prompt_adapter.md §11). Nothing is
cached: each clip is regenerated from its seed in a DataLoader worker; SAM2 encodes on the GPU (encode()).

Per clip (sample_occlusion): person/scene scale s ~ U[0.5, 1] (zoom-out; GPJATK person 0.53 of frame height, NFO
0.33), ρ = 0 (10%) or ρ ~ U[0.2, 0.6], clump ∈ {12, 24, (12, 24)}, flutter ~ U[0, 2]·noise.FLUTTER_AMP.
Alignment (2026-10-16, v3): the kernel-f16 tracker (adapter/tracker.kernel_track on 1 − cos(F_t, F_B)), the same
path as on NFO (fusion_nfo). It located the person in 100% of synthetic frames (centre error 0.092 person heights vs
0.105 on NFO); the blob tracker of v2 covered only 48% and left a third of the training frames without an integral image.
Whole-clip batches (build): each frame is encoded once and reused. Backgrounds: template photo or CanaTree100 (50/50;
every 5th image held out for val). Integral image parameters randomised per clip. Mode 2 (5%): integral image = empty scene.
Integral image: the gated pixel average C and its evidence mass a = Σ w G (adapter/integral_image.py).
"""
import os

import cv2
import numpy as np
import torch

AMP = torch.float16 if os.environ.get('AMP') == 'fp16' else torch.bfloat16   # fp16 on pre-Ampere GPUs

from benchmark.nfo_vos.adapter import dataset as D
from benchmark.nfo_vos.adapter.integral_image import integral_image_and_mass
from benchmark.nfo_vos.synth import noise
from utils.stats_utils import theil_at

P_MODES = (0.95, 0.0, 0.05, 0.0)   # modes 1 / 3 (emulated tracker errors) retired: real tracker
CLUMPS = (12, 24, (12, 24))
SIGMA_A = 3.0                      # t0 prompt-box jitter (px)


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
    if 'Xt' in d:
        r['Xt'] = np.stack([place(x) for x in d['Xt']])
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


CANATREE = sorted(__import__('glob').glob('data/CanaTree100/images/*.png'))
BG_SPLIT = {'train': [p for i, p in enumerate(CANATREE) if i % 5], 'val': [p for i, p in enumerate(CANATREE) if i % 5 == 0]}


NEAR_PX, FAR_W = 8, 0.3


def loss_weights(S, V):
    """Layer-aware loss weights on SAM2's 256² grid (v3.4, 2026-10-16; the pseudo-GT's three pixel classes, cf. BCNet's
    occluder / occludee layers, arXiv 2103.12340): 1 on the visible person and outside the person (background, incl.
    see-through gaps: leaks are errors); 0 on occluded person pixels within NEAR_PX native px of a visible fragment
    (S − V near V: the annotators' void / gap-filling convention); FAR_W on occluded person far from any visible
    fragment (modal: hidden parts are mildly penalised, not ignored)."""
    near = cv2.dilate(V.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR_PX + 1, 2 * NEAR_PX + 1))) > 0
    occ = S & ~V
    w = np.ones(S.shape, np.float32)
    w[occ & near] = 0.0
    w[occ & ~near] = FAR_W
    return cv2.resize(w, (256, 256), interpolation=cv2.INTER_AREA)


def clip_raw(spec, rng):
    """One clip on the CPU (DataLoader worker): rescaled frames, labels, jittered boxes (the t0 prompt), per-frame
    corruption mode, and the integral image parameters drawn for this clip (randomised: history n_h ∈ 4..8, age τ ∈ [2, 6],
    gate z0 ∈ [2, 4]; NFO uses 6 / 3.5 / 3)."""
    seq, cam, _, seed = spec
    occ = sample_occlusion(rng)
    occ.update(nh=int(rng.integers(4, 9)), tau=float(rng.uniform(2, 6)), z0='adaptive')   # z0 range [2, 4] broke input parity
    d = rescale_clip(D.make_clip(seq, cam, occ['rho'], seed, flutter=occ['flutter'], clump=occ['clump']), occ['scale'])
    return dict(X=u8(d['X']), Xt=u8(d['Xt']), bg=u8(d['bg']), V=d['V'], S=d['S'], boxes=d['boxes'], bxj=D.jitter(d['boxes'], SIGMA_A, rng),
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


def hist_warps(trk, t, nh, xy=False):
    """History frames t-nh..t-1 and their shifts onto frame t: Theil–Sen fit of the track over [t-12, t], x only (NFO)
    or x and y (xy=True, apes)."""
    hist = list(range(t - nh, t))
    cs = list(range(max(0, t - 12), t + 1))
    fx = theil_at(cs, [trk[l, 0] for l in cs], [t] + hist)
    fy = theil_at(cs, [trk[l, 1] for l in cs], [t] + hist) if xy else None
    return hist, [np.float32([[1, 0, fx[t] - fx[i]], [0, 1, fy[t] - fy[i] if xy else 0]]) for i in hist]


@torch.no_grad()
def _cover(m, hw):
    """Area coverage > 0.5 of a native-size mask on the hw × hw feature grid (SAM2 resizes anisotropically to 1024²)."""
    return cv2.resize(m.astype(np.float32), (hw, hw), interpolation=cv2.INTER_AREA) > 0.5


def _feat_inputs(pred, raw, ts, trk, nh, Fs, dev):
    """Inputs of the learned feature integral for targets ts (build(feat=True)):
    K16/8/4 [n, nh, C, h, w]   cached features of frames t−nh..t−1, shifted onto t by the x-only Theil–Sen track fit
                               (hist_warps; dx in the 1024 frame), bilinear (feat_integral.shift_keys; G0, 2026-11-19)
    Kvalid16 [n, nh, 1, 64, 64] source cell inside the image; all False for mode-2 targets (integral image withheld)
    age [n, nh]                nh..1
    T16/8/4 [n, C, h, w]       features of Xt (frame t, same noise and sway, no occluder): the clean-feature target
    Om16/8/4 [n, 1, h, w]      hidden-recoverable person cells: S_t ∧ ¬V_t ∧ (∪_k V_k warped onto t), coverage > 0.5
    Vis16 [n, 1, 64, 64]       visible person cells (the loss floor)
    O4 [n, 256, 256]           person visible in ≥ 1 aligned history frame (area fraction; 0 for mode 2): mass target
    phase [n, nh]              dx/16 − round(dx/16): each key's sub-cell phase at stride 16
    mu16 [1, 256, 1, 1]        SAM2's no_mem_embed (centring for stride-16 cosines)"""
    from benchmark.nfo_vos.adapter.feat_integral import shift_keys
    H, W = raw['X'].shape[1:]
    hidx, dx, Om, Vis, Os = [], [], {s: [] for s in ('16', '8', '4')}, [], []
    for t in ts:
        hist, Ms = hist_warps(trk, t, nh)
        hidx.append(hist); dx.append([M[0, 2] * 1024 / W for M in Ms])
        O = np.zeros((H, W), bool)
        for i, M in zip(hist, Ms):
            O |= cv2.warpAffine(raw['V'][i].astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        Os.append(np.clip(cv2.resize(O.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA), 0, 1) * (raw['modes'][t] != 2))
        hid = O & raw['S'][t] & ~raw['V'][t]
        for s, hw in (('16', 64), ('8', 128), ('4', 256)):
            Om[s].append(_cover(hid, hw))
        Vis.append(_cover(raw['V'][t], 64))
    hidx = torch.as_tensor(hidx, device=dev); dx = torch.as_tensor(np.float32(dx), device=dev)
    o = {}
    for s, Fx in zip(('16', '8', '4'), Fs):
        o[f'K{s}'], v = shift_keys(Fx[hidx], dx, int(s))
        if s == '16':
            o['Kvalid16'] = v & torch.as_tensor(raw['modes'][ts] != 2, device=dev)[:, None, None, None, None]
        o[f'Om{s}'] = torch.as_tensor(np.stack(Om[s]), device=dev)[:, None]
    o['Vis16'] = torch.as_tensor(np.stack(Vis), device=dev)[:, None]
    o['O4'] = torch.as_tensor(np.stack(Os), device=dev)                       # target of the evidence mass (mass_loss)
    o['phase'] = dx / 16 - torch.round(dx / 16)                               # sub-cell shift phase at s16 (G0)
    o['age'] = torch.arange(nh, 0, -1, device=dev).repeat(len(ts), 1)
    with torch.autocast('cuda', dtype=AMP):
        T16, T8, T4 = _enc(pred, raw['Xt'][ts])
    o.update(T16=T16, T8=T8, T4=T4)                     # half: only gathered Ω cells enter the (float) cosine
    o['mu16'] = pred.model.no_mem_embed.detach().float().reshape(1, -1, 1, 1)
    return o


def build(pred, raw, max_targets=None, rng=None, dev='cuda', all_targets=False, feat=False, with_pixel=False):
    """GPU side of one clip, the test-time path: encode every frame once (+ the empty scene); kernel-f16 tracker
    (adapter/tracker.kernel_track on 1 − cos(F_t, F_B)) from the jittered t0 box; for each target t, x-only Theil–Sen
    on the track over [t-12, t] aligns frames t-n_h..t-1 into V̂'s gated integral image (integral_image_and_mass with the clip's
    τ, z0); the integral images are encoded. Each frame's F_t is reused as its own target input, so the encode cost per
    sample is ≈ (T + K + 1) / K. Returns (f, meta).
    feat=True (learned feature integral, adapter/feat_integral.py): no pixel integral image / no enc(C); instead the cached
    frame features of each target's history, shifted onto t (K16/8/4, Kvalid16, age), the targets T16/8/4 = features of the
    noise-matched occluder-free frame Xt, and the hidden-recoverable cells Om16/8/4 (see _feat_inputs). with_pixel=True
    additionally keeps the pixel integral's features as P16/8/4 and its mass as a_pix (G1 comparison)."""
    from concurrent.futures import ThreadPoolExecutor
    from benchmark.nfo_vos.adapter.tracker import kernel_track
    X, bg, occ = raw['X'], raw['bg'], raw['occ']
    T, H, W = X.shape
    with torch.autocast('cuda', dtype=AMP):
        F16, F8, F4 = _enc(pred, X)
        B16, B8, B4 = _enc(pred, bg[None])
    cx0, cy0, h0 = raw['bxj'][0]
    S0 = raw['S'][0] if raw['S'][0].any() else raw['S'][int(np.argmax(raw['S'].sum((1, 2))))]
    ys, xs = np.nonzero(S0); asp0 = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
    if 'track' in raw:                     # given track (apes: GT box centres + noise) instead of the kernel-f16 tracker
        trk = raw['track']
    else:
        dmap = 1 - torch.nn.functional.cosine_similarity(F16.float(), B16.float(), dim=1)
        dmap = torch.nn.functional.interpolate(dmap[:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0].cpu().numpy()
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
        hist, Ms = hist_warps(trk, t, nh, occ.get('shift_xy', False))
        return integral_image_and_mass([X[i].astype(np.float32) for i in hist], Ms, bg.astype(np.float32), occ['tau'], occ['z0'])
    pix = not feat or with_pixel
    if pix:
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
         'B16': B16.float().expand(n, -1, -1, -1), 'B8': B8.float().expand(n, -1, -1, -1), 'B4': B4.float().expand(n, -1, -1, -1),
         'V': torch.as_tensor(np.stack([a256(raw['V'][t]) for t in ts]), device=dev),
         'W': torch.as_tensor(np.stack([loss_weights(raw['S'][t], raw['V'][t]) for t in ts]), device=dev),
         'box': torch.as_tensor(np.stack(boxes), device=dev),
         'mode': torch.as_tensor(raw['modes'][ts].astype(np.float32), device=dev)}
    if pix:
        cf = {'C16': C16.float(), 'C8': C8.float(), 'C4': C4.float(),
              'a': torch.as_tensor(np.stack([a256(a) for _, a in CA]), device=dev)}
        f.update({('P' + k[1:] if k[0] == 'C' else 'a_pix'): v for k, v in cf.items()} if feat else cf)
    if feat:
        f.update(_feat_inputs(pred, raw, ts, trk, nh, (F16, F8, F4), dev))
    f['pt'] = torch.stack([(f['box'][:, 0] + f['box'][:, 2]) / 2, (f['box'][:, 1] + f['box'][:, 3]) / 2], 1)
    f['trk'] = torch.as_tensor(trk[ts] * sc[:2], device=dev).float()        # tracker position, 1024 frame
    hb = np.array([asp0 * h0 / 2, h0 / 2], np.float32)                     # tracker box: t0 size, tracker centre
    f['box_trk'] = torch.as_tensor(np.concatenate([trk[ts] - hb, trk[ts] + hb], 1) * sc, device=dev).float()
    Vn = [raw['V'][t] for t in ts]
    A_ = f.get('a', f.get('a_pix'))
    on = [float(np.asarray(A_[i].cpu())[a256(Vn[i]) > 0.5].mean()) if Vn[i].any() else np.nan for i in range(n)] if A_ is not None else []
    meta = dict(Vn=Vn, ts=ts, cid=raw['cid'], occ=occ, mode=raw['modes'][ts], trk=trk, a_on=on,
                a_mass=[float(x) for x in A_.sum((1, 2)).cpu()] if A_ is not None else [])
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
def encode(pred, clips, dev='cuda', feat=False):
    """Frozen SAM2 (bf16) on a list of clip batches: features of every target frame X, its integral image C, and each clip's
    empty scene B (encoded once per clip, broadcast to its frames). Returns one flat batch.
    feat=True (one target per clip, NFO hand items with 'hist' / 'dx1024'): no C encode; instead the history keys
    K16/8/4 shifted onto the target (feat_integral.shift_keys), Kvalid16, age, phase — the inputs of with_feat."""
    rgb = lambda a: [np.stack([x] * 3, -1) for x in a]
    f = {}

    def run(imgs):
        pred.set_image_batch(rgb(imgs))
        hr = pred._features['high_res_feats']
        return pred._features['image_embed'].float(), hr[1].float(), hr[0].float()

    with torch.autocast('cuda', dtype=AMP):
        for key, src in (('X', 'X'),) + ((('C', 'Ct'),) if not feat else ()):
            parts = [run(c[src]) for c in clips]
            for i, l in enumerate(('16', '8', '4')):
                f[f'{key}{l}'] = torch.cat([p[i] for p in parts])
        bparts = run([c['bg'] for c in clips])
        if feat:
            from benchmark.nfo_vos.adapter.feat_integral import shift_keys
            hp = [run(c['hist']) for c in clips]
            dx = torch.as_tensor(np.concatenate([c['dx1024'] for c in clips]), device=dev)
            for i, l in enumerate(('16', '8', '4')):
                f[f'K{l}'], v = shift_keys(torch.stack([p[i] for p in hp]), dx, int(l))
                if l == '16':
                    f['Kvalid16'] = v
            nh = dx.shape[1]
            f['age'] = torch.arange(nh, 0, -1, device=dev).repeat(len(clips), 1)
            f['phase'] = dx / 16 - torch.round(dx / 16)
        idx = torch.cat([torch.full((len(c['X']),), j) for j, c in enumerate(clips)]).to(dev)
        for i, l in enumerate(('16', '8', '4')):
            f[f'B{l}'] = bparts[i][idx]
    for k in ('a', 'V', 'box', 'mode') + tuple(k for k in ('trk', 'box_trk') if all(k in c for c in clips)):
        f[k] = torch.as_tensor(np.concatenate([c[k] for c in clips])).to(dev).float()
    f['pt'] = torch.stack([(f['box'][:, 0] + f['box'][:, 2]) / 2, (f['box'][:, 1] + f['box'][:, 3]) / 2], 1)
    return f
