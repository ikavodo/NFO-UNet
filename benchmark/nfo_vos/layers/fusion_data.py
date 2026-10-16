"""Per-clip online batches for approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11). Nothing is
cached: each clip is regenerated from its seed in a DataLoader worker; SAM2 encodes on the GPU (encode()).

Per clip (sample_occlusion): person/scene scale s ~ U[0.5, 1] (zoom-out; GPJATK person 0.53 of frame height, NFO
0.33), ρ = 0 (10%) or ρ ~ U[0.2, 0.6], clump ∈ {12, 24, (12, 24)}, flutter ~ U[0, 2]·noise.FLUTTER_AMP.
Tracker stand-in: GT boxes jittered by σ_a = 3 px. Corruption, one mode per target frame:
  0 none (75%) · 1 history shift error, 5σ_a (10%) · 2 composite = empty scene, history before entry (7.5%)
  3 tracker on a neighbour: box and alignment offset sideways by U[0.5, 1]·box width (7.5%)
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

P_MODES = (0.75, 0.10, 0.075, 0.075)
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


def clip_batch(spec, rng, frames=None):
    """All valid target frames of one clip (or `frames` of them), stacked. Box in SAM2's 1024 input frame."""
    seq, cam, _, seed = spec
    occ = sample_occlusion(rng)
    d = rescale_clip(D.make_clip(seq, cam, occ['rho'], seed, flutter=occ['flutter'], clump=occ['clump']), occ['scale'])
    H, W = d['bg'].shape
    sc = np.array([1024 / W, 1024 / H] * 2, np.float32)
    bxj = D.jitter(d['boxes'], SIGMA_A, rng)
    ts = [t for t in D.valid_targets(d['boxes']) if d['V'][t].any()]
    if frames is not None and len(ts) > frames:
        ts = sorted(rng.choice(ts, frames, replace=False))
    A = {k: [] for k in ('X', 'Ct', 'a', 'V', 'Vn', 'box', 'mode')}
    for t in ts:
        mode = int(rng.choice(4, p=P_MODES))
        S = d['S'][t]
        ys, xs = np.nonzero(S)
        asp = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        bt = bxj.copy()
        if mode == 3:                                   # tracker on a neighbour: everything offset consistently
            bt[:, 0] += rng.choice([-1, 1]) * rng.uniform(0.5, 1.0) * asp * bxj[t, 2]
        cx, cy, h = bt[t]
        box = np.array([cx - asp * h / 2, cy - h / 2, cx + asp * h / 2, cy + h / 2], np.float32)
        hist = range(t - D.N_HIST + 1, t)
        if mode == 2:
            C, a = d['bg'], np.zeros_like(d['bg'])
        else:
            bh = D.jitter(d['boxes'], 5 * SIGMA_A, rng) if mode == 1 else bt
            C, a = composite_and_mass([d['X'][i] for i in hist], [D.sim(bh[i], bt[t]) for i in hist], d['bg'])
        for k, v in (('X', u8(d['X'][t])), ('Ct', u8(C)), ('a', a256(a)), ('V', a256(d['V'][t])), ('Vn', d['V'][t]),
                     ('box', box * sc), ('mode', mode)):
            A[k].append(v)
    out = {k: (v if k == 'Vn' else np.stack(v)) for k, v in A.items()}
    out.update(bg=u8(d['bg']), cid=d['cid'], occ=occ)
    return out


class Clips(torch.utils.data.IterableDataset):
    """train: endless random clips, FRAMES per clip, per-worker seed stream. val: the fixed clip set, all frames,
    occlusion drawn deterministically per clip."""
    FRAMES = 12

    def __init__(self, split, seed=0):
        self.split, self.seed = split, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        S = specs(self.split)
        if self.split == 'train':
            rng = np.random.default_rng([self.seed, wid])
            while True:
                yield clip_batch(S[rng.integers(len(S))], rng, self.FRAMES)
        else:
            for k in range(wid, len(S), nw):
                yield clip_batch(S[k], np.random.default_rng([1, k]))


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
    for k in ('a', 'V', 'box', 'mode'):
        f[k] = torch.as_tensor(np.concatenate([c[k] for c in clips])).to(dev).float()
    f['pt'] = torch.stack([(f['box'][:, 0] + f['box'][:, 2]) / 2, (f['box'][:, 1] + f['box'][:, 3]) / 2], 1)
    return f
