"""Online samples for the prompt adapter (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §4b). Nothing is cached:
clips are regenerated from their seed (dataset.make_clip, ~1.3 s CPU) in DataLoader workers, and SAM2 encodes each batch
on the GPU (encode()). Replaces the 2026-10-16 feature cache (224 MB per clip; deleted at the user's request).

Stopgap data path until the data session delivers §3: flutter nuisance, dataset.valid_targets frames, jittered GT boxes
as the tracker (σ_a 3 px). Corruption, one mode per frame (flagged): 0 none (75%) · 1 history shift error, 5σ_a (10%)
· 2 C̃ = bg, the history-before-entry case (7.5%) · 3 point on a non-person pixel inside the box (7.5%).
One sample: X (frame t), Ct (V̂'s gated composite C_t; key name kept), a (its evidence mass Σ w G on the 256² grid), bg, uint8 [H, W]; V, S, SO soft targets on SAM2's 256² grid;
Vn native V; pt [2], box [4] in SAM2's 1024 input frame; mode.
"""
import cv2
import numpy as np
import torch

from benchmark.nfo_vos.layers import dataset as D
from benchmark.nfo_vos.layers.prompt_oracle import SIGMA_A, composite_and_mass
from benchmark.nfo_vos.synth import noise

P_MODES = (0.75, 0.10, 0.075, 0.075)


def specs(split):
    return D.clip_specs('train') if split == 'train' else D.clip_specs('val')[::4]


def a256(m):
    return cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float16)


def u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def clip_samples(spec, rng, frames=None):
    """All (or `frames` randomly chosen) valid target frames of one clip, built exactly as at test time."""
    d = D.make_clip(*spec, flutter=noise.FLUTTER_AMP)
    H, W = d['bg'].shape
    to1024 = np.array([1024 / W, 1024 / H], np.float32)
    bxj = D.jitter(d['boxes'], SIGMA_A, rng)
    ts = [t for t in D.valid_targets(d['boxes']) if d['V'][t].any()]
    if frames is not None and len(ts) > frames:
        ts = sorted(rng.choice(ts, frames, replace=False))
    out = []
    for t in ts:
        V, S = d['V'][t], d['S'][t]
        mode = int(rng.choice(4, p=P_MODES))
        cx, cy, h = bxj[t]
        ys, xs = np.nonzero(S)
        w = h * (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)
        pt = np.array([cx, cy], np.float32)
        if mode == 3:
            x0, y0, x1, y1 = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1]).astype(int)
            cand = np.argwhere(~S[y0:y1 + 1, x0:x1 + 1])
            if len(cand):
                yy, xx = cand[rng.integers(len(cand))]
                pt = np.array([x0 + xx, y0 + yy], np.float32)
        hist = range(t - D.N_HIST + 1, t)
        if mode == 2:
            Ct, a = d['bg'], np.zeros_like(d['bg'])
        else:
            bh = D.jitter(d['boxes'], 5 * SIGMA_A, rng) if mode == 1 else bxj
            C, a = composite_and_mass([d['X'][i] for i in hist], [D.sim(bh[i], bxj[t]) for i in hist], d['bg'])
            Ct = C       # V̂'s gated composite, as at NFO test time. C̃ = a·C + (1 − a)·bg (plan §2) erased the person
                         # on synthetic clips (|C̃ − bg| in S 18 vs 9 off S; C: 47 vs 41): ablation only, 2026-10-16
        O = D.observable(d['S'], d['V'], d['boxes'], t)
        out.append(dict(X=u8(d['X'][t]), Ct=u8(Ct), a=a256(a), bg=u8(d['bg']), V=a256(V), S=a256(S), SO=a256(S & O), Vn=V,
                        pt=pt * to1024, box=box * np.tile(to1024, 2), mode=mode, cid=d['cid'], t=t))
    return out


class Online(torch.utils.data.IterableDataset):
    """train: endless, each worker draws random clips (FRAMES_PER_CLIP frames each) with its own seed stream.
    val: the fixed O1 clip set, all frames, deterministic."""
    FRAMES_PER_CLIP = 4

    def __init__(self, split, seed=0):
        self.split, self.seed = split, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        S = specs(self.split)
        if self.split == 'train':
            rng = np.random.default_rng([self.seed, wid])
            while True:
                for s in clip_samples(S[rng.integers(len(S))], rng, self.FRAMES_PER_CLIP):
                    yield s
        else:
            for k in range(wid, len(S), nw):
                yield from clip_samples(S[k], np.random.default_rng([1, k]))


def collate(items):
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        out[k] = v if isinstance(v[0], str) or k == 'Vn' else torch.as_tensor(np.stack(v))
    return out


@torch.no_grad()
def encode(pred, b, dev='cuda'):
    """Frozen SAM2 on the GPU, batched: features of X, C̃, bg (+ no_mem_embed, as set_image_batch) and the point-only
    low-res logits l_C, l_B. Returns the tensors the adapter / decoder consume."""
    from benchmark.nfo_vos.layers.prompt_adapter import decode
    rgb = lambda a: [np.stack([x] * 3, -1) for x in a.numpy()]
    f = {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        uniq = sorted(set(b['cid']), key=b['cid'].index)            # one empty scene per clip: encode it once
        first = [b['cid'].index(c) for c in uniq]
        inv = torch.tensor([uniq.index(c) for c in b['cid']], device=dev)
        for key, imgs in (('X', b['X']), ('C', b['Ct']), ('B', b['bg'][first])):
            pred.set_image_batch(rgb(imgs))
            f[f'{key}16'] = pred._features['image_embed'].float()
            f[f'{key}4'], f[f'{key}8'] = (x.float() for x in pred._features['high_res_feats'])
        for k in ('B16', 'B4', 'B8'):
            f[k] = f[k][inv]
        for k in ('pt', 'box', 'V', 'S', 'SO', 'mode', 'a'):
            f[k] = b[k].to(dev).float()
        for key in ('C', 'B'):
            f[f'l{key}'] = decode(pred.model, dict(f, X16=f[f'{key}16'], X4=f[f'{key}4'], X8=f[f'{key}8']),
                                  box=False)[:, 0].float()
    return f


def show(spec=None, path='images/benchmark/gpjatk_foliage/example_segment.png'):
    """One train clip, 4 target frames: noisy frame X_t | composite C̃_t | empty scene bg | X_t with V (green), S \\ V
    (red = occluded person), box + point (yellow)."""
    spec = spec or specs('train')[0]
    rows = []
    for s in clip_samples(spec, np.random.default_rng(0), 4):
        g = lambda a: cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        H, W = s['X'].shape
        Sn = cv2.resize(s['S'].astype(np.float32), (W, H)) > 0.5
        ov = g(s['X']); ov[s['Vn']] = (0, 200, 0); ov[Sn & ~s['Vn']] = (0, 0, 255)
        sc = np.array([W / 1024, H / 1024] * 2)
        x0, y0, x1, y1 = (s['box'] * sc).astype(int)
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.circle(ov, tuple((s['pt'] * sc[:2]).astype(int)), 6, (0, 255, 255), -1)
        panes = [g(s['X']), g(s['Ct']), g(s['bg']), ov]
        for v, txt in zip(panes, (f"X_t  t={s['t']}  mode {s['mode']}", 'composite C~_t', 'empty scene', 'V green, S\\V red')):
            cv2.putText(v, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        rows.append(cv2.resize(np.hstack(panes), None, fx=0.25, fy=0.25))
    cv2.imwrite(path, np.vstack(rows))
    return s['cid']
