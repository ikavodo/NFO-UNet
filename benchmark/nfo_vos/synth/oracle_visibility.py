"""Oracle kill test for feature-space layers (architecture v3; pre-registered there, 2026-10-15).

Clips: the 24 held-out 'perlin' clips of layered_experiment.py (occluder ≡ background, clump 24, ρ 0.45).
Oracle features (SAM2.1 b+ image encoder, the image predictor's _features; sam2_image_predictor.py:127):
  F_t = Enc(X_t) (occluded frame), F_obj = Enc(clean frame t = person planted on B, no occluder), F_bg = Enc(B).
  B is exact here (static scene, no noise), so this is an upper bound on any learned background.
Score per level L ∈ {stride 16 (image_embed, 256 ch), 8 (64 ch), 4 (32 ch)}:
  s_L = cos(F_t, F_obj) − cos(F_t, F_bg), on the feature grid, bilinearly upsampled to 540×960.
AUC (Mann–Whitney) of s_L for visible person pixels (V) vs occluded person pixels (S ∖ V), per size bin of V's
connected components. Cells use SAM2's true anisotropic cell: the 1024×1024 square resize
(sam2/utils/transforms.py:32) makes a 16-px cell (W/64) × (H/64) native px = 15 × 8.44 here.
PRE-REGISTERED: stride-16 AUC on sub-cell fragments < 0.7 → stride 16 cannot carry the fix (go to levels 8 / 4);
all levels < 0.7 → stop the feature-space route.
Side check: re-bin T1 recall by fragment size with the true cell area, on NFO (handAv) and on these clips.
"""
import glob
import json

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R
from benchmark.nfo_vos.synth.gpjatk_foliage import load_person

ROOT = 'data/synthetic/gpjatk_foliage/layered/perlin'
BINS = ((0, .5), (.5, 1), (1, 4), (4, np.inf))
CELL_SYN = (960 / 64) * (540 / 64)
CELL_NFO = (800 / 64) * (600 / 64)
FRAMES = range(1, 40, 3)


def auc(pos, neg, rng, k=20000):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos, neg = rng.choice(pos, min(k, len(pos))), rng.choice(neg, min(k, len(neg)))
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def feats(pred, g):
    pred.set_image(np.stack([np.clip(g, 0, 255).astype(np.uint8)] * 3, -1))
    f = pred._features
    return [f['image_embed'][0], f['high_res_feats'][1][0], f['high_res_feats'][0][0]]   # strides 16, 8, 4


def cosmap(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0)


def up(x):
    return torch.nn.functional.interpolate(x[None, None], (540, 960), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def binned(M, h, cell, out):
    n, L, cs, _ = cv2.connectedComponentsWithStats(h.astype(np.uint8), connectivity=8)
    for c in range(1, n):
        z = cs[c, 4] / cell
        out[[lo <= z < hi for lo, hi in BINS].index(True)].append((M & (L == c)).sum() / cs[c, 4])


def bin_ids(v, cell=CELL_SYN):
    """Per-pixel fragment-size bin (index into BINS) of v's 8-connected components; −1 outside v."""
    n, Lab, cs, _ = cv2.connectedComponentsWithStats(v.astype(np.uint8), connectivity=8)
    binid = np.full(v.shape, -1)
    for c in range(1, n):
        binid[Lab == c] = [lo <= cs[c, 4] / cell < hi for lo, hi in BINS].index(True)
    return binid


def main():
    rng = np.random.default_rng(0)
    pred = R.image_predictor()
    pos = {(L, b): [] for L in range(3) for b in range(4)}; neg = {L: [] for L in range(3)}
    alone = {k: {'pos': [], 'neg': []} for k in ('obj', 'bg')}
    syn_bins = [[] for _ in BINS]
    for cd in sorted(glob.glob(f'{ROOT}/*')):
        m = json.load(open(f'{cd}/meta.json'))
        I, sil, _ = load_person(m['seq'], m['cam'], 40)
        B = cv2.imread(f'{cd}/bg.png', 0).astype(np.float32)
        V, S = np.load(f'{cd}/V.npy'), np.load(f'{cd}/S.npy')
        M1 = np.load(f'{cd}/masks_t1.npz')['masks']
        with torch.inference_mode():
            Fb = feats(pred, B)
            for t in FRAMES:
                binned(M1[t], V[t], CELL_SYN, syn_bins)
                mf = cv2.GaussianBlur(sil[t].astype(np.float32), (0, 0), 1.0)
                Ft = feats(pred, cv2.imread(f'{cd}/frames/{t:05d}.jpg', 0).astype(np.float32))
                Fo = feats(pred, mf * I[t] + (1 - mf) * B)
                occl = S[t] & ~V[t]
                binid = bin_ids(V[t])
                for L in range(3):
                    co, cb = up(cosmap(Ft[L], Fo[L])), up(cosmap(Ft[L], Fb[L]))
                    s = co - cb
                    neg[L].append(s[occl])
                    for b in range(4):
                        pos[(L, b)].append(s[binid == b])
                    if L == 0:
                        alone['obj']['pos'].append(co[V[t]]); alone['obj']['neg'].append(co[occl])
                        alone['bg']['pos'].append(-cb[V[t]]); alone['bg']['neg'].append(-cb[occl])
        print(cd.split('/')[-1], flush=True)
    names = ('stride 16', 'stride 8', 'stride 4')
    print('\nAUC of s = cos(F_t,F_obj) − cos(F_t,F_bg): visible (by fragment size, true SAM2 cells) vs occluded person')
    for L in range(3):
        ng = np.concatenate(neg[L])
        a = [auc(np.concatenate(pos[(L, b)]), ng, rng) for b in range(4)]
        al = auc(np.concatenate([np.concatenate(pos[(L, b)]) for b in range(4)]), ng, rng)
        print(f'  {names[L]:9s} all {al:.3f} | ' + ' '.join(f'[{lo},{hi}) {x:.3f}' for (lo, hi), x in zip(BINS, a)))
    for k in alone:
        print(f'  stride 16, {k} term alone: AUC {auc(np.concatenate(alone[k]["pos"]), np.concatenate(alone[k]["neg"]), rng):.3f}')
    print('\nT1 recall by fragment size with true cells: synthetic perlin held-out ' +
          ' / '.join(f'{np.mean(b):.2f} (n {len(b)})' for b in syn_bins))
    from benchmark.nfo_vos.learned_thresh import frames, load
    from benchmark.nfo_vos.score_void import lab
    nb, shares = [[] for _ in BINS], np.zeros(4)
    for s_, k, *_ in frames():
        if s_ == 'seq1':
            continue
        h, keep = lab(s_, k)
        binned(load(s_, k)['T1'], h, CELL_NFO, nb)
        n, Lb, cs, _ = cv2.connectedComponentsWithStats(h.astype(np.uint8), connectivity=8)
        for c in range(1, n):
            shares[[lo <= cs[c, 4] / CELL_NFO < hi for lo, hi in BINS].index(True)] += cs[c, 4]
    print('T1 recall by fragment size with true cells: NFO seq2–4 handAv ' + ' / '.join(f'{np.mean(b):.2f} (n {len(b)})' for b in nb))
    print('NFO label area share by bin, true cells: ' + ' / '.join(f'{x:.3f}' for x in shares / shares.sum()))


if __name__ == '__main__':
    main()
