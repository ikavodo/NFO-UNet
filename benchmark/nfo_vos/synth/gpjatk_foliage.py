"""GPJATK walkers behind photographed foliage, with exact layers (plan: docs/layer_sam2/plan_gpjatk_layer_sam2.md).

Per clip: GPJATK frame I^g_t (person in the lab scene, used as is), binary occluder matte from master_thesis
src/occluders.occ_perlin (thresholded bicubic-upsampled random grid: leaf clumps with small gaps; `density` = covered
fraction, exact up to quantisation), softened to a_t ∈ [0, 1] by a Gaussian blur, and filled with grayscale foliage
O cropped from the template photo (rows 0 … 0.6·H, 16:9, resized to 960×540). occ_perlin's own light-canvas
intensities are NOT used: the matte only selects where the photo shows.
  I_t = a_t·O + (1 − a_t)·I^g_t,   V_t = m_t ∧ (a_t < 0.5),   S_t = m_t,   B̂_t = a_t·O + (1 − a_t)·B
Foliage is static (movement 0) in this first version, as NFO's camera and tree are close to static.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

MT = os.path.abspath('../master_thesis')
GP = f'{MT}/GPJATK/Sequences'
TEMPLATE = 'images/veronica111886-yakushima-2336735.jpg'
OUT = 'data/synthetic/gpjatk_foliage'
H, W = 540, 960
HOLDOUT = {f'p{i}' for i in range(27, 33)}          # 6 subjects never used before the final test split
WALK_REL = 0.5    # deviation: the testbed's absolute 0.12 (em_severe_occlusion_testbed.py:61) suits c1/c3 only; c2/c4
                  # walk towards the camera and change 5–10 % per frame, so threshold at 0.5 × the clip's 90th pct


def occ_perlin(*a, **k):
    if MT not in sys.path:
        sys.path.insert(0, MT)
    from src.occluders import occ_perlin as f        # master_thesis src/occluders.py:314
    return f(*a, **k)


def read_avi_gray(p):
    c, fr = cv2.VideoCapture(p), []
    while True:
        ok, f = c.read()
        if not ok:
            return np.stack(fr)
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))


def walking(sil):
    """Walking interval by silhouette differencing (em_severe_occlusion_testbed.py:55-63, relative threshold)."""
    d = np.array([np.logical_xor(sil[t], sil[t - 1]).sum() for t in range(1, len(sil))])
    f = d / np.maximum(sil[1:].reshape(len(sil) - 1, -1).sum(1), 1)
    idx = np.where(f > WALK_REL * np.percentile(f, 90))[0] + 1
    return int(idx.min()), int(idx.max())


def foliage(rng):
    t = cv2.imread(TEMPLATE, 0)
    t = t[:int(0.6 * t.shape[0])]
    cw = int(rng.uniform(960, min(2667, t.shape[1], t.shape[0] * 16 / 9)))
    ch = int(cw * 9 / 16)
    x, y = rng.integers(0, t.shape[1] - cw + 1), rng.integers(0, t.shape[0] - ch + 1)
    return cv2.resize(t[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_AREA).astype(np.float32), \
        dict(crop=[int(x), int(y), cw, ch])


def make_clip(seq, cam, density, seed, n=40, clump=12, blur=1.0, out=OUT, back='lab', fill='photo'):
    """back='lab': GPJATK frame as is. back='photo': person cut out (silhouette feathered by σ = 1) and planted in
    front of a second, independent crop of the foliage photo (one-variable change vs 'lab')."""
    rng = np.random.default_rng(seed)
    d = f'{GP}/{seq}'
    I = read_avi_gray(glob.glob(f'{d}/Images/{cam}_*.avi')[0]).astype(np.float32)
    sil = np.stack([cv2.imread(p, 0) > 127 for p in sorted(glob.glob(f'{d}/Silhouettes/{cam}/*.png'))])
    assert len(sil) == len(I), (seq, cam, len(sil), len(I))
    s0, s1 = walking(sil)
    s0 = min(s0, max(0, s1 - n + 1))
    I, m = I[s0:s0 + n], sil[s0:s0 + n]
    B = np.median(read_avi_gray(f'{d}/Background/{cam}.avi'), 0).astype(np.float32)
    O, fmeta = foliage(rng)
    if back == 'photo':
        Bf, bmeta = foliage(rng)
        mf = np.stack([cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0) for x in m])
        I = mf * I + (1 - mf) * Bf
        B = Bf
        fmeta['back_crop'] = bmeta['crop']
    _, occ = occ_perlin(torch.zeros(len(I), 1, H, W), density=density, noise_h=clump, noise_w=clump,
                        movement_x=0, movement_y=0, light_movement_x=0, light_movement_y=0,
                        seed=int(seed), return_mask=True)
    a = np.stack([cv2.GaussianBlur(o.numpy().astype(np.float32), (0, 0), blur) for o in occ])
    # fill='photo': occluder shows a separate foliage texture O. fill='back': occluder = the scene background itself
    # (MovingMNIST-OcclusionBench occluders.py:450 rule): occluded person pixels show B, so occluder ≡ background.
    X = a * O + (1 - a) * I if fill == 'photo' else a * B + (1 - a) * I
    V, S = m & (a < 0.5), m
    cid = f'{seq}_{cam}_d{int(round(density * 100))}_s{seed}'
    cd = f'{out}/{cid}'
    os.makedirs(f'{cd}/frames', exist_ok=True)
    for t, x in enumerate(X):
        cv2.imwrite(f'{cd}/frames/{t:05d}.jpg', np.stack([np.clip(x, 0, 255).astype(np.uint8)] * 3, -1),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(f'{cd}/V.npy', V); np.save(f'{cd}/S.npy', S); np.save(f'{cd}/occ.npy', a >= 0.5)
    cv2.imwrite(f'{cd}/bg.png', np.clip(a[0] * O + (1 - a[0]) * B, 0, 255).astype(np.uint8))
    ys, xs = np.nonzero(S[0])
    dt = cv2.distanceTransform(V[0].astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(dt), dt.shape)
    meta = dict(seq=seq, cam=cam, frames=[s0, s0 + n - 1], density=density, clump=clump, blur=blur, seed=int(seed), back=back, fill=fill,
                occluder='occ_perlin', coverage_person=float(1 - V.sum() / max(S.sum(), 1)),
                coverage_frame=float((a >= 0.5).mean()), split='holdout' if seq.split('s')[0] in HOLDOUT else 'train',
                box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], point=[int(px), int(py)], **fmeta)
    json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
    return cid, X, V, S


def montage(clips, path):
    """Columns: I_t | V (green) and S − V (red) over I_t, at the middle frame of each clip."""
    rows = []
    for cid, X, V, S in clips:
        t = len(X) // 2
        g = np.clip(X[t], 0, 255).astype(np.uint8)
        o = np.stack([g] * 3, -1).copy()
        o[V[t]] = (0.5 * o[V[t]] + [0, 127, 0]).astype(np.uint8)
        o[S[t] & ~V[t]] = (0.5 * o[S[t] & ~V[t]] + [0, 0, 127]).astype(np.uint8)
        r = np.concatenate([np.stack([g] * 3, -1), o], 1)
        cv2.putText(r, cid, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        rows.append(cv2.resize(r, (960, 270)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.concatenate(rows, 0))


# ----------------------------------------------------------------------------------------------------------------
# Layered clips (2026-10-15): occluder ≡ background, depth-consistent masks from the photo itself.
# ----------------------------------------------------------------------------------------------------------------
DISP = 'data/synthetic/foliage_depth/veronica_disp.npy'     # photo_depth.py (Depth Anything V2 L; larger = nearer)
_TPL = {}


def template():
    if not _TPL:
        t = cv2.imread(TEMPLATE, 0)
        _TPL['g'], _TPL['d'] = t[:int(0.6 * t.shape[0])].astype(np.float32), np.load(DISP).astype(np.float32)
    return _TPL['g'], _TPL['d']


def foliage_pair(rng):
    """Random 16:9 crop (width 960…2667 px of the template = leaf scale), random h-flip: grayscale + disparity."""
    g, d = template()
    cw = int(rng.uniform(960, min(2667, g.shape[1], g.shape[0] * 16 / 9))); ch = int(cw * 9 / 16)
    x, y = rng.integers(0, g.shape[1] - cw + 1), rng.integers(0, g.shape[0] - ch + 1)
    G = cv2.resize(g[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_AREA)
    D = cv2.resize(d[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        G, D = G[:, ::-1].copy(), D[:, ::-1].copy()
    return G, D, [int(x), int(y), cw, ch]


def load_person(seq, cam, n):
    d = f'{GP}/{seq}'
    I = read_avi_gray(glob.glob(f'{d}/Images/{cam}_*.avi')[0]).astype(np.float32)
    sil = np.stack([cv2.imread(p, 0) > 127 for p in sorted(glob.glob(f'{d}/Silhouettes/{cam}/*.png'))])
    assert len(sil) == len(I), (seq, cam, len(sil), len(I))
    s0, s1 = walking(sil)
    s0 = min(s0, max(0, s1 - n + 1))
    return I[s0:s0 + n], sil[s0:s0 + n], s0


def labels(S, occ, open_px=7, zone_px=15):
    """Human-convention label (user's void definition, fixed 2026-10-15 before any scoring):
    coarse = opening(occ, open_px disk): occluder parts wider than ≈ half a 15-px cell (annotator removes them);
    fine = occ ∧ ¬coarse (thin foliage); zone = closing(fine, zone_px disk) ∪ dilate(fine, 3×3): where thin foliage
    is dense or adjacent, the annotator cannot tell (handAv marks such regions void, incl. visible pixels in them).
    H = S ∧ ¬coarse ∧ ¬zone, void = S ∧ ¬coarse ∧ zone (ignored when scoring)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    kz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zone_px, zone_px))
    o = occ[0].astype(np.uint8)                                     # static scene: one mask for all frames
    coarse = cv2.morphologyEx(o, cv2.MORPH_OPEN, k).astype(bool)
    fine = (o > 0) & ~coarse
    zone = cv2.morphologyEx(fine.astype(np.uint8), cv2.MORPH_CLOSE, kz).astype(bool) | \
        cv2.dilate(fine.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return S & ~coarse & ~zone, S & ~coarse & zone


def make_layered_clip(seq, cam, rho, seed, mode, out, n=40, K=3, clump=24, blur=0.7, twigs=0, twig_gain=0.6,
                      save=True):
    """mode 'perlin': B = one photo crop, mask = occ_perlin(density rho, clump) (baseline, fill='back').
    mode 'depth':  B = one photo crop, mask = [disp_B > q_{1−rho}(disp_B over the person's trajectory box)]:
                   the person stands at the depth where a fraction rho of the trajectory's scene is nearer.
    mode 'stack':  dead-leaves stacking (Lee, Mumford & Huang, IJCV 2001): layer 0 = full crop (far), layers 1…K =
                   independent crops cut to their own nearest foliage, mask_k = [disp_k > q_{1−ρ_k}], ρ_k =
                   1 − (1 − rho)^{1/K} so that the union covers ≈ rho; B = painter's composite; person between layer 0
                   and layer 1, so a = ∪_k mask_k.
    twigs > 0: a twig layer drawn into the scene in front of the person: occ_branch lines (1–2 px, master_thesis
                   src/occluders.py:231, mask only), intensity = twig_gain × an independent photo crop (assumed
                   darker than leaves; not calibrated).
    In all modes occluded person pixels show B itself (MovingMNIST-OcclusionBench occluders.py:450), static scene."""
    rng = np.random.default_rng(seed)
    I, m, s0 = load_person(seq, cam, n)
    ys, xs = np.nonzero(m.any(0))
    R = np.zeros((H, W), bool); R[max(0, ys.min() - 10):ys.max() + 11, max(0, xs.min() - 10):xs.max() + 11] = True
    G, D, crop = foliage_pair(rng); crops = [crop]
    if mode == 'perlin':
        _, o = occ_perlin(torch.zeros(1, 1, H, W), density=rho, noise_h=clump, noise_w=clump, movement_x=0,
                          movement_y=0, light_movement_x=0, light_movement_y=0, seed=int(seed), return_mask=True)
        a_bin, B = o[0].numpy().astype(bool), G
    elif mode == 'depth':
        a_bin, B = D > np.quantile(D[R], 1 - rho), G
    elif mode == 'stack':
        B, a_bin, rk = G.copy(), np.zeros((H, W), bool), 1 - (1 - rho) ** (1 / K)
        for _ in range(K):
            Gk, Dk, ck = foliage_pair(rng); crops.append(ck)
            mk = Dk > np.quantile(Dk[R], 1 - rk)
            B[mk] = Gk[mk]; a_bin |= mk
    else:
        raise ValueError(mode)
    if twigs:
        if MT not in sys.path:
            sys.path.insert(0, MT)
        from src.occluders import occ_branch
        _, ob = occ_branch(torch.zeros(1, 1, H, W), num_branches=int(twigs), thickness=2, movement_x=0,
                           light_movement_x=0, light_movement_y=0, seed=int(seed) + 7, return_mask=True)
        tw = ob[0].numpy().astype(bool)
        Gt, _, ct = foliage_pair(rng); crops.append(ct)
        B = B.copy(); B[tw] = twig_gain * Gt[tw]; a_bin = a_bin | tw
    a = cv2.GaussianBlur(a_bin.astype(np.float32), (0, 0), blur)
    mf = np.stack([cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0) for x in m])
    X = a * B + (1 - a) * (mf * I + (1 - mf) * B)
    occ = np.repeat((a >= 0.5)[None], len(I), 0)
    S = m; V = S & ~occ
    Hl, void = labels(S, occ)
    cid = f'{seq}_{cam}_{mode}_r{int(round(rho * 100))}_t{int(twigs)}_s{seed}'
    if not save:
        return cid, X, V, S, Hl, void
    cd = f'{out}/{cid}'
    os.makedirs(f'{cd}/frames', exist_ok=True)
    for t, x in enumerate(X):
        cv2.imwrite(f'{cd}/frames/{t:05d}.jpg', np.stack([np.clip(x, 0, 255).astype(np.uint8)] * 3, -1),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    for k, v in (('V', V), ('S', S), ('occ', occ), ('H', Hl), ('void', void)):
        np.save(f'{cd}/{k}.npy', v)
    cv2.imwrite(f'{cd}/bg.png', np.clip(B, 0, 255).astype(np.uint8))
    yy, xx = np.nonzero(S[0])
    dt = cv2.distanceTransform(V[0].astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(dt), dt.shape)
    meta = dict(seq=seq, cam=cam, frames=[s0, s0 + n - 1], mode=mode, rho=rho, K=K, twigs=int(twigs), twig_gain=twig_gain, clump=clump, blur=blur,
                seed=int(seed), crops=crops, coverage_person=float(1 - V.sum() / max(S.sum(), 1)),
                split='holdout' if seq.split('s')[0] in HOLDOUT else 'train',
                box=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())], point=[int(px), int(py)])
    json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
    return cid, X, V, S, Hl, void
