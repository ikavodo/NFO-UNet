"""docs/gt_conditioned_blend_plan.md, stages 1–3, vs the frozen silhouette label (2026-10-16, defaults fixed before running).

One variable vs the frozen label (extent_label_test.silhouette, align=True, θ = none, ∪ P): the same evidence
(G := P = SAM2-ceiling, N = disparity < 0.1 ∧ |frame − empty| < 8), the same exemplar bank ('clear') and aspect
shortlist; only alignment and blending change.
  Stage 1  ∂G = G \\ erode(G); ∂G_true = ∂G ∩ dilate(N, 3); G_in = G within 5 px of ∂G + 10% of the deep interior;
           N_out = N within 30 px (+ shift range) of the shortlisted exemplars; edge = ∂G_true; each set ≤ 3000 px.
  Stage 2  per exemplar and band (exemplar_gt.BANDS), dx ∈ [−0.25, 0.25]·box width (= SHIFTS + BAND_SHIFTS range),
           2 px steps, E = Σ_in relu(φ + m) + λ Σ_out relu(m − φ) + μ Σ_edge |φ| on the exemplar's SDF shifted by dx
           (exact for a rigid shift); assemble the band-shifted mask, recompute its SDF; keep the top K = 8 by E.
  Stage 3  LP (HiGHS) over convex weights w on the K SDFs with hinge slacks; φ = Σ wᵢ φᵢ.
  Label    L = {φ < 0} ∪ P.   SDF sign: edt(~M) − edt(M) (negative inside), pixels.
Defaults (plan): m = 2, λ = 1, μ = 0.5.  Variants: 'sum' (plan as written) and 'mean' (each term divided by its pixel
count, audit suggestion).  Arms: frozen, S2 uniform top-3 blend, S3 LP, SAM2-ceiling.
Scored vs the hand polygons (17 frames: TUNE 9 were used to freeze the old label, TEST 8 are held out for both) and
vs the brush masks (20); J, P, R, F2, DAVIS boundary F.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.energy_blend
"""
import glob
import json
import os

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt as edt
from scipy.optimize import linprog

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.davis_metrics import f_measure
from benchmark.nfo_vos.exemplar_gt import ASP, BANDS, DEPTH, EXCL, N_DIFF, N_DISP, SEQ, box, place, seg_of
from benchmark.nfo_vos.extent_label_test import TEST, TUNE, bank, scores, silhouette
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

M_, LAM, MU, K, CAP, R_FRAC = 2.0, 1.0, 0.5, 8, 3000, 0.25
OUT = f'{TR.RES}/energy_blend'
IMG = 'images/benchmark/rr_checks/energy_blend'
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def sdf(m):
    return (edt(~m) - edt(m)).astype(np.float32)


def sample(m, rng, cap=CAP):
    ys, xs = np.nonzero(m)
    if len(ys) > cap:
        i = rng.choice(len(ys), cap, replace=False); ys, xs = ys[i], xs[i]
    return ys, xs


def shift_x(a, dx, fill):
    out = np.full_like(a, fill)
    if dx >= 0:
        out[:, dx:] = a[:, :a.shape[1] - dx]
    else:
        out[:, :dx] = a[:, -dx:]
    return out


def band_rows(b):
    x0, y0, x1, y1 = b
    h = y1 - y0
    edges = [y0 + int(f * h) for f in BANDS]
    edges[0], edges[-1] = 0, 600                       # rows above / below the box join the outer bands
    return list(zip(edges[:-1], edges[1:]))


def constraints(k, bg, d, P, near_masks, R, rng):
    N = (d < N_DISP) & (np.abs(V.native(k, SEQ).astype(np.float32) - bg) < N_DIFF)
    G = P
    dG = G & ~cv2.erode(G.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dG_true = dG & cv2.dilate(N.astype(np.uint8), K3).astype(bool)
    inner = G & (edt(G) <= 5)
    deep = G & ~inner & (rng.random(G.shape) < 0.1)
    U = np.any(near_masks, 0).astype(np.uint8)
    reach = cv2.dilate(U, np.ones((2 * (30 + R) + 1, 2 * (30 + R) + 1), np.uint8)).astype(bool)
    return dict(inn=sample(inner | deep, rng), out=sample(N & reach, rng), edge=sample(dG_true, rng),
                dG_true=dG_true, dG_occ=dG & ~dG_true, N=N)


def energy(phi_at, C, norm):
    """phi_at(ys, xs) -> SDF values; C: constraint sets. Per-term sums (plan) or means."""
    t = []
    for key, f in (('inn', lambda p: np.maximum(p + M_, 0)), ('out', lambda p: LAM * np.maximum(M_ - p, 0)),
                   ('edge', lambda p: MU * np.abs(p))):
        ys, xs = C[key]
        v = f(phi_at(ys, xs)).sum() if len(ys) else 0.0
        t.append(v / max(len(ys), 1) if norm else v)
    return sum(t)


def align(c, b, C, R, norm):
    """Band-wise shift search on the exemplar's SDF; returns (E, assembled mask)."""
    M0 = place(c, b, 0.0)
    if not M0.any():
        return None
    phi0 = sdf(M0)
    total, out = 0.0, np.zeros_like(M0)
    for r0, r1 in band_rows(b):
        sub = {key: tuple(a[(C[key][0] >= r0) & (C[key][0] < r1)] for a in C[key]) for key in ('inn', 'out', 'edge')}
        best = None
        for dx in range(-R, R + 1, 2):
            ph = lambda ys, xs: phi0[ys, np.clip(xs - dx, 0, 799)]
            e = energy(ph, sub, norm) + 1e-3 * abs(dx)
            if best is None or e < best[0]:
                best = (e, dx)
        total += best[0]
        out[r0:r1] = shift_x(M0, best[1], False)[r0:r1]
    return total, out


def lp_blend(Phis, C, norm):
    """min Σ s_in/n_in + λ Σ s_out/n_out + μ Σ (e⁺ + e⁻)/n_e  (norm) or plain sums; w ≥ 0, Σ w = 1."""
    A = {key: np.stack([p[C[key][0], C[key][1]] for p in Phis], 1) for key in ('inn', 'out', 'edge')}
    k, ni, no, ne = len(Phis), len(A['inn']), len(A['out']), len(A['edge'])
    sc = lambda n: 1 / max(n, 1) if norm else 1.0
    c = np.concatenate([np.zeros(k), np.full(ni, sc(ni)), np.full(no, LAM * sc(no)), np.full(2 * ne, MU * sc(ne))])
    Z = lambda r, cc: sp.csr_matrix((r, cc))
    A_ub = sp.vstack([sp.hstack([sp.csr_matrix(A['inn']), -sp.eye(ni), Z(ni, no), Z(ni, 2 * ne)]),
                      sp.hstack([sp.csr_matrix(-A['out']), Z(no, ni), -sp.eye(no), Z(no, 2 * ne)])])
    b_ub = np.full(ni + no, -M_)
    A_eq = sp.vstack([sp.hstack([sp.csr_matrix(A['edge']), Z(ne, ni + no), -sp.eye(ne), sp.eye(ne)]),
                      sp.hstack([sp.csr_matrix(np.ones((1, k))), Z(1, ni + no + 2 * ne)])])
    b_eq = np.concatenate([np.zeros(ne), [1.0]])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    w = r.x[:k] if r.success else np.full(k, 1 / k)
    return w, (r.fun if r.success else np.nan) / max(ni + no + ne, 1)


def label(k, E, bg, d, norm, rng):
    b = box(k)
    a = (b[2] - b[0]) / (b[3] - b[1])
    R = int(round(R_FRAC * (b[2] - b[0])))
    left = seg_of(k)['dir'] == 'L'
    P = gt_mask_native(SEQ, k)
    cand = [e for e in E if abs(e[0] - k) > EXCL]
    near = [e for e in cand if abs(e[1] - a) / a < ASP] or sorted(cand, key=lambda e: abs(e[1] - a))[:20]
    cs = [(c[:, ::-1] if left else c) for _, _, c in near]
    C = constraints(k, bg, d, P, [place(c, b, 0.0) for c in cs], R, rng)
    al = sorted((r for r in (align(c, b, C, R, norm) for c in cs) if r is not None), key=lambda t: t[0])[:K]
    Phis = [sdf(m) for _, m in al]
    top3 = np.mean(Phis[:3], 0) < 0
    w, viol = lp_blend(Phis, C, norm)
    lp = np.tensordot(w, np.stack(Phis), 1) < 0
    neff = float(np.exp(-(w[w > 0] * np.log(w[w > 0])).sum()))
    return dict(top3=top3 | P, lp=lp | P), dict(viol=viol, neff=neff), C, P


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    d, E, rng = np.load(DEPTH), bank('clear'), np.random.default_rng(0)
    poly = {k: cv2.imread(f'data/nfo_processed/{SEQ}_gt/{k:05d}_handpoly.png', 0) == 1 for k in TUNE + TEST}
    brush = {int(os.path.basename(f)[:5]): cv2.imread(f, 0) > 0
             for f in sorted(glob.glob(f'data/nfo_processed/{SEQ}_gt/*_handmask_raw.png'))}
    bgs = {}
    res, logs = {}, {}
    if os.path.exists(f'{OUT}/labels.npz'):          # cached labels: score only (f_measure needs skimage)
        Z = np.load(f'{OUT}/labels.npz')
        for key in Z.files:
            k, arm = key.split('|'); res.setdefault(int(k), {})[arm] = Z[key]
    for k in ([] if res else sorted(set(poly) | set(brush))):
        s = seg_of(k)
        if s['idx'] not in bgs:
            w0, w1 = s['warmup']
            bgs[s['idx']] = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
        bg = bgs[s['idx']]
        S, P = silhouette(k, E, bg, d, align=True)
        arms = {'SAM2-ceiling': P, 'frozen': S | P}
        for norm in (False, True):
            L, lg, C, _ = label(k, E, bg, d, norm, rng)
            tag = 'mean' if norm else 'sum'
            arms[f'S2-top3 {tag}'], arms[f'S3-LP {tag}'] = L['top3'], L['lp']
            logs[(k, tag)] = lg
        res[k] = arms
        if k in poly:
            raw = cv2.cvtColor(V.native(k, SEQ), cv2.COLOR_GRAY2BGR)
            v1 = raw.copy(); v1[C['N']] = (0.6 * v1[C['N']] + 0.4 * np.array([255, 128, 0])).astype(np.uint8)
            v1[C['dG_true']] = (0, 255, 0); v1[C['dG_occ']] = (0, 0, 255)
            panes = [v1]
            for a in ('frozen', 'S3-LP mean', 'S3-LP sum'):
                v = raw.copy(); m = arms[a]
                v[m] = (0.5 * v[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
                cnt, _ = cv2.findContours(poly[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(v, cnt, -1, (0, 255, 255), 1)
                cv2.putText(v, f'{k} {a}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                panes.append(v)
            x0, y0, x1, y1 = box(k)
            ys, xs = slice(max(0, y0 - 40), min(600, y1 + 40)), slice(max(0, x0 - 60), min(800, x1 + 60))
            cv2.imwrite(f'{IMG}/{k:05d}.jpg', np.hstack([p[ys, xs] for p in panes]))
        print(k, {t: {kk: round(v, 2) for kk, v in logs[(k, t)].items()} for t in ('sum', 'mean')}, flush=True)
    if not os.path.exists(f'{OUT}/labels.npz'):
        np.savez_compressed(f'{OUT}/labels.npz', **{f'{k}|{a}': m for k, arms in res.items() for a, m in arms.items()})
    rng = np.random.default_rng(1)
    for name, H, sets in (('polygons', poly, (('TUNE', TUNE), ('TEST', TEST), ('all', TUNE + TEST))),
                          ('brush', brush, (('all', sorted(brush)),))):
        for sname, ks in sets:
            print(f'\nvs {name} [{sname}, n={len(ks)}]: J / P / R / F2 / boundary F')
            sc = {a: [] for a in res[ks[0]]}
            for k in ks:
                for a, m in res[k].items():
                    s = scores(m, H[k]); sc[a].append([s['J'], s['P'], s['R'], s['F2'], f_measure(m, H[k])])
            for a, v in sc.items():
                print(f'  {a:14s} ' + ' / '.join(f'{x:.3f}' for x in np.mean(v, 0)))
            for a in sc:
                if a in ('frozen', 'SAM2-ceiling'):
                    continue
                dF = np.array(sc[a])[:, 3] - np.array(sc['frozen'])[:, 3]
                bs = [dF[rng.integers(0, len(dF), len(dF))].mean() for _ in range(5000)]
                print(f'    ΔF2 {a} − frozen {dF.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]'
                      f' better {(dF > 0).sum()}/{len(dF)}')
    if logs:
        json.dump({f'{k}|{t}': v for (k, t), v in logs.items()}, open(f'{OUT}/lp_logs.json', 'w'), indent=0)


if __name__ == '__main__':
    main()
