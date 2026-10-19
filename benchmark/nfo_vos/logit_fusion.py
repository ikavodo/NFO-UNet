"""Uncertainty-weighted logit fusion of the static-layer cue into T1's mask logits (ViperSAM-style late
fusion, docs/external/ViperSAM_with_equations.md, eq. for K_t; pre-registered in docs/vos_benchmark/archive/nfo_vos_pilot_findings_full.md,
2026-10-12). Output-only.

    python -m benchmark.nfo_vos.logit_fusion tune            # seq1: theta, lambda for fuse and const
    python -m benchmark.nfo_vos.logit_fusion write <theta> <lam> <lam_const>   # masks/t1-lgfuse, t1-lgconst

L' = L + lam * R * nu,  R = H(sigmoid(L)) / ln 2,  nu = tanh((z - theta) / 2)  (z: static_gate z, stride 16)
control: nu = -1 everywhere.  L: T1-lg logits at 300x400 (run_sam2_video --save-logits).
"""
import json
import os
import sys

import cv2
import numpy as np

from benchmark.nfo_vos import metrics, trials as TR
from benchmark.nfo_vos.run_sam2_video import gt_mask_native

ZDIR, LG = f'{TR.RES}/static_z', f'{TR.RES}/masks/t1-lg'
THETAS, LAMS = [3.0, 4.5, 6.0], [2.0, 4.0, 8.0, 16.0]


def trials():
    return [t for t in json.load(open(TR.OUT)) if t['admissible']]


def parts(t):
    L = np.load(f"{LG}/{t['id']}.npz")['logits'].astype(np.float32)
    z = np.load(f"{ZDIR}/{t['id']}.npz")['z'].astype(np.float32)
    zu = np.stack([cv2.resize(zi, (L.shape[2], L.shape[1]), interpolation=cv2.INTER_LINEAR) for zi in z])
    p = 1 / (1 + np.exp(-np.clip(L, -30, 30)))
    R = -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9)) / np.log(2)
    return L, R, zu


def fuse(L, R, zu, theta, lam, const=False):
    nu = -np.ones_like(L) if const else np.tanh((zu - theta) / 2)
    Lf = L + lam * R * nu
    return np.stack([cv2.resize(l, (800, 600), interpolation=cv2.INTER_LINEAR) > 0 for l in Lf])


def tune():
    res = {}
    for t in [t for t in trials() if t['seq'] == 'seq1']:
        L, R, zu = parts(t)
        G = [metrics.native_to_224(gt_mask_native(t['seq'], k)) for k in t['frames']]
        ks = [k for k in range(1, len(t['frames']) - 1) if G[k].any()]
        J = lambda M: [metrics.frame_metrics(metrics.native_to_224(M[k]), G[k])['J'] for k in ks]
        res.setdefault(('T1-lg',), []).extend(J(np.load(f"{LG}/{t['id']}.npz")['masks']))
        for lam in LAMS:
            res.setdefault(('const', lam), []).extend(J(fuse(L, R, zu, 0, lam, const=True)))
            for th in THETAS:
                res.setdefault(('fuse', th, lam), []).extend(J(fuse(L, R, zu, th, lam)))
    for k, v in sorted(res.items(), key=str):
        print(k, f'mean J {np.mean(v):.4f}')
    bf = max((k for k in res if k[0] == 'fuse'), key=lambda k: np.mean(res[k]))
    bc = max((k for k in res if k[0] == 'const'), key=lambda k: np.mean(res[k]))
    print(f'best fuse {bf} ({np.mean(res[bf]):.4f}); best const {bc} ({np.mean(res[bc]):.4f}); T1-lg {np.mean(res[("T1-lg",)]):.4f}')


def write(theta, lam, lam_const):
    for name in ('t1-lgfuse', 't1-lgconst'):
        os.makedirs(f'{TR.RES}/masks/{name}', exist_ok=True)
    for t in trials():
        src = {k: v for k, v in np.load(f"{LG}/{t['id']}.npz").items() if k != 'logits'}
        L, R, zu = parts(t)
        np.savez_compressed(f"{TR.RES}/masks/t1-lgfuse/{t['id']}.npz", **{**src, 'masks': fuse(L, R, zu, theta, lam)})
        np.savez_compressed(f"{TR.RES}/masks/t1-lgconst/{t['id']}.npz", **{**src, 'masks': fuse(L, R, zu, 0, lam_const, const=True)})
    print('-> masks/t1-lgfuse, t1-lgconst')


if __name__ == '__main__':
    tune() if sys.argv[1] == 'tune' else write(*map(float, sys.argv[2:5]))
