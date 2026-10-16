"""Are CanaTree100 forest images comparable to NFO as synthetic backgrounds? (2026-10-16, user request.)

NFO maps are Depth Anything V2 L relative disparity (affine-invariant inverse depth), min–max normalised
(results/benchmark/rr/depth/*_bg_disp_1036.npy). CanaTree100 ships metric depth (uint16, presumably mm): its comparable
quantity is 1/z, min–max normalised per image (1st–99th percentile, clipped). Primary: images_depth_neural (dense);
check: images_depth (stereo, zeros = holes, excluded). The template photo's disparity is the cached Depth Anything map
(data/synthetic/foliage_depth/veronica_disp.npy), cropped to the rows the generator uses (0–0.6·H).
Per image: disparity percentiles p5 / p50 / p95; share 'far' (< 0.1: the N_DISP threshold of the silhouette label) and
'near' (> 0.5); grayscale mean / std; texture = share of pixels with |Laplacian| > 20 (fine foliage edges).
CanaTree metric depth (images_depth, uint16, presumably mm) is reported too: median and p90 range in m.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.synth.canatree_stats
"""
import glob
import json
import sys

import cv2
import numpy as np

ROOT = 'data/CanaTree100'
TEMPLATE = 'images/veronica111886-yakushima-2336735.jpg'
OUT = 'results/benchmark/rr/depth/canatree_stats.json'
IMG = 'images/benchmark/rr_checks/canatree_depth.png'


def stats(gray, disp):
    """disp may hold NaN (depth holes). Texture is measured at the generator's output width (960 px; NFO native is 800)."""
    g = gray if gray.ndim == 1 else cv2.resize(gray, (960, int(round(gray.shape[0] * 960 / gray.shape[1]))), interpolation=cv2.INTER_AREA)
    lap = np.abs(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)) if g.ndim == 2 else np.zeros(1)
    d = disp[np.isfinite(disp)]
    return dict(p5=float(np.percentile(d, 5)), p50=float(np.median(d)), p95=float(np.percentile(d, 95)),
                far=float((d < 0.1).mean()), near=float((d > 0.5).mean()), mean=float(gray.mean()),
                std=float(gray.std()), texture=float((lap > 20).mean()))


def inv_norm(z):
    """metric depth (uint16, 0 = hole) → 1/z normalised to [0, 1] by its 1st–99th percentile; NaN on holes."""
    z = z.astype(np.float32)
    d = np.where(z > 0, 1.0 / np.maximum(z, 1), np.nan)
    lo, hi = np.nanpercentile(d, 1), np.nanpercentile(d, 99)
    return np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)


def main(n=101):
    res = {'nfo': [], 'template': [], 'canatree_neural': [], 'canatree_stereo': []}
    for s in ('seq1', 'seq2', 'seq3', 'seq4'):
        d = np.load(f'results/benchmark/rr/depth/{s}_bg_disp_1036.npy')
        g = cv2.imread(sorted(glob.glob(f'data/nfo_final/nfo_final/{s}/*.jpg'))[0], 0)
        res['nfo'].append(stats(g, d))
    t = cv2.imread(TEMPLATE, 0); rows = int(0.6 * t.shape[0])
    td = np.load('data/synthetic/foliage_depth/veronica_disp.npy').astype(np.float32)
    td = cv2.resize(td, t.shape[::-1])[:rows]; td = (td - td.min()) / (td.max() - td.min())
    res['template'].append(stats(t[:rows], td))
    files = sorted(glob.glob(f'{ROOT}/images/*.png'))[:n]
    metric, panes = [], []
    for k, f in enumerate(files):
        g = cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        zn = cv2.imread(f.replace('/images/', '/images_depth_neural/'), cv2.IMREAD_UNCHANGED)
        zs = cv2.imread(f.replace('/images/', '/images_depth/'), cv2.IMREAD_UNCHANGED)
        dn, ds = inv_norm(zn), inv_norm(zs)
        res['canatree_neural'].append(stats(g, dn))
        res['canatree_stereo'].append(stats(g, ds) | {'holes': float((~np.isfinite(ds)).mean())})
        res['canatree_neural'][-1]['holes'] = float((~np.isfinite(dn)).mean())
        metric.append((float(np.median(zn[zn > 0])) / 1000, float(np.percentile(zn[zn > 0], 90)) / 1000))
        if k < 6:
            panes.append(np.hstack([cv2.resize(g, (320, 180)), cv2.resize((np.nan_to_num(dn) * 255).astype(np.uint8), (320, 180))]))
    summ = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in res.items()}
    print(f'{len(files)} CanaTree images; mean per image: disparity p5 / p50 / p95 | far<0.1 | near>0.5 | gray mean / std | texture')
    for k, v in summ.items():
        print(f"  {k:16s} {v['p5']:.2f} / {v['p50']:.2f} / {v['p95']:.2f} | {v['far']:.2f} | {v['near']:.2f} | "
              f"{v['mean']:.0f} / {v['std']:.0f} | {v['texture']:.3f}" + (f"  holes {v['holes']:.2f}" if 'holes' in v else ''))
    print(f"  CanaTree neural metric depth: median {np.median([m[0] for m in metric]):.1f} m, p90 {np.median([m[1] for m in metric]):.1f} m")
    cn = res['canatree_neural']
    print('  CanaTree spread: p50 quartiles', np.round(np.percentile([r['p50'] for r in cn], [25, 50, 75]), 2).tolist(),
          '| far quartiles', np.round(np.percentile([r['far'] for r in cn], [25, 50, 75]), 2).tolist(),
          '| texture quartiles', np.round(np.percentile([r['texture'] for r in cn], [25, 50, 75]), 3).tolist())
    json.dump(dict(summary=summ, per_image=res), open(OUT, 'w'), indent=1)
    cv2.imwrite(IMG, np.vstack(panes))

if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:]))
