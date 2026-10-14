"""Test (a), pre-registered 2026-10-14: does an amodal video model move SAM2's SAM2-ceiling toward the hand polygons?

Model: Diffusion-VAS (Chen, Ramanan & Khurana, CVPR 2025, arXiv 2412.04623), amodal-segmentation stage only.
Code: github.com/Kaihua-Chen/diffusion-vas @ bbea113 (cloned to ../diffusion-vas, MIT); weights
kaihuac/diffusion-vas-amodal-segmentation (HF). Env: conda `dvas` (clone of `occ` + diffusers 0.29.1,
transformers 4.44.2).

Input per seq1 polygon frame k: a 25-frame clip at stride 3 (footage 25 fps, model trained at 8 fps), k at
position 12 (shifted later where k-36 < 0, e.g. frame 30) of SAM2-ceiling modal masks (empty where no SAM2-ceiling exists) + Depth Anything V2-L depth.
Output A = upsampled prediction ∪ native-resolution modal mask; Δ = A ∖ modal (the pixels the model adds).
Scored at the centre frame vs the polygon: J, P, R of SAM2-ceiling and of A (paired, same frames), 3 seeds.
Pass: mean J(A) > mean J(SAM2-ceiling) with paired bootstrap CI above 0, and P(A) ≥ 0.75.

Deviations from demo.py (flagged):
- union with the modal mask at native 800x600, not at 256x512 (demo.py unions before upsampling, which
  would also degrade the input's edges); the demo-faithful union is reported too.
- depth from the HF port Depth-Anything-V2-Large-hf (same weights as the demo's vitl checkpoint), run on the
  256x512 RGB like demo.py rgb_to_depth, then the same clip-wide min-max to [-1, 1].
- grayscale frames replicated to 3 channels (PIL convert('RGB'), as demo.py does for any input).
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# on sys.path only for this import: its top-level utils.py would shadow our utils/ (a namespace package)
sys.path.append(os.path.abspath('../diffusion-vas'))
from models.diffusion_vas.pipeline_diffusion_vas import DiffusionVASPipeline  # noqa: E402
sys.path.pop()

SEQ, GT = 'seq1', 'data/nfo_processed/seq1_gt'
CKPT = os.path.expanduser('~/models/diffusion-vas/diffusion-vas-amodal-segmentation')
DEPTH = 'depth-anything/Depth-Anything-V2-Large-hf'
RES, NF, HALF, STRIDE = (256, 512), 25, 12, 3
SEEDS = (23, 0, 1)
OUT = 'results/benchmark/rr/amodal_dvas'
IMG = 'images/benchmark/rr_checks/amodal'


def frame(k):
    from benchmark.nfo_vos import velocity as V
    return V.native(k, SEQ)


def modal(k):
    p = f'{GT}/{k:05d}_sammask_native.png'
    return cv2.imread(p, 0) > 127 if os.path.exists(p) else np.zeros((600, 800), bool)


# demo.py load_and_transform_masks / load_and_transform_rgbs (diffusion-vas @ bbea113), on in-memory frames
MASK_T = transforms.Compose([transforms.Resize(RES), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
RGB_T = transforms.Compose([transforms.Resize(RES), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def depth_clip(rgb, proc, net):
    """demo.py rgb_to_depth: depth per 256x512 frame, clip-wide min-max to [-1, 1], 3 channels."""
    out = []
    for x in rgb:
        img = ((x.numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        with torch.inference_mode():
            d = net(**proc(images=img, return_tensors='pt').to('cuda')).predicted_depth
        out.append(torch.nn.functional.interpolate(d[None], RES, mode='bicubic', align_corners=False)[0, 0].cpu().numpy())
    d = np.array(out)
    d = (d - d.min()) / (d.max() - d.min()) * 2 - 1
    return torch.tensor(d, dtype=torch.float32)[:, None].repeat(1, 3, 1, 1)[None]


def stats(a, h):
    tp = (a & h).sum()
    return tp / max((a | h).sum(), 1), tp / max(a.sum(), 1), tp / max(h.sum(), 1)


def main():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    ks = sorted(int(f[:5]) for f in os.listdir(GT) if f.endswith('_handpoly.png'))
    os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    pipe = DiffusionVASPipeline.from_pretrained(CKPT, torch_dtype=torch.float16).to('cuda')
    pipe.enable_model_cpu_offload(); pipe.set_progress_bar_config(disable=True)
    proc, net = AutoImageProcessor.from_pretrained(DEPTH), AutoModelForDepthEstimation.from_pretrained(DEPTH).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for k in ks:
        c = min(HALF, k // STRIDE)   # position of k in the clip
        idx = [k + STRIDE * (j - c) for j in range(NF)]
        M = [modal(i) for i in idx]
        mp = torch.stack([MASK_T(Image.fromarray(m.astype(np.uint8) * 255).point(lambda p: 255 if p > 128 else 0))
                          for m in M])[None]
        rgb = torch.stack([RGB_T(Image.fromarray(frame(i)).convert('RGB')) for i in idx])
        dp = depth_clip(rgb, proc, net)
        h = cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1
        low = (mp[0, c, 0].numpy() > 0).astype(np.uint8)
        preds = []
        for s in SEEDS:
            t = time.time()
            # demo.py main(): identical pipeline arguments
            fr = pipe(mp, dp, height=RES[0], width=RES[1], num_frames=NF, decode_chunk_size=8, motion_bucket_id=127,
                      fps=8, noise_aug_strength=0.02, min_guidance_scale=1.5, max_guidance_scale=1.5,
                      generator=torch.manual_seed(s)).frames[0]
            p = np.array(fr[c]).astype(np.int32).sum(-1) > 600   # demo.py binarisation
            up = cv2.resize(p.astype(np.uint8), (800, 600), interpolation=cv2.INTER_NEAREST) > 0
            demo = cv2.resize((p | (low > 0)).astype(np.uint8), (800, 600), interpolation=cv2.INTER_NEAREST) > 0
            A = up | M[c]
            preds.append(A)
            rows.append(dict(k=k, seed=s, sec=time.time() - t, pgt=stats(M[c], h), A=stats(A, h), demo=stats(demo, h),
                             d_in_poly=(A & ~M[c] & h).sum() / max((A & ~M[c]).sum(), 1),
                             d_area=(A & ~M[c]).sum() / max(M[c].sum(), 1)))
            r = rows[-1]
            print(f"{k} seed {s}: pGT J/P/R {'/'.join(f'{v:.3f}' for v in r['pgt'])}  A {'/'.join(f'{v:.3f}' for v in r['A'])}"
                  f"  Δ in polygon {r['d_in_poly']:.2f}, Δ area {r['d_area']:.2f}× pGT  ({r['sec']:.0f}s)", flush=True)
        np.savez_compressed(f'{OUT}/{k:05d}.npz', amodal=np.stack(preds), seeds=np.array(SEEDS))
    np.save(f'{OUT}/rows.npy', rows, allow_pickle=True)
    print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')

    # paired summary: per frame, seed-mean
    f = lambda key, i: np.array([np.mean([r[key][i] for r in rows if r['k'] == k]) for k in ks])
    rng = np.random.default_rng(0)
    for name, key in (('SAM2-ceiling', 'pgt'), ('amodal (native union)', 'A'), ('amodal (demo union)', 'demo')):
        print(f'{name:24s} J {f(key, 0).mean():.3f}  P {f(key, 1).mean():.3f}  R {f(key, 2).mean():.3f}  (n={len(ks)})')
    d = f('A', 0) - f('pgt', 0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    print(f'ΔJ (A - pGT) {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], '
          f'better on {(d > 0).sum()}/{len(d)}; seed spread of J(A) per frame: '
          f'{np.mean([np.std([r["A"][0] for r in rows if r["k"] == k]) for k in ks]):.3f}')
    print(f"Δ in polygon {np.mean([r['d_in_poly'] for r in rows]):.2f}, Δ area {np.mean([r['d_area'] for r in rows]):.2f}× pGT")

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, (len(ks) + 1) // 2, figsize=(12, 6))
    for a_, k in zip(ax.ravel(), ks):
        A, m, h = np.load(f'{OUT}/{k:05d}.npz')['amodal'][0], modal(k), cv2.imread(f'{GT}/{k:05d}_handpoly.png', 0) == 1
        ys, xs = np.nonzero(A | h | m)
        y0, y1, x0, x1 = max(0, ys.min() - 15), min(600, ys.max() + 15), max(0, xs.min() - 15), min(800, xs.max() + 15)
        a_.imshow(frame(k)[y0:y1, x0:x1], cmap='gray')
        ov = np.zeros((y1 - y0, x1 - x0, 4)); ov[(A & ~m)[y0:y1, x0:x1]] = (1, 0, 0, 0.5)
        a_.imshow(ov)
        for mm, col in ((h, 'cyan'), (m, 'yellow')):
            a_.contour(mm[y0:y1, x0:x1].astype(np.uint8), [0.5], colors=col, linewidths=0.8)
        a_.set_title(str(k), fontsize=8); a_.axis('off')
    for a_ in ax.ravel()[len(ks):]:
        a_.axis('off')
    fig.suptitle('cyan = your polygon, yellow = SAM2-ceiling (modal input), red fill = pixels Diffusion-VAS adds (seed 23)',
                 fontsize=9)
    fig.tight_layout(); fig.savefig(f'{IMG}/dvas_vs_polygon.png', dpi=75)


if __name__ == '__main__':
    main()
