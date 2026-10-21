"""On-the-fly clips for the feature-layer plan (docs/layer_sam2/training/plan_feature_layers_training.md §2).

Clip = gpjatk_foliage.make_layered_clip(mode='perlin', occluder ≡ background, clump 24, ρ ~ U[0.35, 0.55]) with the
noise.py nuisance (sway on the scene, then gain / white noise / JPEG on the frames). Targets are nuisance-free:
  clean_t  person on the unswayed B, no occluder, no noise;  S_t silhouette;  V_t = S_t ∧ ¬occluder_t (swayed matte)
  O_t      = S_t ∧ ∪_{i=t−6..t} A_{i→t}(V_i): pixels visible in at least one GT-aligned history frame (incl. t)
  bg       pixel median of N_WARM noisy empty-scene frames (the NFO warm-up protocol)
Alignment A_{i→t}: similarity (scale + translation) mapping box i onto box t, p_t = c_t + s·(p_i − c_i), s = h_t / h_i,
boxes from S (GT). Jittered boxes: centre + N(0, σ_a) per axis, height × (1 + N(0, 0.03)).
Splits by subject: train p1–p20, val p21–p26, test p27–p32 (= gpjatk_foliage.HOLDOUT).
"""
import os

import numpy as np

from benchmark.nfo_vos.synth import noise
from benchmark.nfo_vos.synth.gpjatk_foliage import GP, HOLDOUT, make_layered_clip

SPLITS = dict(train=range(1, 21), val=range(21, 27), test=range(27, 33))
CAMS = ('c1', 'c2', 'c3', 'c4')
N, N_WARM = 40, 20
RHO = (0.35, 0.55)
assert {f'p{i}' for i in SPLITS['test']} == HOLDOUT


def sequences(split):
    subj = {f'p{i}' for i in SPLITS[split]}
    return sorted(d for d in os.listdir(GP) if d.split('s')[0] in subj)


def clip_specs(split, per_cam=1, seed=0):
    """Deterministic (seq, cam, ρ, seed) list: every sequence × camera, per_cam clips each."""
    rng = np.random.default_rng(seed)
    return [(s, c, float(rng.uniform(*RHO)), int(rng.integers(2 ** 31)))
            for s in sequences(split) for c in CAMS for _ in range(per_cam)]


def boxes_of(S):
    """[T, 3] (cx, cy, h) from silhouettes; NaN for empty frames."""
    out = np.full((len(S), 3), np.nan)
    for t, m in enumerate(S):
        ys, xs = np.nonzero(m)
        if len(ys):
            out[t] = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, ys.max() - ys.min() + 1)
    return out


def jitter(bx, sigma_a, rng):
    j = bx.copy()
    j[:, :2] += rng.normal(0, sigma_a, (len(bx), 2))
    j[:, 2] *= 1 + rng.normal(0, 0.03, len(bx))
    return j


def make_clip(seq, cam, rho, seed, flutter=0.0, clump=24):
    """flutter > 0: add noise.flutter_fields of that per-axis RMS (px) to the rigid sway, warm-up included.
    clump: perlin clump size, or a tuple for a mixed-scale union. Defaults reproduce the step-3 'base' clips exactly
    (flutter draws from its own seed, so the other random streams are unchanged)."""
    rng = np.random.default_rng(seed)
    p = noise.sample(rng)
    sway = list(noise.sway_offsets(p, N_WARM + N, -N_WARM))
    if flutter:
        sway = [f + d for f, d in zip(noise.flutter_fields(p, N_WARM + N, flutter), sway)]
    cid, *_, L = make_layered_clip(seq, cam, rho, seed, 'perlin', None, n=N, clump=clump, save=False, layers=True,
                                   sway=sway[N_WARM:])
    warm = noise.photometric(np.stack([noise.displace(L['B'], d) for d in sway[:N_WARM]]), p, salt=1)
    return dict(cid=cid, X=noise.photometric(L['X'], p), X0=L['X'], clean=L['clean'], B=L['B'], bg=np.median(warm, 0), warm=warm,
                S=L['S'], V=L['V'], boxes=boxes_of(L['S']), noise=p, rng=rng)


if __name__ == '__main__':
    print('val clips:', len(clip_specs('val')), '| train clips:', len(clip_specs('train')))
