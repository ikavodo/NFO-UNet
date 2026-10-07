import numpy as np
import pandas as pd

from benchmark.nfo_vos import analyze


def _synthetic(b=0.2, c=-0.004, d=0.003, e=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t0 in range(0, 100, 10):
        v, nf = rng.uniform(0.1, 1.2), int(rng.integers(1, 5))
        for dt in range(51):
            t = t0 + dt
            a_t = 0.5 + 0.3 * np.sin(t / 7)               # frame difficulty
            rows.append(dict(trial=f't{t0}', seg=0, t0=t0, t=t, dt=dt, v=v, n_frag=nf,
                             J=a_t + b * v + c * dt + d * v * dt + e * nf + rng.normal(0, 1e-4)))
    return pd.DataFrame(rows)


def test_fixed_effects_recovers_known_coefficients_despite_frame_difficulty():
    est = analyze.fe_fit(_synthetic())
    assert np.allclose([est['b'], est['c'], est['d'], est['e']], [0.2, -0.004, 0.003, 0.01], atol=2e-3)


def test_position_fixed_effects_recover_coefficients_without_frame_overlap():
    """8 segments of one scene, non-overlapping 40-frame windows at round-robin phases; difficulty
    is a function of position x only (the shared occluder field), never of the frame index."""
    rng = np.random.default_rng(1)
    rows = []
    for sgi in range(8):
        speed = (1.4 if sgi < 4 else 2.3) * (1 if sgi % 2 == 0 else -1)
        x_start = 10 if speed > 0 else 214
        for m in range(3):
            t0 = (sgi % 4) * 10 + m * 40
            v = rng.uniform(0.1, 1.2)
            for dt in range(40):
                x = x_start + speed * (t0 + dt)
                if not 0 <= x < 224: continue
                a_x = 0.6 + 0.25 * np.sin(x / 15)                          # occluder field
                rows.append(dict(trial=f's{sgi}m{m}', seq='seq1', seg=sgi, t=1000 * sgi + t0 + dt, dt=dt, x=x,
                                 v=v, n_frag=1, J=a_x + 0.2 * v - 0.004 * dt + 0.003 * v * dt + rng.normal(0, 1e-3)))
    est = analyze.fe_fit(pd.DataFrame(rows), effect='pos')
    assert np.allclose([est['b'], est['c'], est['d']], [0.2, -0.004, 0.003], atol=0.03)


def test_cluster_bootstrap_ci_contains_true_mean_and_is_wider_than_naive_for_clustered_data():
    rng = np.random.default_rng(3)
    rows = []
    for g in range(16):                       # 16 clusters with a shared offset each
        off = rng.normal(0, 0.1)
        for k in range(4):
            rows.append(dict(cluster=f'g{g}', d=0.02 + off + rng.normal(0, 0.01)))
    d = pd.DataFrame(rows)
    lo, hi = analyze.cluster_bootstrap_ci(d, 'd', 'cluster', B=4000, seed=0)
    naive = np.percentile([rng.choice(d.d.values, len(d)).mean() for _ in range(4000)], [2.5, 97.5])
    assert lo < d.d.mean() < hi
    assert (hi - lo) > 1.5 * (naive[1] - naive[0])
