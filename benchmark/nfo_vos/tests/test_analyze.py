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


def test_t4_2_star_is_the_m_with_best_mean_JF_over_the_whole_pilot():
    df = pd.DataFrame([dict(method=f't4-2_m{m}', trial=f'x{i}', JF=0.5 + 0.1 * (m == 3) + 0.01 * i)
                       for m in range(1, 8) for i in range(4)])
    assert analyze.pick_vote_m(df) == 3


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
