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


def test_t4_2_star_is_the_m_with_best_mean_J50_over_the_whole_pilot():
    df = pd.DataFrame([dict(method=f't4-2_m{m}', trial=f'x{i}', **{'J@50': 0.5 + 0.1 * (m == 3) + 0.01 * i})
                       for m in range(1, 8) for i in range(4)])
    assert analyze.pick_vote_m(df) == 3
