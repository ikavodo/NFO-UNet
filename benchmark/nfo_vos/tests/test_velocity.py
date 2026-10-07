import json

import numpy as np

from benchmark.nfo_vos import velocity as V


def test_horizon_is_causal_strided_and_filtered():
    assert V.horizon_frames(112, 7, 2, lambda k: k >= 100) == [100, 102, 104, 106, 108, 110, 112]
    assert V.horizon_frames(103, 7, 2, lambda k: k >= 100) == [101, 103]


def _synthetic_dets(t0=20, n=13, vx=3.0):
    """Person at x = 100 + vx*k, fragmented (head/legs alternate +-8 px), plus a static distractor."""
    dets = {}
    for k in range(t0 - n + 1, t0 + 1):
        x = 100 + vx * k
        o = 8 if k % 2 else -8
        dets[k] = [{'x': x + o, 'y': 50, 'bbox': (x - 4 + o, 30, x + 4 + o, 70)},
                   {'x': 10.0, 'y': 50, 'bbox': (6, 30, 14, 70)}]
    return dets


def test_chain_recovers_velocity_across_fragments_without_track_ids():
    t0 = 20
    pts, v = V.chain_positions(_synthetic_dets(t0), t0, p0=(100 + 3.0 * t0, 50), v0=(0.0, 0.0), radius=25)
    assert abs(v[0] - 3.0) < 0.5 and abs(v[1]) < 1e-6
    assert all(abs(p[0] - 10) > 30 for p in pts.values())          # distractor never picked


def test_hybrid_reaches_full_horizon_and_is_horizontal_only():
    for t in json.load(open('results/benchmark/rr/trials.json'))[:3]:
        off = V.estimate_shifts(t, 'hybrid_x')
        assert len(off) >= 5 and all(dy == 0.0 for _, dy in off.values())
