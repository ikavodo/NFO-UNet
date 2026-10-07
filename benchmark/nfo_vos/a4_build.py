"""Build A/4 composites C_t for every window frame t of every admissible trial (CPU; heavy imports
kept out of a4.py so the SAM2 runner stays light).

    NFO_RUN=rr python -m benchmark.nfo_vos.a4_build          # -> results/benchmark/rr/composites_a4/<trial>/<t>.jpg

C_t: recency-weighted mean of the native frames t-6..t (absolute frames, may precede t0 -- causal),
each shifted onto frame t by a GT-free constant velocity v_t (x only):
  - OLS on x over T4's causal tracker chain (run_t4.track: follows the person forward from the
    shared t0 prompt) within [t-6, t], if it has >= 3 points;
  - else the pre-t0 hybrid estimate (velocity.estimate_shifts 'hybrid_x', 7.7 native px median
    error vs GT on the old pilot).
"""
import argparse
import json
import os

import cv2
import numpy as np

from benchmark.nfo_vos import a4, run_t4 as R, trials as TR, velocity

SC = max(TR.NATIVE_W, TR.NATIVE_H) / TR.SIZE          # 224 px -> native px
OUT = f'{TR.RES}/composites_a4'
TRACE = f'{TR.IMG}_checks/a4'


def velocity_at(chain, t, fallback, n=a4.N):
    """Native px/frame from the chain's own detections in [t-n+1, t] (224 coords)."""
    pts = [(k, chain[k][0]) for k in range(max(0, t - n + 1), t + 1) if k in chain]
    if len(pts) < 3:
        return fallback
    return float(np.polyfit([k for k, _ in pts], [x for _, x in pts], 1)[0] * SC)


def fallback_velocity(trial):
    off = velocity.estimate_shifts(trial, 'hybrid_x')
    ks = [k for k in off if k != trial['t0']]
    if not ks:
        return 0.0
    k = min(ks)
    return float(off[k][0] * SC / (trial['t0'] - k))


def build(trial, ts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    chain = R.track(trial)['chain']
    vfb = fallback_velocity(trial)
    vs = {}
    for t in ts:
        f = trial['t0'] + t
        ks = [k for k in range(f - a4.N + 1, f + 1) if k >= 0]
        vx = velocity_at(chain, t, vfb)
        frames = [velocity.native(k, trial['seq']) for k in ks]
        cv2.imwrite(f'{out_dir}/{t:02d}.jpg', a4.recency_fusion(frames, [vx * (f - k) for k in ks]))
        vs[t] = vx
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(TRACE, exist_ok=True)
    trials = [t for t in json.load(open(TR.OUT)) if t['admissible']][:a.limit]
    for i, t in enumerate(trials):
        d = f"{OUT}/{t['id']}"
        vs = build(t, list(range(len(t['frames']))), d)
        if i % 4 == 0:                                   # small visual trace: raw t0 | C_0 | C_20
            x0, y0, x1, y1 = (int(v) for v in t['box_native'])
            pad = int(0.6 * (y1 - y0))
            sl = (slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad))
            tiles = [velocity.native(t['t0'], t['seq'])[sl], cv2.imread(f'{d}/00.jpg', 0)[sl],
                     cv2.imread(f'{d}/20.jpg', 0)[sl]]
            cv2.imwrite(f"{TRACE}/{t['id']}.png", np.hstack([cv2.resize(x, (160, 220)) for x in tiles]))
        print(f"{t['id']}: {len(vs)} composites, v median {np.median(list(vs.values())):+.2f} native px/frame", flush=True)


if __name__ == '__main__':
    main()
