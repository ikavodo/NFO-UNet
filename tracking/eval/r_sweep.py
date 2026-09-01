"""Kill test: does the Kalman's measurement noise R matter at all here?

    python -m tracking.eval.r_sweep

Run this BEFORE building anything that adapts R per detection (NSA-Kalman style confidence
weighting, innovation-based R estimation, fragment-geometry covariance). Adaptive R can only help
if the fixed value sits on a SLOPE of the performance curve. If a 400x sweep of R moves nothing,
the filter is on a plateau, no weighting scheme can help, and the whole direction is dead for ten
minutes of compute instead of a day of implementation.

WHY A PLATEAU IS THE LIKELY OUTCOME, from reading the code rather than from running it. R enters in
exactly one place: _Track.update's gain, which shapes the filtered state, which is used by
_Track.predict, which supplies the PREDICTED POSITION for the Hungarian assignment cost. That is
all. update() stores the RAW measurement in tr.history, not the filtered state, and both the score's
linear fit and the readout's merged_center read history and the raw detections. So R never touches
the reported position directly - it can only change WHICH detection gets assigned to which track,
and only in frames where the assignment was close enough to flip. Everything else is invariant to it.

Reported per R: median normalised centre error against the ground-truth boxes, hit@0.1, and the
fraction of valid centres where no track was produced at all.
"""
import argparse

import numpy as np

from tracking.core.blob_tracker import _Track
from tracking.eval.eval_nfo import SEQS, eval_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--r', type=float, nargs='+', default=[0.5, 2.0, 9.0, 50.0, 200.0],
                   help='measurement-noise variances to try (default spans 400x around the 9.0 '
                        'in the code, i.e. R/Q from 0.25 to 100 at the fixed Q_VAR=2.0)')
    p.add_argument('--seqs', nargs='+', default=None)
    p.add_argument('--shape-scoring', action='store_true')
    a = p.parse_args()

    seqs = a.seqs or list(SEQS)
    baseline, rows = _Track.R_VAR, []
    print(f'Q_VAR={_Track.Q_VAR}, sequences: {", ".join(seqs)}\n')
    try:
        for r_var in a.r:
            _Track.R_VAR = r_var
            res, no_track, valid = [], 0, 0
            for seq in seqs:
                rs, nt, nv = eval_sequence(seq, a.shape_scoring)
                res += rs
                no_track += nt
                valid += nv
            hit = float(np.mean(np.array(res) < 0.1)) if res else float('nan')
            rows.append((r_var, float(np.median(res)) if res else float('nan'), hit,
                         no_track / max(valid, 1), len(res)))
            print(f'  R_VAR {r_var:7.1f}  (R/Q {r_var / _Track.Q_VAR:6.1f})   median err '
                  f'{rows[-1][1]:.4f}   hit@0.1 {100 * hit:5.1f}%   no-track '
                  f'{100 * rows[-1][3]:5.1f}%   n={len(res)}', flush=True)
    finally:
        _Track.R_VAR = baseline

    errs = [r[1] for r in rows]
    hits = [r[2] for r in rows]
    print(f'\nspread across a {max(a.r) / min(a.r):.0f}x sweep of R: median error '
          f'{min(errs):.4f}-{max(errs):.4f} (range {max(errs) - min(errs):.4f}), '
          f'hit@0.1 {100 * min(hits):.1f}-{100 * max(hits):.1f}%')
    if max(hits) - min(hits) < 0.01 and max(errs) - min(errs) < 0.002:
        print('VERDICT: plateau. R is inert over this range, so no scheme that ADAPTS R can pay '
              'off. Spend the effort on interpretability of the score instead.')
    else:
        print('VERDICT: there is a slope. The direction of improvement tells you which way an '
              'adaptive R should push; the best fixed value is the baseline to beat.')


if __name__ == '__main__':
    main()
