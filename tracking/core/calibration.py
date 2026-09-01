"""Put the track score on a probability scale, so the internals read as something interpretable.

The score out of score_and_fit is span * net_disp / (1 + resid_std) * exp(-rel_dev^2), in units of
frames*px/(1+px). It ranks candidates correctly - AUC 0.946 for person-present on the labelled
clips - but it is unbounded, heavy-tailed, and on a scale nobody can reason about. "score 143" does
not tell you whether the tracker is confident, and `--min-score 5` is a threshold whose units are
frames*px/(1+px).

Platt scaling (Platt 1999, "Probabilistic outputs for support vector machines...") is the standard
fix: fit a two-parameter logistic from an existing uncalibrated score to a posterior probability,

    P(person present | score) = sigma(a * log1p(score) + b)

leaving the ranking untouched - a logistic in a monotone transform of the score cannot reorder
anything, so calibration costs exactly zero AUC and buys a readable number. log1p because the score
is unbounded and heavy-tailed, and because the existing feature code already uses log1p(score).

ONE INPUT ON PURPOSE. A logistic over the eleven features in presence_learning.py would score
better, but it would be a learned classifier standing beside the tracker rather than an
interpretation OF the tracker - and the point here is interpretability of the internals, not
accuracy. Calibrating the statistic the pipeline already computes keeps one number on screen with a
reliability curve behind it.

Fit and refresh the constants with:  python -m tracking.eval.calibrate_score
"""
import json
import math
import os
from dataclasses import dataclass

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'score_calibration.json')


@dataclass(frozen=True)
class Calibration:
    """P(present) = sigma(a * log1p(score) + b). `meta` records what it was fitted on."""
    a: float
    b: float
    meta: dict = None


def confidence(score: float, cal: Calibration) -> float:
    """Calibrated P(person present) for one score. Clamped rather than fussy: a negative score is
    impossible by construction but must not raise from log1p, and the exponent is bounded so a
    huge score returns 1.0 instead of overflowing."""
    z = cal.a * math.log1p(max(float(score), 0.0)) + cal.b
    return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def score_for_confidence(p: float, cal: Calibration) -> float:
    """The score threshold equivalent to a probability threshold, so a --min-confidence can drive
    the existing score gate without touching it. p<=0 gives 0 (gate everything through) and p>=1
    gives a finite ceiling rather than inf, so both extremes stay usable."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    z = math.log(p / (1 - p))
    return max(math.expm1((z - cal.b) / cal.a), 0.0)


def load(path: str = DEFAULT_PATH) -> Calibration:
    with open(path) as f:
        d = json.load(f)
    return Calibration(a=d['a'], b=d['b'], meta=d.get('meta'))


def save(cal: Calibration, path: str = DEFAULT_PATH) -> None:
    json.dump({'a': cal.a, 'b': cal.b, 'meta': cal.meta}, open(path, 'w'), indent=1)
