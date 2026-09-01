"""Turning the track score into a probability, so the number on screen means something.

The score is span * net_disp / (1 + resid_std) * exp(-rel_dev^2): units of frames*px/(1+px). A
threshold of "5" in those units is uninterpretable to everyone including its author, and so is a
HUD reading "score 143" - is that twice as certain as 71? There is no answer, because the score is
not on a probability scale and is not even bounded.

Platt scaling (Platt 1999) fixes exactly this: fit a two-parameter logistic mapping the existing
score onto P(person present), so the gate becomes --min-confidence 0.5 and the HUD reads P 0.87.
Deliberately ONE input, the score itself - this calibrates the statistic the tracker already
computes rather than replacing it with a learned classifier over eleven features, which would be a
different (and less interpretable) thing.
"""
import pytest

from tracking.core.calibration import Calibration, confidence, score_for_confidence

CAL = Calibration(a=1.4, b=-3.2)


def test_confidence_is_between_zero_and_one_for_any_score():
    assert all(0.0 <= confidence(s, CAL) <= 1.0 for s in (0.0, 1e-9, 5.0, 1e6, 1e30))


def test_confidence_increases_with_score():
    ps = [confidence(s, CAL) for s in (0.0, 1.0, 5.0, 20.0, 100.0, 1000.0)]
    assert ps == sorted(ps) and ps[0] < ps[-1]


def test_a_zero_score_is_not_certainty_either_way():
    # score 0 means "no candidate track", which is evidence of absence but not proof
    assert 0.0 < confidence(0.0, CAL) < 0.5


def test_score_for_confidence_inverts_confidence():
    for s in (0.5, 5.0, 50.0, 500.0):
        assert score_for_confidence(confidence(s, CAL), CAL) == pytest.approx(s, rel=1e-6)


def test_confidence_for_a_threshold_round_trips():
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert confidence(score_for_confidence(p, CAL), CAL) == pytest.approx(p, abs=1e-9)


def test_extreme_thresholds_do_not_produce_infinities():
    # a caller passing --min-confidence 0 or 1 must get a usable score threshold, not inf/nan
    lo, hi = score_for_confidence(0.0, CAL), score_for_confidence(1.0, CAL)
    assert lo == 0.0 and hi < float('inf')


def test_a_negative_score_is_clamped_rather_than_producing_a_domain_error():
    # log1p of a negative number is a domain error; scores are non-negative by construction but
    # the gate must not crash if one ever arrives
    assert 0.0 <= confidence(-1.0, CAL) <= 1.0
