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


# --- recalibrating something that is ALREADY a probability -------------------------------------
#
# YOLO reports a confidence in (0,1), but a network's sigmoid output is not a calibrated posterior -
# detectors are systematically over-confident. Printing YOLO's raw conf as "P" beside the tracker's
# Platt-calibrated P would look like parity without being parity, so YOLO gets the same treatment.
# The only difference is the link input: log1p for an unbounded score, logit for a probability,
# which is the standard way to recalibrate one probability into another.

LOGIT_CAL = Calibration(a=0.8, b=0.3, transform='logit')


def test_a_probability_input_is_mapped_through_its_logit():
    # a=1, b=0 through the logit link must be the identity, which pins the convention
    ident = Calibration(a=1.0, b=0.0, transform='logit')
    for q in (0.1, 0.4, 0.75, 0.99):
        assert confidence(q, ident) == pytest.approx(q, abs=1e-9)


def test_logit_transform_stays_in_range_at_the_boundaries():
    # conf exactly 0 or 1 must not produce inf through the logit
    assert 0.0 <= confidence(0.0, LOGIT_CAL) <= 1.0
    assert 0.0 <= confidence(1.0, LOGIT_CAL) <= 1.0


def test_logit_round_trips_like_the_log1p_link():
    for q in (0.05, 0.3, 0.6, 0.95):
        assert score_for_confidence(confidence(q, LOGIT_CAL), LOGIT_CAL) == pytest.approx(q, rel=1e-6)


def test_an_overconfident_detector_is_pulled_toward_the_middle():
    # a<1 flattens the curve, which is what correcting over-confidence looks like: a raw 0.95
    # must come back LOWER, not higher
    assert confidence(0.95, Calibration(a=0.5, b=0.0, transform='logit')) < 0.95


def test_the_default_transform_is_unchanged_for_existing_calibrations():
    # the committed score_calibration.json has no 'transform' key and must keep working
    assert Calibration(a=1.0, b=0.0).transform == 'log1p'
