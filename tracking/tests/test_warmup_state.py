"""What the app should be showing while MOG2 has not converged yet.

Two separate things gate the first output and they are easy to conflate. MOG2's per-pixel Gaussians
need ~2x bg_frames before the mask stops being full of spurious foreground (measured: at 1x, 57-75%
of emissions in a person-absent window carried a box, against 0-22% mid-clip). Separately, the
13-frame buffer has to fill before a centred window exists at all. Whichever is still unsatisfied is
the one worth telling the viewer about, and with the defaults that is MOG2 - but with
suppress_warmup=False the buffer is the only constraint, so both cases have to render.

Suppressing the BOX during this is correct and measured. Suppressing the FRAME was an accident of
step() returning None: the window froze for ~2s, ESC stopped responding, and nothing confirmed the
camera was even working.
"""
import pytest

from tracking.stream.stream import warmup_state


def test_nothing_seen_yet_reports_zero_progress_against_the_mog2_target():
    frac, caption = warmup_state(seen=0, warmup=60, have=0, need=13)
    assert frac == 0.0 and 'MOG2' in caption and '0/60' in caption


def test_halfway_through_the_mog2_warmup():
    frac, caption = warmup_state(seen=30, warmup=60, have=13, need=13)
    assert frac == pytest.approx(0.5) and '30/60' in caption


def test_ready_once_both_constraints_are_satisfied():
    assert warmup_state(seen=60, warmup=60, have=13, need=13) is None


def test_the_buffer_is_reported_when_mog2_suppression_is_disabled():
    # suppress_warmup=False sets warmup=0, so filling the window is the only thing left to wait for
    frac, caption = warmup_state(seen=5, warmup=0, have=5, need=13)
    assert frac == pytest.approx(5 / 13) and 'window' in caption and '5/13' in caption


def test_a_full_mog2_warmup_with_an_unfilled_buffer_is_still_not_ready():
    state = warmup_state(seen=60, warmup=60, have=12, need=13)
    assert state is not None and 'window' in state[1]


def test_mog2_is_reported_first_when_both_are_outstanding():
    # it is the constraint that finishes LAST under the defaults, so it is the honest one to show
    _, caption = warmup_state(seen=5, warmup=60, have=5, need=13)
    assert 'MOG2' in caption


def test_progress_never_leaves_the_unit_interval():
    for seen in (0, 1, 59, 60, 500):
        state = warmup_state(seen=seen, warmup=60, have=0, need=13)
        if state is not None:
            assert 0.0 <= state[0] <= 1.0
