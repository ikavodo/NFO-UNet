"""Building an align_frames()-compatible 'winner' dict FROM ground truth, instead of from the
Kalman tracker. Same downstream code (align_frames fits vx once and extrapolates from a single
center anchor - see integrate_image.py), different source of positions.

The one thing worth pinning down with a test: align_frames indexes both the frame stack AND
winner['history'] by the LOCAL 0..T-1 position within the segment, never by the absolute video
frame number - get that wrong and every crop lands on the wrong frame.
"""
import numpy as np
import pytest

from tracking.eval.gt_integrated_image import build_gt_winner


def test_history_keys_are_local_indices_not_absolute_frame_numbers():
    # a person at constant pixel position, labelled at absolute frames 100, 101, 102 (a run
    # gt_runs would report as (100, 102)) - history must be keyed 0, 1, 2, not 100, 101, 102
    gt = {100: (0.4, 0.4, 0.1, 0.2), 101: (0.4, 0.4, 0.1, 0.2), 102: (0.4, 0.4, 0.1, 0.2)}
    winner, abs_idx = build_gt_winner(gt, (100, 102), w=200, h=200)
    assert set(winner['history']) == {0, 1, 2}
    assert abs_idx == [100, 101, 102]


def test_recovers_a_known_constant_velocity_exactly():
    # x moves 5px/frame in normalised units of 5/200=0.025 per frame over a 200px-wide frame
    w = h = 200
    gt = {f: (0.1 + 0.025 * i, 0.3, 0.1, 0.2) for i, f in enumerate(range(50, 56))}
    winner, _ = build_gt_winner(gt, (50, 55), w=w, h=h)
    assert winner['vx'] == pytest.approx(5.0)


def test_history_keeps_the_true_per_frame_y_not_a_pre_averaged_one():
    # align_frames computes ay = mean(history[f][1] for f in frames) ITSELF once it receives the
    # winner dict (see integrate_image.py), so build_gt_winner must hand it the true per-frame y -
    # collapsing to a mean here would double up with align_frames' own averaging and also throw
    # away the one place the true trajectory is actually recorded
    gt = {0: (0.2, 0.1, 0.1, 0.1), 1: (0.2, 0.5, 0.1, 0.1), 2: (0.2, 0.9, 0.1, 0.1)}
    winner, _ = build_gt_winner(gt, (0, 2), w=100, h=100)
    ys = [winner['history'][t][1] for t in sorted(winner['history'])]
    assert ys == pytest.approx([15.0, 55.0, 95.0])   # (y + bh/2) * h = (0.1+0.05)*100, etc.


def test_a_single_frame_segment_has_zero_fitted_velocity():
    gt = {7: (0.5, 0.5, 0.1, 0.1)}
    winner, abs_idx = build_gt_winner(gt, (7, 7), w=100, h=100)
    assert winner['vx'] == 0.0 and abs_idx == [7]

