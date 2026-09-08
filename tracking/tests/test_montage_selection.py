"""Which frames a montage shows, and why 'evenly spaced across everything' is the wrong default
when the point is to demonstrate ONE specific outcome (tracker right, YOLO wrong).

Row shape: (frame_index, vis, yolo_fired: bool, tracker_has_box: bool, gt_present: bool).
"""
import numpy as np
import pytest

from tracking.eval.yolo_vs_tracker import select_montage

ROWS = [(i, f'v{i}', yolo, tracker, present) for i, (yolo, tracker, present) in enumerate([
    (True,  True,  True),   # 0  both right
    (False, True,  True),   # 1  TRACKER WINS: present, tracker found it, yolo did not
    (False, False, True),   # 2  both miss
    (True,  True,  False),  # 3  both false-positive
    (False, True,  True),   # 4  TRACKER WINS
    (False, True,  True),   # 5  TRACKER WINS
    (False, True,  True),   # 6  TRACKER WINS
])]


def test_spread_ignores_outcome_and_just_samples_evenly():
    picked = select_montage(ROWS, 'spread', want=4)
    assert [r[0] for r in picked] == [0, 2, 4, 6]


def test_tracker_wins_keeps_only_present_tracker_yes_yolo_no():
    picked = select_montage(ROWS, 'tracker-wins', want=4)
    assert {r[0] for r in picked} == {1, 4, 5, 6}


def test_tracker_wins_excludes_a_false_positive_even_though_yolo_also_missed_it():
    # row 2 is (yolo=False, tracker=False, present=True): tracker did NOT win here, it also missed
    picked = select_montage(ROWS, 'tracker-wins', want=4)
    assert 2 not in {r[0] for r in picked}


def test_tracker_wins_excludes_absent_frames_even_if_tracker_fired_and_yolo_did_not():
    # row 3 has present=False - a tracker box there is a false positive, not a "win"
    picked = select_montage(ROWS, 'tracker-wins', want=1000)
    assert 3 not in {r[0] for r in picked}


def test_asking_for_fewer_than_available_still_spans_the_range():
    picked = select_montage(ROWS, 'tracker-wins', want=2)
    idx = [r[0] for r in picked]
    assert idx[0] == 1 and idx[-1] == 6 and len(idx) == 2


def test_no_qualifying_frames_returns_empty_rather_than_falling_back_to_spread():
    only_misses = [(0, 'v0', False, False, True), (1, 'v1', True, True, False)]
    assert select_montage(only_misses, 'tracker-wins', want=4) == []


def test_asking_for_more_than_available_returns_all_of_them_not_a_padded_list():
    picked = select_montage(ROWS, 'tracker-wins', want=100)
    assert len(picked) == 4
