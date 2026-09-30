"""Extrapolated boxes for the frames just outside a GT segment (person entering/leaving frame).

NFO's annotation stops while the person is still partly visible (seq1 GT ends at f137 with the
box right edge at 0.916 of the frame). These boxes prompt+clip SAM2 over those frames. Centre is
extrapolated at constant velocity; SIZE is held at the recent median, because GT boxes widen fast
near the edge (seq1: w 0.121 -> 0.188 over f130-137) and extrapolating that would blow up.
"""
import pytest

from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import extrapolate_edge_boxes
from utils.bb_utils import BoundingBox


def walker(frames, vx=0.02, w=0.1, h=0.3):
    # centre x = 0.5 + vx*(i-100), constant size
    return {i: [BoundingBox(0.5 + vx * (i - 100) - w / 2, 0.3, w, h)] for i in frames}


def test_after_side_continues_at_constant_velocity_with_fixed_size():
    bbs = walker(range(100, 111))                     # centre 0.5 .. 0.7
    ext = extrapolate_edge_boxes(bbs, 100, 110, 'after', lo=0, hi=500)
    assert list(ext)[:2] == [111, 112]
    assert ext[111].x + ext[111].w / 2 == pytest.approx(0.72)
    assert ext[111].w == pytest.approx(0.1) and ext[111].h == pytest.approx(0.3)


def test_stops_once_the_box_is_fully_outside_the_frame():
    bbs = walker(range(100, 111))
    ext = extrapolate_edge_boxes(bbs, 100, 110, 'after', lo=0, hi=500)
    last = ext[max(ext)]
    assert last.x < 1.0                                # last kept box still overlaps the frame
    assert last.x + 0.02 >= 1.0 - 1e-9                 # the next one would not


def test_before_side_runs_backwards_in_time():
    bbs = walker(range(100, 111))
    ext = extrapolate_edge_boxes(bbs, 100, 110, 'before', lo=0, hi=500)
    assert max(ext) == 99 and ext[99].x + ext[99].w / 2 == pytest.approx(0.48)


def test_never_enters_a_neighbouring_segment():
    bbs = walker(range(100, 111))
    ext = extrapolate_edge_boxes(bbs, 100, 110, 'after', lo=0, hi=114)
    assert max(ext) == 114


def test_size_is_the_median_not_the_last_widened_box():
    bbs = walker(range(100, 111))
    bbs[110] = [BoundingBox(0.6, 0.3, 0.3, 0.3)]      # one widened edge box
    ext = extrapolate_edge_boxes(bbs, 100, 110, 'after', lo=0, hi=500)
    assert ext[111].w == pytest.approx(0.1)


def test_max_frames_caps_the_extension():
    bbs = walker(range(100, 111), vx=0.001)            # slow: would take hundreds of frames
    assert len(extrapolate_edge_boxes(bbs, 100, 110, 'after', lo=0, hi=5000, max_frames=40)) == 40
