"""The pairing rule for the live YOLO-vs-tracker split screen.

In the offline script the two panels align exactly: YOLO runs on every frame, so the tracker's
emitted frame index is always present in the buffer. Live, YOLO cannot run on every frame - it
costs ~48ms against the tracker's ~8ms - so the exact index is usually MISSING and the buffer has
to be interrogated rather than indexed. That is the part that can silently mis-pair, which is why
it is a pure function with tests rather than an expression inlined in the render loop.

The rule: newest buffered detection whose frame index is NOT NEWER than the tracker's readout.
Never showing a future frame is the load-bearing half - a detector shown a later frame than the
tracker is being flattered, and on these clips the person moves ~9px/frame, so a few frames of
leak is a whole body width of unearned alignment.
"""
import pytest

from tracking.stream.live import pick_aligned


def test_exact_index_present_is_returned():
    assert pick_aligned({4: ['a'], 7: ['b']}, 7) == (7, ['b'])


def test_never_returns_a_frame_newer_than_the_readout():
    # 9 is in the future relative to the tracker's frame 7: showing it would flatter YOLO
    assert pick_aligned({4: ['old'], 9: ['future']}, 7) == (4, ['old'])


def test_picks_the_newest_of_several_older_entries():
    assert pick_aligned({1: ['a'], 5: ['b'], 6: ['c']}, 7) == (6, ['c'])


def test_returns_none_when_every_entry_is_in_the_future():
    assert pick_aligned({8: ['x'], 9: ['y']}, 7) is None


def test_returns_none_on_an_empty_buffer():
    assert pick_aligned({}, 7) is None


def test_an_empty_detection_list_is_a_real_answer_not_a_miss():
    # YOLO finding nothing is a measurement; it must not be confused with having no answer yet,
    # or the panel would fall back to a stale box and show a person who is no longer detected
    assert pick_aligned({7: []}, 7) == (7, [])
