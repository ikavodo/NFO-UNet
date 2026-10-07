"""Round-robin schedule (spec §3, rev. 5): every frame used at most once; cover/backup twins."""
import numpy as np
import pytest

from benchmark.nfo_vos import trials

T = trials.build_trials(seqs=('seq1',))


def test_window_is_W_frames_starting_at_t0():
    assert trials.W == 40
    assert all(len(t['frames']) == trials.W and t['frames'][0] == t['t0'] for t in T)


def test_each_frame_used_at_most_once():
    used = [(t['seq'], f) for t in T for f in t['frames']]
    assert len(used) == len(set(used))


def test_windows_stay_inside_their_segment():
    assert all(t['seg_start'] <= t['frames'][0] and t['frames'][-1] <= t['seg_end'] for t in T)


def test_every_segment_has_a_window_and_two_segments_per_dir_gait():
    segs = trials.segments('seq1')
    assert len(segs) == 8
    assert {s['idx'] for s in segs} == {t['seg_idx'] for t in T}
    groups = {}
    for s in segs:
        groups.setdefault((s['dir'], s['gait']), []).append(s)
    assert sorted(groups) == [('L', 'run'), ('L', 'walk'), ('R', 'run'), ('R', 'walk')]
    assert all(len(g) == 2 for g in groups.values())


def test_backup_is_the_twin_of_cover_same_dir_gait_phase():
    segs = trials.segments('seq1')
    for dg in {(s['dir'], s['gait']) for s in segs}:
        cover, backup = sorted((s for s in segs if (s['dir'], s['gait']) == dg), key=lambda s: s['start'])
        assert (cover['half'], backup['half']) == ('cover', 'backup')
        assert cover['phase'] == backup['phase']
    assert sorted({s['phase'] for s in segs if s['half'] == 'cover'}) == [0, 10, 20, 30]   # W/4 steps


def test_warmup_is_person_free_long_enough_and_causal_when_preceding():
    for q in trials.ALL_SEQS:
        segs = trials.segments(q)
        for s in segs:
            w0, w1 = s['warmup']
            assert w1 - w0 + 1 >= trials.MIN_WARMUP
            assert all(w1 < o['start'] - trials.GAP_MARGIN or w0 > o['end'] + trials.GAP_MARGIN for o in segs)
            if s['warmup_source'] == 'preceding':
                assert w1 < s['start'] - trials.GAP_MARGIN


def test_visibility_is_scale_free_one_on_clear_frames():
    for seg in trials.segments('seq1'):
        vs = [trials.visibility(i, seg) for i in trials.clear_frames(seg)]
        if vs:
            assert abs(np.median(vs) - 1) < 1e-9


def test_native_point_inside_native_box():
    for t in T:
        x0, y0, x1, y1 = t['box_native']
        px, py = t['point_native']
        assert x0 <= px <= x1 and y0 <= py <= y1


def test_to_native_inverts_padded_square():
    assert np.allclose(trials.to_native(111.0, 111.0), (400, 300), atol=2)
