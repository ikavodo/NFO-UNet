import numpy as np

from benchmark.nfo_vos import trials

T = trials.build_trials()


def test_pilot_segments_match_spec():
    assert [(s['start'], s['end']) for s in trials.pilot_segments()] == [(1154, 1308), (1476, 1621)]


def test_start_counts_match_spec():
    by_seg = {}
    for t in T:
        by_seg.setdefault(t['seg_start'], []).append(t['t0'])
    assert by_seg[1154] == list(range(1154, 1255, 10))
    assert by_seg[1476] == list(range(1476, 1567, 10))


def test_warmup_ranges_match_spec():
    w = {t['seg_start']: tuple(t['warmup']) for t in T}
    assert w[1154] == (946, 1113) and w[1476] == (1349, 1435)


def test_window_is_51_frames():
    assert all(len(t['frames']) == 51 and t['frames'][0] == t['t0'] for t in T)


def test_native_point_inside_native_box():
    for t in T:
        x0, y0, x1, y1 = t['box_native']
        px, py = t['point_native']
        assert x0 <= px <= x1 and y0 <= py <= y1


def test_to_native_inverts_padded_square():
    # 224 pixel centre 111.5 sits at the middle of the 800x800 padded square = native (400, 300)
    assert np.allclose(trials.to_native(111.5 - 0.5, 111.5 - 0.5), (400, 300), atol=2)


def test_visibility_is_scale_free_one_on_clear_frames():
    # v is normalised by the amodal box, so on confirmed-clear frames its median is 1 by
    # construction, regardless of how the person's size changes along the path
    for seg in trials.pilot_segments():
        vs = [trials.visibility(i, seg) for i in trials.clear_frames(seg)]
        assert abs(np.median(vs) - 1) < 1e-9
