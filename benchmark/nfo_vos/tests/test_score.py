import json

import numpy as np

from benchmark.nfo_vos import score, trials


def _trial():
    return json.load(open('results/benchmark/pilot/trials.json'))[0]


def test_native_stack_is_mapped_224_stack_passes_through():
    m224 = np.zeros((2, 224, 224), bool)
    assert score.to_224(m224) is m224
    assert score.to_224(np.zeros((2, 600, 800), bool)).shape == (2, 224, 224)


def test_scoring_the_gt_itself_is_perfect():
    t = _trial()
    gt = np.stack([trials.gt_mask(i) for i in t['frames']])
    rows, summary = score.score_trial('oracle', t, gt)
    assert len(rows) == 51 and all(r['J'] == 1 for r in rows)
    assert summary['J@50'] == 1 and summary['DRE'] == 0 and summary['NRE'] == 0


def test_headline_drops_first_and_last_frame_davis_and_flags_init_failure():
    t = _trial()
    gt = np.stack([trials.gt_mask(i) for i in t['frames']])
    P = gt.copy(); P[0] = False; P[-1] = False       # wrong only on the prompted and the last frame
    _, s = score.score_trial('x', t, P)
    assert s['JF'] == 1 and s['J0'] == 0 and s['init_fail']


def test_montage_frames_fit_any_window_length():
    assert score.montage_dts(40) == [0, 8, 16, 23, 31, 39]
    assert score.montage_dts(51)[-1] == 50
