import json

import numpy as np
import pytest
import torch

from benchmark.nfo_vos import run_t4 as R
from tracking.eval.gt_sam_gate import visibility_within_mask

TRIALS = json.load(open('results/benchmark/pilot/trials.json'))


def test_vote_m1_is_union_mN_is_intersection_and_m_clips_to_N():
    a = np.zeros((3, 8, 8), bool); a[0, :4] = True; a[1, 2:6] = True; a[2, 3:5] = True
    assert (R.blob_vote(a, 1) == a.any(0)).all()
    assert (R.blob_vote(a, 3) == a.all(0)).all()
    assert (R.blob_vote(a[:2], 7) == a[:2].all(0)).all()      # buffer shorter than m


def test_single_frame_reference_makes_visibility_the_identity():
    # T4-1: reference = current frame -> residual 0 -> the visibility step passes S unchanged
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (40, 40), dtype=np.uint8)
    S = np.zeros((40, 40), bool); S[10:30, 12:25] = True
    assert (visibility_within_mask(img[None], img, S)[0] == S).all()


def test_track_at_t0_starts_inside_gt_box():
    for t in TRIALS[:4]:
        tr = R.track(t)
        x, y = tr['chain'][0][:2]
        x0, y0, x1, y1 = t['box_224']
        assert x0 <= x <= x1 and y0 <= y <= y1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_causal_output_at_t_ignores_later_frames():
    t = TRIALS[6]
    full = R.run_trial(t, max_frames=12)
    short = R.run_trial(t, max_frames=9)
    for k in full:
        assert (full[k]['masks'][:9] == short[k]['masks']).all(), k
