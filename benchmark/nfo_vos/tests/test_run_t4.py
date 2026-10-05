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


def test_prompt_at_t0_is_the_shared_gt_prompt_in_crop_coords():
    t = TRIALS[3]
    pt, box = R.prompt_in_crop(None, 10, 20, gt=t)
    assert np.allclose(pt, (t['point_224'][0] - 10, t['point_224'][1] - 20))
    assert np.allclose(box, np.array(t['box_224']) - [10, 20, 10, 20])


def test_prompt_after_t0_is_the_deepest_point_of_the_own_blob():
    blob = np.zeros((60, 60), bool); blob[10:50, 20:30] = True; blob[5:8, 0:3] = True   # body + speck
    pt, box = R.prompt_in_crop(blob, 5, 5)
    assert blob[int(pt[1]) + 5, int(pt[0]) + 5] and 20 <= pt[0] + 5 < 30 and 10 <= pt[1] + 5 < 50
    assert tuple(box) == (0 - 5, 5 - 5, 30 - 5, 50 - 5)                    # merged blob box
    assert R.prompt_in_crop(np.zeros((60, 60), bool), 0, 0) == (None, None)


def test_box_only_variant_names():
    assert R.methods('box') == ['t4b', 't4b-1']
    assert R.methods() == R.methods('point') and 't4' in R.methods()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_segment_prompted_accepts_box_without_point():
    R.use_base_plus()
    img = np.zeros((100, 100), np.uint8); img[30:70, 40:60] = 200
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        m = R.segment_prompted(img, None, np.array([35, 25, 65, 75], float), fallback=None)
    assert m.shape == (100, 100) and m[50, 50]


def test_select_trials_one_index_for_slurm_arrays():
    ok = [t for t in TRIALS if t['admissible']]
    assert R.select_trials(TRIALS, index=4) == [ok[4]]
    assert R.select_trials(TRIALS, limit=2) == ok[:2]
    assert R.select_trials(TRIALS) == ok
