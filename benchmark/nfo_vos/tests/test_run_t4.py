import json

import numpy as np
import pytest
import torch

from benchmark.nfo_vos import run_t4 as R

TRIALS = json.load(open('results/benchmark/rr/trials.json'))


def test_track_at_t0_starts_inside_gt_box():
    for t in TRIALS[:4]:
        x, y = R.track(t)['chain'][0][:2]
        x0, y0, x1, y1 = t['box_224']
        assert x0 <= x <= x1 and y0 <= y <= y1


def test_detect_and_track_returns_one_detection_list_per_frame():
    t = TRIALS[6]
    out = R.detect_and_track(t['frames'][:5], t['warmup'], t['box_224'][3] - t['box_224'][1])
    assert out['frames'].shape[0] == 5 and len(out['dets']) == 5 and 'tracks' in out


def test_prompt_from_blob_is_the_deepest_point_and_merged_box():
    blob = np.zeros((60, 60), bool); blob[10:50, 20:30] = True; blob[5:8, 0:3] = True   # body + speck
    pt, box = R.prompt_from_blob(blob)
    assert blob[int(pt[1]), int(pt[0])] and 20 <= pt[0] < 30 and 10 <= pt[1] < 50
    assert tuple(box) == (0, 5, 30, 50)
    assert R.prompt_from_blob(np.zeros((60, 60), bool)) == (None, None)


def test_box_224_to_native_maps_edges_through_the_padded_square():
    assert np.allclose(R.box_to_native((0, 0, 224, 224)), (0, -100, 800, 700))


def test_select_trials_one_index_for_slurm_arrays():
    ok = [t for t in TRIALS if t['admissible']]
    assert R.select_trials(TRIALS, index=4) == [ok[4]]
    assert R.select_trials(TRIALS, limit=2) == ok[:2] and R.select_trials(TRIALS) == ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_t41n_is_causal_output_at_t_ignores_later_frames():
    t = TRIALS[6]
    full = R.run_trial(t, max_frames=5)['masks']
    short = R.run_trial(t, max_frames=3)['masks']
    assert (full[:3] == short).all()
