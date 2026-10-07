import numpy as np
import pytest
import torch

from benchmark.nfo_vos import a4


def test_recency_weights_sum_to_one_and_favour_the_newest_frame():
    w = a4.recency_weights(7)
    assert np.isclose(w.sum(), 1) and np.all(np.diff(w) > 0)          # oldest first, newest last
    assert np.isclose(w[-1] / w[-2], np.exp(1 / 3.5))


def test_recency_fusion_of_a_static_scene_reproduces_it():
    img = np.random.default_rng(0).integers(0, 255, (60, 80)).astype(np.uint8)
    out = a4.recency_fusion([img] * 7, [0.0] * 7)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_fixed_schedule_inserts_every_kth_frame():
    assert a4.schedule_fixed(10, k=4) == [0, 4, 8]


def test_gated_schedule_uses_only_past_confidence_and_caps_density():
    piou = np.array([0.45, 0.40, 0.42, 0.41, 0.44, 0.43, 0.40, 0.88, 0.90, 0.30, 0.30, 0.30])
    s = a4.schedule_gated(piou, tau=0.6, k=4)
    assert s == [0, 4, 10]                    # t=0 from frame 0's own (prompted) confidence,
    #                                            later t from t-1; never two within 4 frames;
    #                                            frame 8 is not inserted (piou[7]=0.88 is confident)
    assert a4.schedule_gated(piou, tau=0.1, k=4) == []      # confident everywhere -> baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_empty_schedule_reproduces_the_plain_run_bit_for_bit():
    import json
    from benchmark.nfo_vos import run_sam2_video as R
    trial = json.load(open('results/benchmark/pilot/trials.json'))[5]
    pred = R.build_predictor('b0')
    plain = R.run_trial(pred, trial, max_frames=4)
    sched = R.run_trial(pred, trial, max_frames=4, schedule=[], comp_dir=None)
    assert (plain['masks'] == sched['masks']).all() and not sched['inserted'].any()
