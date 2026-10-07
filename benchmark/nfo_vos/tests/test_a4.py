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
    trial = json.load(open('results/benchmark/rr/trials.json'))[5]
    pred = R.build_predictor('b0')
    plain = R.run_trial(pred, trial, max_frames=4)
    sched = R.run_trial(pred, trial, max_frames=4, schedule=[], comp_dir=None)
    assert (plain['masks'] == sched['masks']).all() and not sched['inserted'].any()


def test_velocity_from_the_causal_chain_window_in_native_px():
    from benchmark.nfo_vos import a4_build as B
    chain = {k: (2.0 * k, 5.0) for k in range(0, 12)}               # 2 px/frame in 224 space
    assert np.isclose(B.velocity_at(chain, 10, fallback=-9.0), 2.0 * 800 / 224)
    assert B.velocity_at({0: (1.0, 5.0), 1: (3.0, 5.0)}, 1, fallback=-9.0) == -9.0   # < 3 points


def test_build_writes_native_composites_for_requested_times(tmp_path):
    import json, cv2
    from benchmark.nfo_vos import a4_build as B
    trial = [t for t in json.load(open('results/benchmark/rr/trials.json')) if t['admissible']][3]
    B.build(trial, [0, 4], str(tmp_path))
    imgs = [cv2.imread(str(tmp_path / f'{t:02d}.jpg'), 0) for t in (0, 4)]
    assert all(im is not None and im.shape == (600, 800) for im in imgs)


def test_runner_schedule_for_fixed_and_gated(tmp_path):
    import json
    from benchmark.nfo_vos import run_sam2_video as R
    trial = {'id': 'x', 'frames': list(range(40))}
    assert R.schedule_for(trial, 'fixed', None, None) == list(range(0, 40, 4))
    np.savez(tmp_path / 'x.npz', pred_iou=np.r_[0.4, np.full(39, 0.9)])
    assert R.schedule_for(trial, 'gated', 0.6, str(tmp_path / 'x.npz')) == [0]
