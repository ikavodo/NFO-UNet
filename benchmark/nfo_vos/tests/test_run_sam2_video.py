import json

import numpy as np
import pytest
import torch

from benchmark.nfo_vos import run_sam2_video as R


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_b0_smoke_three_frames_native_res_and_hits_prompt_box(tmp_path):
    trial = json.load(open('results/benchmark/rr/trials.json'))[5]
    out = R.run_trial(R.build_predictor('b0'), trial, max_frames=3)
    assert out['masks'].shape == (3, 600, 800) and out['masks'].dtype == bool
    assert out['pred_iou'].shape == (3,) and np.isfinite(out['pred_iou']).all()   # SAM2's own confidence
    assert out['obj_score'].shape == (3,)
    x0, y0, x1, y1 = (int(round(v)) for v in trial['box_native'])
    m0 = out['masks'][0]
    assert m0[y0:y1, x0:x1].sum() > 0.8 * m0.sum() > 0      # prompted frame lands in its box


def test_select_trials_one_index_for_slurm_arrays():
    trials = json.load(open('results/benchmark/rr/trials.json'))
    ok = [t for t in trials if t['admissible']]
    assert R.select_trials(trials, index=7) == [ok[7]]
    assert R.select_trials(trials, limit=2) == ok[:2] and R.select_trials(trials) == ok


def test_gt_mask_native_inverts_the_padded_square_downscale():
    from benchmark.nfo_vos import metrics, trials as TR
    t = json.load(open('results/benchmark/rr/trials.json'))[5]
    m = R.gt_mask_native(t['seq'], t['t0'])
    assert m.shape == (600, 800) and m.dtype == bool
    g = TR.gt_mask(t['t0'], t['seq']); back = metrics.native_to_224(m)
    assert (g & back).sum() / (g | back).sum() > 0.85            # round trip 224 -> native -> 224


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU')
def test_oracle_mask_init_starts_on_the_gt():
    from benchmark.nfo_vos import metrics, trials as TR
    t = json.load(open('results/benchmark/rr/trials.json'))[5]
    out = R.run_trial(R.build_predictor('b0'), t, max_frames=3, init_mask=True)
    g = TR.gt_mask(t['t0'], t['seq']); p = metrics.native_to_224(out['masks'][0])
    assert (g & p).sum() / (g | p).sum() > 0.8
