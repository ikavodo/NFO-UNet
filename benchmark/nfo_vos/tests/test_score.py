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
