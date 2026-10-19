"""Downstream benchmark scorers (docs/vos_benchmark/plan_downstream_benchmark.md)."""
import numpy as np

from benchmark.nfo_vos.downstream import box_metrics as BM


def test_mask_box_empty():
    assert BM.mask_box(np.zeros((60, 80), bool)) is None


def test_mask_box_two_blobs():
    m = np.zeros((100, 100), bool)
    m[10:12, 10:12] = True          # rows = y, cols = x
    m[79:81, 49:51] = True
    np.testing.assert_array_equal(BM.mask_box(m), [10, 10, 50, 80])   # xyxy, inclusive max


def test_ope_perfect():
    gt = [np.array([10., 20., 60., 120.])] * 5
    r = BM.ope(gt, gt)
    assert r['p20'] == 1 and r['pnorm'] == 1 and r['fail'] == 0
    # IoU = 1 passes every overlap threshold 0, 0.05, …, 0.95 (IoU > thr) but not 1.0 → 20/21
    assert abs(r['auc'] - 20 / 21) < 1e-9


def test_ope_lost():
    gt = [np.array([300., 200., 360., 400.])] * 4
    r = BM.ope([None] * 4, gt)
    assert r['auc'] == 0 and r['fail'] == 1 and r['p20'] == 0
