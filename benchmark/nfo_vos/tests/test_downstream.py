"""Downstream benchmark scorers (docs/vos_benchmark/plan_downstream_benchmark.md)."""
import numpy as np

from benchmark.nfo_vos.eval import box_metrics as BM
from benchmark.nfo_vos.eval import recognition as RC


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


def test_stack_no_mask_is_bg():
    bg = np.full((20, 30, 3), 7, np.uint8)
    imgs = [np.full((20, 30, 3), 200, np.uint8)] * 3
    out = RC.masked_stack(imgs, [np.zeros((20, 30), bool)] * 3, [(0, 0)] * 3, bg, fill='bg')
    np.testing.assert_array_equal(out, bg)


def test_stack_shift():
    im = np.zeros((20, 30, 3), np.uint8)
    im[5, 10] = 255
    out = RC.masked_stack([im], [np.ones((20, 30), bool)], [(5, 0)], np.zeros_like(im))
    assert out[5, 15, 0] == 255 and out[5, 10, 0] == 0


def test_stack_uniform_mean():
    a, b = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 100, np.uint8)
    out = RC.masked_stack([a, b], [np.ones((4, 4), bool)] * 2, [(0, 0)] * 2, np.zeros_like(a))
    assert np.all(out == 50)


def test_gt_shifts_zero_at_t():
    boxes = {1: np.array([0., 0., 10., 10.]), 2: np.array([4., 2., 14., 12.])}
    s = RC.gt_shifts(boxes, [1, 2], 2)
    assert s[-1] == (0.0, 0.0) and s[0] == (4.0, 2.0)


def test_stack_mean_fill_where_unmasked():
    a, b = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 100, np.uint8)
    m = np.zeros((4, 4), bool); m[0, 0] = True
    out = RC.masked_stack([a, b], [m, np.zeros((4, 4), bool)], [(0, 0)] * 2, np.full_like(a, 7))
    assert out[0, 0, 0] == 0 and np.all(out[1:, :, 0] == 50)      # masked pixel: frame a only; elsewhere plain mean
