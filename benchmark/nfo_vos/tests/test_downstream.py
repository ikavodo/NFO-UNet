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


from benchmark.nfo_vos.downstream import visible_box as VB

GT = np.array([20., 10., 59., 89.])          # x 20..59, y 10..89


def test_visible_box_no_occluder():
    b, fb = VB.visible_box(GT, np.full((100, 100), 0.3), q=0.5, rho=0.5)
    np.testing.assert_array_equal(b, GT)
    assert not fb


def test_visible_box_top_occluded():
    d = np.zeros((100, 100))
    d[10:50, :] = 1.0                        # near foliage over the top half of the box (rows 10..49)
    b, fb = VB.visible_box(GT, d, q=0.5, rho=0.5)
    assert abs(b[1] - 50) <= 1 and b[3] == 89 and b[0] == 20 and b[2] == 59 and not fb


def test_visible_box_all_trimmed_falls_back():
    d = np.zeros((100, 100))
    d[10:90, 20:60] = np.random.default_rng(0).random((80, 40))
    b, fb = VB.visible_box(GT, d, q=0.5, rho=1.01)   # no row/column can reach support > 1 → everything trimmed
    np.testing.assert_array_equal(b, GT)
    assert fb


from benchmark.nfo_vos.downstream import recognition as RC


def test_composite_no_mask_is_bg():
    bg = np.full((20, 30, 3), 7, np.uint8)
    imgs = [np.full((20, 30, 3), 200, np.uint8)] * 3
    out = RC.masked_composite(imgs, [np.zeros((20, 30), bool)] * 3, [(0, 0)] * 3, bg)
    np.testing.assert_array_equal(out, bg)


def test_composite_shift():
    im = np.zeros((20, 30, 3), np.uint8)
    im[5, 10] = 255
    out = RC.masked_composite([im], [np.ones((20, 30), bool)], [(5, 0)], np.zeros_like(im))
    assert out[5, 15, 0] == 255 and out[5, 10, 0] == 0


def test_composite_uniform_mean():
    a, b = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 100, np.uint8)
    out = RC.masked_composite([a, b], [np.ones((4, 4), bool)] * 2, [(0, 0)] * 2, np.zeros_like(a))
    assert np.all(out == 50)


def test_gt_shifts_zero_at_t():
    boxes = {1: np.array([0., 0., 10., 10.]), 2: np.array([4., 2., 14., 12.])}
    s = RC.gt_shifts(boxes, [1, 2], 2)
    assert s[-1] == (0.0, 0.0) and s[0] == (4.0, 2.0)
