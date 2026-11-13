import numpy as np, torch
from benchmark.nfo_vos.ape.foliage import near_alpha, best_window, N


def test_near_alpha_half_at_rank_08_and_monotone():
    D = torch.arange(1001.).reshape(1, 1, 1001)
    a = near_alpha(D)[0, 0]
    assert abs(a[800].item() - 0.5) < 1e-3
    assert (a[1:] >= a[:-1]).all()
    assert a[700] < 0.05 and a[900] > 0.95


def test_best_window_avoids_ape_and_nocut_covers_dilated_box():
    boxes = {k: [[100, 100, 200, 200]] for k in range(0, 60)}          # ape present in frames 0..59 only
    start, nocut = best_window(boxes, 200, 404, 720)
    assert start >= 60 and not nocut.any()
    start, nocut = best_window({k: [[100, 100, 200, 200]] for k in range(200)}, 200, 404, 720)
    assert nocut[85:215, 85:215].all() and not nocut[:80].any() and not nocut[:, 220:].any()
