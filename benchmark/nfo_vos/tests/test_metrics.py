import numpy as np

from benchmark.nfo_vos import metrics


def _rand_masks(seed):
    rng = np.random.default_rng(seed)
    P = np.zeros((60, 60), bool); G = np.zeros((60, 60), bool)
    P[10:40, 12:35] = True; G[18:50, 20:45] = True
    P &= rng.random(P.shape) > 0.2
    return P, G


def test_jaccard_is_harmonic_combination_of_precision_and_recall():
    for s in range(5):
        m = metrics.frame_metrics(*_rand_masks(s))
        assert np.isclose(1 / m['J'], 1 / m['p'] + 1 / m['r'] - 1)


def test_empty_prediction_on_empty_gt_scores_one_davis_convention():
    z = np.zeros((20, 20), bool)
    assert metrics.frame_metrics(z, z)['J'] == 1


def test_dre_counts_disjoint_nonempty_nre_counts_empty():
    G = np.zeros((3, 20, 20), bool); G[:, 5:10, 5:10] = True
    P = np.zeros_like(G)
    P[0, 5:10, 5:10] = True       # hit
    P[1, 12:18, 12:18] = True     # drift: non-empty, disjoint
    # P[2] empty                  # not reported
    dre, nre = metrics.dre_nre(P, G)
    assert np.isclose(dre, 1 / 3) and np.isclose(nre, 1 / 3)


def test_pnorm_perfect_centre_is_one_and_empty_is_zero():
    G = np.zeros((2, 50, 50), bool); G[:, 10:30, 10:30] = True
    boxes = [(10, 10, 30, 30)] * 2          # x0, y0, x1, y1 in 224/px space
    assert np.isclose(metrics.pnorm(G, boxes), 1.0)
    assert metrics.pnorm(np.zeros_like(G), boxes) == 0.0


def test_decay_positive_for_decreasing_scores():
    assert metrics.decay(np.linspace(1, 0, 51)) > 0.5


def test_native_to_224_maps_through_padded_square():
    m = np.zeros((600, 800), bool); m[200:400, 300:500] = True
    out = metrics.native_to_224(m)
    assert out.shape == (224, 224) and out.dtype == bool
    ys, xs = np.nonzero(out)
    # native centre (399.5, 299.5) -> padded (399.5, 399.5) -> x 0.28
    assert abs(xs.mean() - 111.9) < 1 and abs(ys.mean() - 111.9) < 1
