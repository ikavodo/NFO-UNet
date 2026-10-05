from benchmark.nfo_vos import diag_integration as D
from benchmark.nfo_vos import run_t4 as R


def test_buffer_is_causal_strided_and_ends_at_t():
    assert D.buffer_indices(20, 3) == [2, 5, 8, 11, 14, 17, 20]
    assert D.buffer_indices(4, 3) == [1, 4]                       # before t0 nothing exists


def test_velocity_uses_every_detection_in_the_strided_span():
    chain = {k: (2.0 * k, 5.0) for k in range(0, 13)}            # exactly 2 px/frame
    x, y, vx = R.readout_line(chain, 12, span=13)
    assert abs(vx - 2.0) < 1e-9 and abs(x - 24.0) < 1e-9
