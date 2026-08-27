import numpy as np

from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.blob_tracker import detect_blobs
from tracking.core.track_window import _result_from_detections


def track_windows_in_sequence(all_frames: np.ndarray, window_centers, span: int, nth_frame: int,
                              bg_frames: int = 30, var_threshold: float = 16.0,
                              close_kernel_size: int = 6, open_kernel_size: int = 4,
                              min_area: float = 50, min_solidity: float = 0.1,
                              max_dist: float = 12.5, max_age: int = 6, min_track_length: int = 3,
                              merge_radius: float = 50.0, expected_height: float = None,
                              height_tolerance: float = 0.5, adaptive_learning_rate: bool = False):
    """Like calling track_window() once per window, but amortized: MOG2 runs ONE
    continuous pass over the whole sequence (all_frames, in chronological order) instead of
    restarting from zero history for every window. Empirically, this consistently matches
    or beats a curated "person-absent warm-up" snapshot (tested on real NFO footage) - and
    needs no person-absence ground truth at all, since it's just accumulating real history
    as it goes: early windows in the sequence get less benefit, later ones get more, same
    as how an online system would actually behave.

    all_frames: [T_total, H, W] the ENTIRE sequence, in chronological order.
    window_centers: iterable of center frame indices (into all_frames) to produce an
    estimate for. Each must satisfy span <= center < T_total - span.
    span: margin * nth_frame, matching the same convention track_window's caller already
    uses to build a window (e.g. tracking/eval_nfo.py's SPAN).
    bg_frames default (30): a fixed MOG2 history/adaptation-rate value must suit the
    EARLIEST window you care about, since any window occurring before bg_frames real
    frames have elapsed is under-adapted (verified empirically: bg_frames=100 left a
    window at frame 38 badly broken - 697px off - while bg_frames=30 got the same window
    to 9px, with no cost to later windows; more history never hurts once enough real
    frames exist, it only hurts when it exceeds what's actually elapsed). So: keep this
    at or below the earliest center you'll query, not tuned to the "ideal" steady-state
    history for a long sequence.

    adaptive_learning_rate: an alternative to a well-chosen constant bg_frames - tried a
    1/frames_seen_so_far decay schedule to handle early windows automatically, but it
    over-weights whichever frame happens to be first (learningRate=1.0 there) and gave
    mixed results (helped one window, regressed another quite badly). Defaults to False;
    a correctly-sized constant bg_frames outperformed it in every case tested. Kept as an
    option in case it's worth revisiting with a gentler ramp, not because it's recommended
    now.

    Returns {center: result_dict_or_None} for every center in window_centers, in the same
    format track_window() returns for a single window.
    """
    masks_all = foreground_mask(all_frames, bg_frames=bg_frames, var_threshold=var_threshold,
                                adaptive_learning_rate=adaptive_learning_rate)
    masks_all = refine_mask(masks_all, close_kernel_size=close_kernel_size, open_kernel_size=open_kernel_size)
    masks_all = filter_by_shape(masks_all, min_area=min_area, min_solidity=min_solidity)

    results = {}
    for center in window_centers:
        frame_indices = list(range(center - span, center + span + 1, nth_frame))
        window_detections = detect_blobs(masks_all[frame_indices], min_area=min_area)
        results[center] = _result_from_detections(window_detections, min_track_length, expected_height,
                                                   height_tolerance, max_dist, max_age, merge_radius)
    return results
