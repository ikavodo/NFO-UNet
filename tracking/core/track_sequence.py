import numpy as np

from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.blob_tracker import detect_blobs, _Track
from tracking.core.track_window import _result_from_detections

# Every pixel parameter of this pipeline, expressed as a multiple of the measured person
# height (preprocess.estimate_person_height) instead of an absolute pixel count. Calibrated
# once, on multi-scale KTH, and intended to be reused at any scale - that is the whole
# point. Measured effect: worst-bucket accuracy across a 4x range of person sizes went from
# 5.9% (one dataset's absolute constants, reused everywhere) to 75.0%, and the spread
# across scales from 74pp to 18pp. See docs/deepsort_blob_scoring_compatibility.md.
#
# ALPHA_MAX_DIST is the association gate. Ground-truth motion is only ~0.10-0.13 of a body
# height per step, but blob centroids also jump between fragments of the same person, so the
# gate must be looser than real motion. It is a very flat optimum (0.15-0.40 changes results
# by ~2pp) - it mainly has to be big enough.
# ALPHA_MERGE is the radius over which fragments of one person are merged. It is the
# sensitive one, and it is where residual scale-dependence still lives.
ALPHA_MAX_DIST = 0.25
ALPHA_EXP_HEIGHT = 0.95   # estimate_person_height lands within ~5% of true height
ALPHA_MERGE = 0.625       # swept directly on real NFO (tracking/eval/merge_radius_sweep.py):
                          # 0.625 gives mean residual 0.0660 / hit 90.0%, against 0.0704 /
                          # 89.6% at eval_nfo's old 0.5 and 0.0671 / 89.1% at the 0.75 the
                          # synthetic sweep had preferred. Note the risk here is asymmetric in
                          # the OPPOSITE direction to the association gate: too large is
                          # catastrophic (1.5x -> 51.3% hit) while too small degrades gently
                          # (0.0x -> 84.7%), so "bias loose" is not a universal rule.
ALPHA_MIN_AREA = 50.0 / 120.0 ** 2   # base min_area 50px^2 at a 120px person
H_CALIB = 120.0           # person height the base morphology/Kalman values were tuned at


def scale_relative_params(person_height: float):
    """One measured person height in -> every scale-dependent parameter out.

    Returns (kwargs for track_windows_in_sequence, (P_VAR, Q_VAR, R_VAR) for _Track). The
    Kalman variances are pixel-variance constants, so they scale with the square of the
    length ratio. Callers normally get this applied for them by passing person_height to
    track_windows_in_sequence.
    """
    rel = person_height / H_CALIB
    kwargs = dict(max_dist=ALPHA_MAX_DIST * person_height,
                  merge_radius=ALPHA_MERGE * person_height,
                  expected_height=ALPHA_EXP_HEIGHT * person_height,
                  min_area=ALPHA_MIN_AREA * person_height ** 2,
                  close_kernel_size=max(2, int(round(6 * rel))),
                  open_kernel_size=max(1, int(round(4 * rel))))
    return kwargs, (50.0 * rel ** 2, 2.0 * rel ** 2, 9.0 * rel ** 2)


def track_windows_in_sequence(all_frames: np.ndarray, window_centers, span: int, nth_frame: int,
                              bg_frames: int = 30, var_threshold: float = 16.0,
                              close_kernel_size: int = 6, open_kernel_size: int = 4,
                              min_area: float = 50, min_solidity: float = 0.1,
                              max_dist: float = 12.5, max_age: int = 6, min_track_length: int = 3,
                              merge_radius: float = 50.0, expected_height: float = None,
                              height_tolerance: float = 0.5, adaptive_learning_rate: bool = False,
                              person_height: float = None):
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

    person_height: the measured typical person height in pixels for this sequence, from
    preprocess.estimate_person_height. When given, it OVERRIDES max_dist, merge_radius,
    expected_height, min_area and both kernel sizes with scale-relative values
    (scale_relative_params) and rescales the Kalman covariances to match, so the caller
    supplies one measured number instead of eight absolute-pixel constants that are only
    valid at the resolution and camera distance they were measured at. Pass
    expected_height=None alongside it to keep shape-aware scoring switched off.
    Defaults to None, i.e. the original explicit-constants behavior.

    Returns {center: result_dict_or_None} for every center in window_centers, in the same
    format track_window() returns for a single window.
    """
    kalman_restore = None
    if person_height is not None:
        derived, kalman = scale_relative_params(person_height)
        if expected_height is None:
            derived.pop('expected_height')  # caller explicitly wants no shape term
        max_dist, merge_radius = derived['max_dist'], derived['merge_radius']
        expected_height = derived.get('expected_height', None)
        min_area = derived['min_area']
        close_kernel_size, open_kernel_size = derived['close_kernel_size'], derived['open_kernel_size']
        kalman_restore = (_Track.P_VAR, _Track.Q_VAR, _Track.R_VAR)
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = kalman

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
    if kalman_restore is not None:
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = kalman_restore
    return results
