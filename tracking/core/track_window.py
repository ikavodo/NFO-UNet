import numpy as np

from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.blob_tracker import detect_blobs, track_blobs, score_and_fit, merged_center


def position_from_track(winner, window_detections, merge_radius, center_t: int = None,
                        return_box: bool = False):
    """Read a center-frame (x, y) position off one scored track. Split out of
    _result_from_detections so the same read-out can be applied to *any* candidate track,
    not only the winning one - needed to study the ranking offline
    (tracking/eval/stage2_rank_learning.py). Behavior is unchanged.

    center_t: which window index to read the position out at. Defaults to the window
    center, which is the only choice the offline evaluator ever uses. It is exposed so a
    streaming caller can measure the cost of reading out at the newest frame instead
    (zero lookahead latency): the linear fit below is then evaluated at the edge of its own
    support rather than its center, where its prediction variance is minimal.
    return_box: also report the merged multi-blob box, not only its center."""
    if center_t is None:
        center_t = len(window_detections) // 2
    extrapolated = center_t not in winner["history"]
    if not extrapolated:
        anchor_x, anchor_y = winner["history"][center_t][:2]
    else:
        # center frame had no detection in the winning track -> extrapolate x from the
        # fitted line, and use the mean y (this tracker's motion model is horizontal-only)
        frames_arr = np.array(winner["frames"])
        xs = np.array([winner["history"][f][0] for f in winner["frames"]])
        A = np.vstack([frames_arr, np.ones(len(frames_arr))]).T
        coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
        anchor_x = coef[0] * center_t + coef[1]
        ys = np.array([winner["history"][f][1] for f in winner["frames"]])
        anchor_y = ys.mean()

    # a single frame's mask commonly fragments a person into disconnected blobs
    # (head/torso/legs) - merge everything near the tracked anchor point into one
    # combined bbox so the reported position matches a whole-person centroid, not
    # whichever fragment score_and_fit happened to track
    merged = merged_center(window_detections[center_t], anchor_x, anchor_y, merge_radius,
                           return_box=return_box)
    (cx, cy, box) = merged if return_box else (merged[0], merged[1], None)

    return dict(x=float(cx), y=float(cy), vx=float(winner["vx"]),
               score=float(winner["score"]), resid_std=float(winner["resid_std"]),
               box=box, extrapolated=extrapolated, winner=winner)


def _result_from_detections(window_detections, min_track_length, expected_height, height_tolerance,
                            max_dist, max_age, merge_radius, center_t: int = None,
                            return_box: bool = False):
    """Shared by track_window() and track_sequence.track_windows_in_sequence() - given one
    window's already-computed per-frame detections, run Kalman/Hungarian tracking +
    scoring and read off a center-frame position estimate. Returns the same result dict
    both callers expose, or None if no track of sufficient length was found."""
    tracks = track_blobs(window_detections, max_dist=max_dist, max_age=max_age)
    winner = score_and_fit(tracks, min_track_length=min_track_length,
                           expected_height=expected_height, height_tolerance=height_tolerance)
    if winner is None:
        return None
    return position_from_track(winner, window_detections, merge_radius, center_t=center_t,
                               return_box=return_box)


def track_window(frames: np.ndarray, bg_frames: int = None, var_threshold: float = 16.0,
                 close_kernel_size: int = 6, open_kernel_size: int = 4,
                 min_area: float = 50, min_solidity: float = 0.1,
                 max_dist: float = 12.5, max_age: int = 6, min_track_length: int = 3,
                 merge_radius: float = 50.0, expected_height: float = None,
                 height_tolerance: float = 0.5, warmup_frames: np.ndarray = None):
    """Run bg-sub -> morph refine -> shape filter -> Kalman/Hungarian tracking over one
    [T, H, W] uint8 grayscale window, and return a person-centroid estimate for the
    window's center frame.

    max_dist default (12.5px) is rescaled from master_thesis's 1024x1024-tuned value
    (80px) to KTH's native 160x120 resolution by width ratio - see
    docs/superpowers/specs/2026-08-25-blob-tracking-design.md, "Hyperparameters". Retune
    empirically per dataset resolution, don't assume this transfers as-is.

    merge_radius default (50px) is NOT resolution-rescaled from master_thesis (that
    approach was tried and measured wrong - see git history) - it's derived directly from
    KTH's own ground truth: person height in data/kth_staged is ~90-95px in a 120px-tall
    frame (person fills ~78% of frame height), so a fragment near one end of the person
    (e.g. head) needs roughly half that height to reach a fragment at the other end (e.g.
    feet). Resolution ratio is the wrong basis for this parameter - it depends on camera
    framing/zoom, not raw pixel dimensions.

    expected_height/height_tolerance: optional shape-aware scoring - see score_and_fit's
    docstring. Motion consistency alone can't distinguish a person from any other
    smoothly-moving blob (e.g. wind-blown foliage); pass expected_height (measured typical
    person height in this frame's pixel space) to penalize tracks of the wrong size.
    Defaults to None (no penalty, original behavior).

    warmup_frames: optional [Tw, H, W] stack of frames from the same (static-camera)
    sequence, run through MOG2 first and discarded, so it has real accumulated history
    before this window is scored - see preprocess.foreground_mask's docstring. For
    processing many windows from the same sequence, track_sequence.track_windows_in_sequence
    is far cheaper (one continuous pass, no per-window re-warm) and was empirically found to
    work as well or better than a curated warmup_frames snapshot - prefer it when you have
    more than one window to process from the same sequence.

    Returns a dict with keys 'x', 'y', 'vx', 'score', 'resid_std', or None if no track of
    at least min_track_length frames was found in the window.
    """
    masks = foreground_mask(frames, bg_frames=bg_frames, var_threshold=var_threshold,
                            warmup_frames=warmup_frames)
    masks = refine_mask(masks, close_kernel_size=close_kernel_size, open_kernel_size=open_kernel_size)
    masks = filter_by_shape(masks, min_area=min_area, min_solidity=min_solidity)
    detections = detect_blobs(masks, min_area=min_area)
    return _result_from_detections(detections, min_track_length, expected_height, height_tolerance,
                                   max_dist, max_age, merge_radius)
