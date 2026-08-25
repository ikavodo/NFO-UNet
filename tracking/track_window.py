import numpy as np

from tracking.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.blob_tracker import detect_blobs, track_blobs, score_and_fit, merged_center


def track_window(frames: np.ndarray, bg_frames: int = 5, var_threshold: float = 16.0,
                 close_kernel_size: int = 6, open_kernel_size: int = 4,
                 min_area: float = 50, min_solidity: float = 0.1,
                 max_dist: float = 12.5, max_age: int = 6, min_track_length: int = 3,
                 merge_radius: float = 50.0):
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

    Returns a dict with keys 'x', 'y', 'vx', 'score', 'resid_std', or None if no track of
    at least min_track_length frames was found in the window.
    """
    masks = foreground_mask(frames, bg_frames=bg_frames, var_threshold=var_threshold)
    masks = refine_mask(masks, close_kernel_size=close_kernel_size, open_kernel_size=open_kernel_size)
    masks = filter_by_shape(masks, min_area=min_area, min_solidity=min_solidity)
    detections = detect_blobs(masks, min_area=min_area)
    tracks = track_blobs(detections, max_dist=max_dist, max_age=max_age)
    winner = score_and_fit(tracks, min_track_length=min_track_length)
    if winner is None:
        return None

    center_t = frames.shape[0] // 2
    if center_t in winner["history"]:
        anchor_x, anchor_y = winner["history"][center_t]
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
    cx, cy = merged_center(detections[center_t], anchor_x, anchor_y, merge_radius)

    return dict(x=float(cx), y=float(cy), vx=float(winner["vx"]),
               score=float(winner["score"]), resid_std=float(winner["resid_std"]))
