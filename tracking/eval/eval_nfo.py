import os
import sys

import cv2
import numpy as np

from tracking.core.preprocess import estimate_person_height
from tracking.core.track_sequence import track_windows_in_sequence

def _detect_layout():
    """Find whichever NFO layout is actually on disk. Two exist and they differ in more than the
    path:

      nfo_final     data/nfo_final/nfo_final/seqN/, 800x600, *.jpg, TWO groundtruth files - a
                    `groundtruth.txt` plus a second `groundtruth*` holding one DENSE line of
                    `x,y,w,h` per frame with x<0 meaning absent. This is what the published
                    numbers (0.0661 mean residual, 90.0% hit) were measured on, and what
                    MAX_DIST/MERGE_RADIUS/EXPECTED_HEIGHT below were calibrated at.
      nfo_processed data/nfo_processed/seqN_gt/, 224x224, *_or.jpg frames alongside
                    *_sammask.png, and ONE groundtruth.txt with SPARSE five-column
                    `frame,x,y,w,h` rows covering only labelled frames.

    Returns (root, seqs, calibrated) where `calibrated` says whether the absolute pixel constants
    apply to this layout.
    """
    for root, pat, calibrated in (('data/nfo_final/nfo_final', 'seq%d', True),
                                  ('data/nfo_processed', 'seq%d_gt', False)):
        seqs = [pat % i for i in (1, 2, 3, 4) if os.path.isdir(os.path.join(root, pat % i))]
        if seqs:
            return root, seqs, calibrated
    return 'data/nfo_final/nfo_final', ['seq%d' % i for i in (1, 2, 3, 4)], True


IN_DIR, SEQS, CALIBRATED_LAYOUT = _detect_layout()
SEQ_SIZE = 7  # matches config/train_config.py's seq_size=7
NTH_FRAME = 2  # matches config/train_config.py's nth_frame=2
MARGIN = SEQ_SIZE // 2
SPAN = MARGIN * NTH_FRAME

# derived from NFO's own native-resolution (800x600) ground truth, not reused/rescaled
# from KTH - see conversation for the measurement:
# - MAX_DIST from measured GT centroid displacement at nth_frame=2 stride (p99 ~= 25px)
# - MERGE_RADIUS was measured person height / 2 = 100px until it was swept directly
#   (tracking/eval/merge_radius_sweep.py): 0.625 x height = 122px is better, and 100px was
#   slightly too TIGHT, not too loose as had been assumed. Baseline numbers quoted in docs
#   from before 2026-08-28 used 100px; see docs/scale_generalization_plan.md for both.
# These are ABSOLUTE PIXEL constants, valid only at this dataset's resolution and camera
# distance. Reusing them on footage where people appear at a different pixel size fails
# badly (measured: accuracy 91% -> 6% over a 2x change in person size). The
# scale_relative=True path below replaces all of them with multiples of a person height
# measured from the footage itself, and needs no ground truth to do it - see
# docs/deepsort_blob_scoring_compatibility.md, "Step 1b".
MAX_DIST = 25.0
MERGE_RADIUS = 122.0  # = 0.625 * EXPECTED_HEIGHT, matching ALPHA_MERGE in track_sequence.py
EXPECTED_HEIGHT = 195.0  # measured mean NFO person height at native 800x600 resolution
BG_FRAMES = 30  # must suit the earliest window queried - see track_sequence's docstring

if not CALIBRATED_LAYOUT:
    print(f"NOTE: using {IN_DIR} ({', '.join(SEQS)}). MAX_DIST/MERGE_RADIUS/EXPECTED_HEIGHT above\n"
          f"were calibrated on nfo_final at 800x600 and DO NOT transfer here: these frames are\n"
          f"224x224 AND a different field of view (the ground-truth person is 0.21 of frame height\n"
          f"against nfo_final's 0.33), so neither a resolution ratio nor a crop factor rescales\n"
          f"them correctly - which is exactly the failure F1 documents. Run the 'relative' config,\n"
          f"which measures person height off the footage and needs no such constant.")


def parse_normalized_bbs(file_path):
    """Ground-truth boxes as a list INDEXED BY FRAME, with None where the person is absent or
    unlabelled. Both layouts return that same shape, so every caller can keep indexing by frame.

    Dense four-column (nfo_final): one line per frame, `x,y,w,h`, x<0 meaning absent.
    Sparse five-column (nfo_processed): `frame,x,y,w,h` for labelled frames only, so the list is
    padded with None up to the highest labelled index. Padding rather than compacting matters:
    the caller's `boxes[center]` is a FRAME index, and compacting would silently shift every box
    against its image.
    """
    rows = [ln.strip().split(',') for ln in open(file_path) if ln.strip()]
    if rows and len(rows[0]) == 5:
        keyed = {int(r[0]): tuple(float(v) for v in r[1:]) for r in rows}
        return [keyed.get(i) for i in range(max(keyed) + 1)]
    boxes = []
    for r in rows:
        x, y, w, h = (float(v) for v in r)
        boxes.append(None if x < 0 else (x, y, w, h))
    return boxes


def seq_dir(seq: str) -> str:
    return os.path.join(IN_DIR, seq)


def load_frames(seq: str, up_to: int = None, indices=None) -> np.ndarray:
    """Greyscale frames for one sequence, from whichever layout _detect_layout found. Shared so
    the path and the frame-naming live in ONE place: five files each had their own copy of this
    against a hardcoded data/nfo_final/nfo_final, and all five broke together when the data
    changed. `*.jpg` matches nfo_final's frames and nfo_processed's `*_or.jpg` while excluding its
    `*_sammask.png`."""
    d = seq_dir(seq)
    jpgs = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    idx = indices if indices is not None else range(len(jpgs) if up_to is None else up_to)
    return np.stack([cv2.imread(os.path.join(d, jpgs[i]), 0) for i in idx], axis=0)


def load_boxes(seq: str):
    """Ground-truth boxes INDEXED BY FRAME, None where absent or unlabelled. nfo_final carries a
    second groundtruth* file holding the normalised boxes; nfo_processed has only
    groundtruth.txt, already normalised."""
    d = seq_dir(seq)
    norm = next((f for f in sorted(os.listdir(d))
                 if f != 'groundtruth.txt' and f.startswith('groundtruth')), 'groundtruth.txt')
    return parse_normalized_bbs(os.path.join(d, norm))


def eval_sequence(seq, use_shape_scoring, scale_relative=False):
    boxes = load_boxes(seq)
    n_jpg = len([f for f in os.listdir(seq_dir(seq)) if f.endswith('.jpg')])
    n = min(n_jpg, len(boxes))

    valid_centers = [c for c in range(SPAN, n - SPAN) if boxes[c] is not None]
    frames_all = load_frames(seq, up_to=n)
    h, w = frames_all.shape[1], frames_all.shape[2]

    expected_height = EXPECTED_HEIGHT if use_shape_scoring else None
    person_height = None
    if scale_relative:
        person_height = estimate_person_height(frames_all, bg_frames=BG_FRAMES)
        ref = (f"hand-measured ground-truth value: {EXPECTED_HEIGHT:.0f}px" if CALIBRATED_LAYOUT
               else f"EXPECTED_HEIGHT={EXPECTED_HEIGHT:.0f}px is nfo_final's and does not apply "
                    f"at {h}x{w}")
        print(f"  {seq}: measured person height = {person_height:.0f}px ({ref})")
    results = track_windows_in_sequence(frames_all, valid_centers, span=SPAN, nth_frame=NTH_FRAME,
                                        max_dist=MAX_DIST, merge_radius=MERGE_RADIUS,
                                        expected_height=expected_height, bg_frames=BG_FRAMES,
                                        person_height=person_height)

    residuals, n_no_track = [], 0
    for center in valid_centers:
        result = results[center]
        if result is None:
            n_no_track += 1
            continue
        gt = boxes[center]
        gt_cx, gt_cy = (gt[0] + gt[2] / 2) * w, (gt[1] + gt[3] / 2) * h
        dist_norm = np.hypot((result['x'] - gt_cx) / w, (result['y'] - gt_cy) / h)
        residuals.append(dist_norm)

    return residuals, n_no_track, len(valid_centers)


def run(use_shape_scoring, scale_relative=False):
    label = "WITH shape-aware scoring + sequence warm-start" if use_shape_scoring else \
        "WITHOUT shape-aware scoring, WITH sequence warm-start"
    if scale_relative:
        label += ", SCALE-RELATIVE constants (measured, no NFO-specific pixel values)"
    print(f"=== {label} ===")
    all_residuals, total_no_track, total_valid = [], 0, 0
    for seq in SEQS:
        residuals, n_no_track, n_valid = eval_sequence(seq, use_shape_scoring, scale_relative)
        all_residuals.extend(residuals)
        total_no_track += n_no_track
        total_valid += n_valid
        msg = f"mean_resid={np.mean(residuals):.4f}" if residuals else "no tracks found"
        print(f"{seq}: valid_centers={n_valid} no_track={n_no_track} tracked={len(residuals)} {msg}")

    all_residuals = np.array(all_residuals)
    print()
    print(f"TOTAL valid_centers={total_valid} no_track={total_no_track} "
          f"({100 * total_no_track / total_valid:.1f}%) tracked={len(all_residuals)}")
    if len(all_residuals):
        print(f"residual (normalized [0,1] units, diagonal distance): "
              f"mean={all_residuals.mean():.4f} median={np.median(all_residuals):.4f} "
              f"p90={np.percentile(all_residuals, 90):.4f} p99={np.percentile(all_residuals, 99):.4f} "
              f"max={all_residuals.max():.4f}")
        # BOTH rates, because hit alone is gameable: it scores only frames where a position was
        # reported, so anything that declines to report on hard frames inflates it. localized
        # divides by every valid centre instead, so a no_track counts as a miss.
        hit = float((all_residuals <= 0.1).mean())
        print(f"hit@0.1 = {100 * hit:.1f}% of the {len(all_residuals)} tracked centres; "
              f"localized = {100 * hit * len(all_residuals) / total_valid:.1f}% of all "
              f"{total_valid} valid centres (the difference is the {total_no_track} no_track)")
        print("for reference: the eval pipeline's max_dist_error threshold is 0.1 (10% of frame)")
    print()


CONFIGS = {
    'noshape': dict(use_shape_scoring=False),
    'fixed': dict(use_shape_scoring=True),
    # same tracker, same data, but every pixel constant derived from a person height measured
    # off the footage instead of hand-measured from NFO's ground truth. If this matches
    # 'fixed', NFO no longer needs any dataset-specific constant at all.
    'relative': dict(use_shape_scoring=True, scale_relative=True),
}


def main():
    """No arguments: run all three configs in sequence (~11 min) on the calibrated layout, or
    only 'relative' on a layout the constants were not calibrated for. Named configs run only
    those, so a scheduler can put one per job - see tracking/eval/eval_nfo.sbatch."""
    default = list(CONFIGS) if CALIBRATED_LAYOUT else ['relative']
    names = [a for a in sys.argv[1:] if not a.startswith('-')] or default
    for name in names:
        run(**CONFIGS[name])


if __name__ == '__main__':
    main()
