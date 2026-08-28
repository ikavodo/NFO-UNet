"""Measure how fragmented a person's foreground mask really is on NFO, and tune the synthetic
occluder generator to match it.

Why: docs/scale_generalization_plan.md, finding F6 - the synthetic occluders demonstrably do
not reproduce NFO's failure mode (on NFO the shape term is worth a 2.7x error reduction and
its job is picking the right *fragment* of a heavily-broken person; on synthetic KTH the same
term is worth 2-3pp and two distractor designs measured nothing). Training anything on
occluders that fragment differently from the target data optimizes the wrong objective, so
this measurement is the blocking prerequisite for the learned components rather than a
footnote.

Four statistics, all dimensionless, measured inside the person's own ground-truth box:
  blobs_per_person  how many separate foreground components the person is broken into
  largest_frac      tallest component's height / box height  (1.0 = person intact)
  fill_frac         foreground pixels inside the box / box area
  gap_frac          mean nearest-neighbour distance between components / box height

Measured strictly *inside* the ground-truth box, so no blob-identity decision is needed and
real foliage outside the person cannot contaminate the count - the same treatment applies to
both datasets, which is what makes them comparable.

The segmentation front-end (min_area, morphology kernels) is derived from each dataset's own
ground-truth person height via scale_relative_params, so neither dataset gets a front-end
tuned to it and the comparison is not confounded by pixel scale. Person height comes from
ground truth here, NOT from estimate_person_height - F4 showed that estimator is unreliable
on real NFO, and it has no business inside a calibration loop.

    python -m tracking.eval.occluder_calibration
"""
import os
import sys

import cv2
import numpy as np

from tracking.core.preprocess import foreground_mask, refine_mask, filter_by_shape
from tracking.core.track_sequence import scale_relative_params
from tracking.eval import kill_test_scale as kt
from tracking.eval.eval_nfo import IN_DIR, SEQS as NFO_SEQS, parse_normalized_bbs, BG_FRAMES

DENSITIES = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35]
THICKNESSES = [1, 5]
STAT_NAMES = ['blobs_per_person', 'largest_frac', 'fill_frac', 'gap_frac']
# fill_frac is deliberately EXCLUDED from the mismatch score. Adding occlusion can only lower
# it, and synthetic KTH's *unoccluded* fill (0.230) is already far below NFO's occluded fill
# (0.475) - MOG2 recovers a smaller share of the person on KTH than on NFO to begin with. No
# occluder setting can close that gap; it is a property of the source footage, and scoring it
# would just drive the search to density 0. The three shape statistics are what the occluder
# actually controls.
SHAPE_STATS = ['blobs_per_person', 'largest_frac', 'gap_frac']


def fragmentation_stats(frames, boxes, person_h, bg_frames):
    """frames [T,H,W]; boxes {frame: (x,y,w,h) normalized}; person_h in px.
    -> {stat: mean over frames}."""
    pre, _ = scale_relative_params(person_h)
    masks = filter_by_shape(refine_mask(foreground_mask(frames, bg_frames=bg_frames),
                                        close_kernel_size=pre['close_kernel_size'],
                                        open_kernel_size=pre['open_kernel_size']),
                            min_area=pre['min_area'], min_solidity=0.1)
    H, W = frames.shape[1:]
    acc = {k: [] for k in STAT_NAMES}
    for f, b in boxes.items():
        if f >= len(masks):
            continue
        x1, y1 = max(int(b[0] * W), 0), max(int(b[1] * H), 0)
        x2, y2 = min(int((b[0] + b[2]) * W), W), min(int((b[1] + b[3]) * H), H)
        box_h = y2 - y1
        if box_h < 4 or x2 - x1 < 2:
            continue
        crop = (masks[f, y1:y2, x1:x2] > 0).astype(np.uint8)
        n, _, stats, cent = cv2.connectedComponentsWithStats(crop, connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= pre['min_area']]
        acc['fill_frac'].append(crop.mean())
        acc['blobs_per_person'].append(len(keep))
        if not keep:
            acc['largest_frac'].append(0.0)
            continue
        acc['largest_frac'].append(max(stats[i, cv2.CC_STAT_HEIGHT] for i in keep) / box_h)
        if len(keep) >= 2:
            pts = np.array([cent[i] for i in keep])
            d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            np.fill_diagonal(d, np.inf)
            acc['gap_frac'].append(float(d.min(axis=1).mean()) / box_h)
    return {k: float(np.mean(v)) if v else float('nan') for k, v in acc.items()}


def nfo_reference():
    """Pooled fragmentation statistics for real NFO, plus per-sequence values."""
    per_seq, weights = [], []
    for seq in NFO_SEQS:
        seq_in = os.path.join(IN_DIR, seq)
        jpgs = sorted(f for f in os.listdir(seq_in) if f.endswith('.jpg'))
        norm_file = next(f for f in os.listdir(seq_in)
                         if f != 'groundtruth.txt' and f.startswith('groundtruth'))
        raw = parse_normalized_bbs(os.path.join(seq_in, norm_file))
        n = min(len(jpgs), len(raw))
        boxes = {i: raw[i] for i in range(n) if raw[i] is not None}
        frames = np.stack([cv2.imread(os.path.join(seq_in, jpgs[i]), 0) for i in range(n)], axis=0)
        person_h = float(np.mean([b[3] * frames.shape[1] for b in boxes.values()]))
        s = fragmentation_stats(frames, boxes, person_h, BG_FRAMES)
        print(f"  {seq}: person_h={person_h:.0f}px  " +
              '  '.join(f"{k}={s[k]:.3f}" for k in STAT_NAMES))
        per_seq.append(s)
        weights.append(len(boxes))
    pooled = {k: float(np.average([s[k] for s in per_seq], weights=weights)) for k in STAT_NAMES}
    return pooled, per_seq


def synthetic_stats(density, seqs, thickness=1):
    """Same statistics on KTH + the branch occluder at the given density, at 1x scale."""
    kt.OCC_DENSITY = density      # ponytail: module constants are the generator's knobs
    kt.OCC_THICKNESS = thickness
    out, weights = [], []
    for seq_i, name in enumerate(seqs):
        frames_native, boxes = kt.load_sequence(name)
        built = kt.build_bucket(frames_native, boxes, 1.0, seed=seq_i)
        frames, boxes_b = built['frames'], built['boxes']
        person_h = float(np.mean([b[3] * frames.shape[1] for b in boxes_b.values()]))
        first = next(c for c in sorted(boxes_b) if c >= kt.SPAN)
        s = fragmentation_stats(frames, boxes_b, person_h, int(max(5, min(30, first))))
        out.append(s)
        weights.append(len(boxes_b))
    return {k: float(np.average([s[k] for s in out], weights=weights)) for k in STAT_NAMES}


def mismatch(a, b):
    """Mean relative difference over the four statistics - a legible stand-in for a real
    distribution distance, which would be overkill for picking one density."""
    terms = []
    for k in SHAPE_STATS:
        scale = max(abs(a[k]), abs(b[k]), 1e-6)
        terms.append(abs(a[k] - b[k]) / scale)
    return float(np.mean(terms))


def main():
    seqs = kt.SEQS[:int(sys.argv[1])] if len(sys.argv) > 1 else kt.SEQS
    print("real NFO (the target):")
    nfo, _ = nfo_reference()
    print("  POOLED: " + '  '.join(f"{k}={nfo[k]:.3f}" for k in STAT_NAMES))

    print(f"\nsynthetic KTH + branch occluder, {len(seqs)} sequences, 1x scale")
    print("(mismatch scored on the three shape statistics only - see SHAPE_STATS comment)")
    print(f"{'thick':>6}{'density':>8} " + ' '.join(f"{k:>17}" for k in STAT_NAMES)
          + f"{'mismatch':>10}")
    results = {}
    for th in THICKNESSES:
        for d in DENSITIES:
            s = synthetic_stats(d, seqs, thickness=th)
            results[(th, d)] = s
            print(f"{th:>6}{d:>8.2f} " + ' '.join(f"{s[k]:>17.3f}" for k in STAT_NAMES)
                  + f"{mismatch(s, nfo):>10.3f}")

    best = min(results, key=lambda k: mismatch(results[k], nfo))
    print(f"\nclosest to NFO: thickness={best[0]} density={best[1]:.2f} "
          f"(mismatch {mismatch(results[best], nfo):.3f})")
    print("used throughout the kill test so far: density 0.35, thickness 1")
    for k in STAT_NAMES:
        tag = '' if k in SHAPE_STATS else '   (not scored)'
        print(f"  {k}: synthetic {results[best][k]:.3f} vs NFO {nfo[k]:.3f}{tag}")


if __name__ == '__main__':
    main()
