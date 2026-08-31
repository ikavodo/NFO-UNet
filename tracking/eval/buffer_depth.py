"""How deep must the integration buffer be before the person is actually recoverable?

    python -m tracking.eval.buffer_depth --path data/ido_walk.mkv --person-height 421
    python -m tracking.eval.buffer_depth --path data/nfo_final/nfo_final/seq1 --person-height 195 --scale 1.0

Two effects fight each other as the buffer grows, and both are measured here.

  (+) OCCLUDER CLEARING. In aligned coordinates the person is static and a static occluder
      translates at -vx, so an aligned pixel is covered for about w/vx of the strided samples
      and its occluded fraction is d ~ min(1, w / (vx*(T-1))) for occluder width w along the
      motion axis. Two separate thresholds follow, and they are NOT the same test:
        - MEDIAN fusion needs a majority of clean looks: d < 1/2, i.e. vx*(T-1) > 2w.
        - MASK-AWARE / support-weighted fusion needs only that some look was clean, so it
          improves with the expected number of clean looks and has no majority requirement.
      A scene whose occluder duty cycle already exceeds 1/2 is unreachable by the median at
      ANY depth, which is why both are reported.

  (-) POSE DRIFT. The alignment is a rigid horizontal translation but limbs articulate, so
      the person's own silhouette stops overlapping itself as |dt| grows.

  Plus a hard ceiling: T <= frame_width / vx, beyond which the person has crossed the frame.

The tracker is held at its calibrated SEQ_SIZE=7 and only supplies the motion model (vx,
anchor); the integration depth is varied independently. That decoupling is the point -
integration depth is not a tracking parameter, and buying more of it costs lookahead latency
but no tracking accuracy.

MEASURED on data/ido_walk.mkv at scale 0.5 (person 421px, |vx| median 14.6px per strided
frame), 44 paired centres, every window fully inside one person-present walk so the same
centre set is used at every depth:

    T   real span   sweep px   mean nu   nu>0.5 (median)   nu>0 (>=1 look)   sharpness
    7      12          88       0.261         0.214              0.569          1.78
   13      24         176       0.230         0.162              0.679          1.63
   21      40         293       0.188         0.086              0.726          1.45
   31      60         439       0.161         0.025              0.775          1.28
   41      80         586       0.144         0.009              0.801          1.12
   61     120         878       0.128         0.007              0.821          0.92

THE TWO CRITERIA MOVE IN OPPOSITE DIRECTIONS, and that is the whole finding:

  - The MEDIAN's reachable fraction FALLS monotonically, 0.214 -> 0.007. Depth is actively
    harmful. The reason is upstream of any tuning: the measured not-detected duty cycle inside
    the person's box is 0.59, already past the median's 1/2 breakdown point, so a majority of
    clean looks is unreachable at any depth and extra frames only dilute. Equivalently, mean
    nu = 0.261 at T=7 is 1.83 clean looks per pixel out of 7 where the median needs 4.
  - The AT-LEAST-ONE-CLEAN-LOOK fraction RISES monotonically and saturates, 0.569 -> 0.821,
    flattening around T=31-41. So depth does buy coverage - but only a fusion rule that can
    use a single good look can spend it.

Conclusion: buffer depth is the wrong knob until the fusion rule changes. Under median fusion
keep T=7. Under a support-map / best-look rule the payoff saturates near T=31, i.e. 60 frames
= 2.5s of lookahead, which is a latency budget decision (fine for review, hopeless for live),
not an algorithmic one.

The closed form is consistent with this but does NOT predict it alone: clearing the p50/p75/p90
occluder widths (42/81/122px) needs T >= 6.7/12.1/17.7, all of which the median never reaches
because of the duty cycle. Transit ceiling T <= 66 (frame width / vx).

Silhouette self-overlap under the rigid alignment, IoU against the centre frame:

    |dt| strided    0     1     2     3     4     5     6
    IoU           1.00  0.50  0.33  0.24  0.18  0.15  0.11

IoU halves at |dt| = 2 strided frames (4 real frames), i.e. far faster than the 12-18 needed
to clear the occluder. Something decorrelates well before clearing can pay.

TWO CONFOUNDS, both unresolved, and they block the MECHANISM but not the DECISION:
  - mask==0 conflates true occlusion with MOG2 detection failure (a white shirt against a
    white wall). Detection failure travels WITH the person, so in aligned coordinates it never
    sweeps away and no depth removes it - which could account for the 0.59 duty cycle on its
    own.
  - the IoU curve conflates pose drift with frame-to-frame mask fragmentation. A 421px person
    2 real frames apart should overlap far more than 0.50, so most of that drop is probably
    segmentation noise, not articulation.
Separating them needs footage where the occluded pixels are KNOWN: NFO (foliage over a
textured background, with ground truth) or a synthetic occluder laid over unoccluded frames.
Until then, attribute the numbers to "decorrelation", not to pose drift specifically. The
decision above holds either way, since it rests on the measured curves.

Scope: one sequence, one camera geometry, person nearly filling the frame. --path accepts a
directory of jpgs, so run it on NFO before generalising anything here.
"""
import argparse
import os

import cv2
import numpy as np

from tracking.core.blob_tracker import detect_blobs, _Track
from tracking.core.integrate_image import crop_at, fuse
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask
from tracking.core.track_sequence import scale_relative_params
from tracking.core.track_window import _result_from_detections
from tracking.stream.stream import NTH_FRAME, SEQ_SIZE, SPAN, frames_from_video

DEPTHS = (7, 13, 21, 31, 41, 51, 61)


def load_frames(path: str, scale: float = 1.0) -> np.ndarray:
    """A video file or a directory of greyscale jpgs (the NFO layout)."""
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path) if f.endswith('.jpg'))
        assert names, f"no .jpg files in {path}"
        out = []
        for n in names:
            g = cv2.imread(os.path.join(path, n), 0)
            out.append(g if scale == 1.0 else
                       cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        return np.stack(out)
    return np.stack(list(frames_from_video(path, scale)))


def track_centre(masks, kw, centre):
    """The calibrated SEQ_SIZE=7 window result at one centre frame, or None."""
    idx = list(range(centre - SPAN, centre + SPAN + 1, NTH_FRAME))
    dets = detect_blobs(masks[idx], min_area=kw['min_area'])
    return _result_from_detections(dets, 3, kw['expected_height'], 0.5, kw['max_dist'], 6,
                                   kw['merge_radius'], return_box=True)


def occluder_widths(frames, masks, kw):
    """Horizontal runs of not-detected pixels inside the person's own merged box, i.e. the
    occluder width w along the motion axis, plus the box's foreground duty cycle."""
    runs, fills = [], []
    for c in range(SPAN, len(frames) - SPAN):
        r = track_centre(masks, kw, c)
        if r is None or r['box'] is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in r['box'])
        sub = masks[c][max(0, y1):y2, max(0, x1):x2] > 0
        if sub.size == 0:
            continue
        fills.append(sub.mean())
        for row in sub:
            d = np.diff(np.concatenate(([0], (~row).view(np.int8), [0])))
            runs.extend(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return np.array(runs), np.array(fills)


def depth_sweep(frames, masks, kw, centres, crop, depths=DEPTHS):
    rows, panels, vxs = {}, {}, []
    for c in centres:
        r = track_centre(masks, kw, c)
        if r is None or r['winner'] is None:
            continue
        w = r['winner']
        mid = SEQ_SIZE // 2
        ax = w['history'][mid][0] if mid in w['history'] else r['x']
        ay = float(np.mean([w['history'][f][1] for f in w['frames']]))
        vx = w['vx']
        vxs.append(abs(vx))
        ones = np.ones(masks[0].shape, np.uint8)
        for T in depths:
            half = (T // 2) * NTH_FRAME
            if c - half < 0 or c + half >= len(frames):
                continue
            ts = np.arange(T) - T // 2
            fg = np.stack([crop_at((masks[c + t * NTH_FRAME] > 0).astype(np.uint8),
                                   ax + vx * t, ay, crop) for t in ts]).sum(0)
            # a crop that has left the frame is zero-padded, and that padding is not evidence
            # of occlusion - normalise by how many samples actually had image data there
            cover = np.stack([crop_at(ones, ax + vx * t, ay, crop) for t in ts]).sum(0)
            nu = fg / np.maximum(cover, 1).astype(float)
            img = np.stack([crop_at(frames[c + t * NTH_FRAME], ax + vx * t, ay, crop)
                            for t in ts])
            b, hh = crop // 2, int(0.5 * kw['expected_height'] / 0.95)
            region = np.zeros_like(nu, bool)
            region[max(0, b - hh):b + hh, max(0, b - hh // 3):b + hh // 3] = True
            region &= cover >= T / 2.0
            if region.sum() < 100:
                continue
            med = fuse(img, 'median')
            rows.setdefault(T, []).append(
                (float(nu[region].mean()), float((nu[region] > 0.5).mean()),
                 float((nu[region] > 0).mean()),
                 float(np.abs(np.gradient(med.astype(float))[1]).mean())))
            if c == centres[len(centres) // 2]:
                panels[T] = med
    return rows, panels, float(np.median(vxs)) if vxs else float('nan')


def mask_decorrelation(frames, masks, kw, centres, crop, max_dt=30):
    """How fast does the person's own silhouette stop overlapping itself under the rigid
    horizontal alignment? IoU between the aligned mask at offset dt and the aligned mask at
    the centre frame, averaged over centres. This is the ceiling on integration depth that
    has nothing to do with the occluder: past the dt where IoU collapses, extra frames add
    a differently-posed body, not more looks at the same one."""
    ones = np.ones(masks[0].shape, np.uint8)
    acc = {}
    for c in centres:
        r = track_centre(masks, kw, c)
        if r is None or r['winner'] is None:
            continue
        w = r['winner']
        mid = SEQ_SIZE // 2
        ax = w['history'][mid][0] if mid in w['history'] else r['x']
        ay = float(np.mean([w['history'][f][1] for f in w['frames']]))
        vx = w['vx']

        def al(dt):
            i = c + dt * NTH_FRAME
            if not 0 <= i < len(masks):
                return None, None
            return (crop_at((masks[i] > 0).astype(np.uint8), ax + vx * dt, ay, crop),
                    crop_at(ones, ax + vx * dt, ay, crop))

        ref, ref_cov = al(0)
        if ref is None or ref.sum() == 0:
            continue
        for dt in range(0, max_dt + 1):
            for sgn in ((1,) if dt == 0 else (1, -1)):
                m, cov = al(sgn * dt)
                if m is None:
                    continue
                valid = (cov > 0) & (ref_cov > 0)
                inter = float((m & ref & valid).sum())
                union = float(((m | ref) & valid).sum())
                if union > 0:
                    acc.setdefault(dt, []).append(inter / union)
    return {dt: float(np.mean(v)) for dt, v in sorted(acc.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--path', default='data/ido_walk.mkv')
    p.add_argument('--person-height', type=float, required=True)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--crop-factor', type=float, default=1.2)
    p.add_argument('--centres', type=int, nargs='*', default=None)
    p.add_argument('--out-dir', default='images/stream')
    a = p.parse_args()

    kw, kalman = scale_relative_params(a.person_height)
    _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = kalman
    frames = load_frames(a.path, a.scale)
    masks = filter_by_shape(refine_mask(foreground_mask(frames, bg_frames=30),
                                        kw['close_kernel_size'], kw['open_kernel_size']),
                            min_area=kw['min_area'], min_solidity=0.1)
    print(f'{len(frames)} frames {frames.shape[2]}x{frames.shape[1]}')

    runs, fills = occluder_widths(frames, masks, kw)
    print(f"\nperson box: foreground fill median {np.median(fills):.2f} => not-detected duty "
          f"cycle {1 - np.median(fills):.2f}  (median fusion needs < 0.50 at ANY depth)")
    print(f"occluder width w along the motion axis, from {len(runs)} horizontal runs: "
          f"median {np.median(runs):.0f}px  p75 {np.percentile(runs, 75):.0f}px  "
          f"p90 {np.percentile(runs, 90):.0f}px")

    centres = a.centres or list(np.linspace(SPAN + 60, len(frames) - SPAN - 60, 4).astype(int))
    rows, panels, vx = depth_sweep(frames, masks, kw, centres,
                                   int(round(a.crop_factor * a.person_height)))
    print(f"\n|vx| median {vx:.1f}px per strided frame; transit ceiling "
          f"T <= {frames.shape[2] / vx:.0f}; closed form vx*(T-1) > 2w requires T >= "
          + ", ".join(f"{1 + 2 * np.percentile(runs, q) / vx:.1f} (w=p{q:.0f})"
                      for q in (50, 75, 90)))
    print(f"\n{'T':>4} {'real span':>10} {'sweep px':>9} {'mean nu':>8} "
          f"{'nu>0.5 (median)':>16} {'nu>0 (>=1 look)':>16} {'sharpness':>10} {'n':>4}")
    for T in sorted(rows):
        m = np.array(rows[T])
        print(f"{T:>4} {(T - 1) * NTH_FRAME:>10} {vx * (T - 1):>9.0f} {m[:, 0].mean():>8.3f} "
              f"{m[:, 1].mean():>16.3f} {m[:, 2].mean():>16.3f} {m[:, 3].mean():>10.2f} "
              f"{len(m):>4}")
    best = max(rows, key=lambda T: np.array(rows[T])[:, 0].mean())
    print(f"\nmean-nu optimum at T={best} ({(best - 1) * NTH_FRAME}-frame real span, "
          f"{(best - 1) // 2 * NTH_FRAME} frames of lookahead). Note frac nu>0.5 is the "
          f"MEDIAN's reachable fraction and may fall while mean nu rises - see the module "
          f"docstring on why those are different tests.")

    iou = mask_decorrelation(frames, masks, kw, centres,
                             int(round(a.crop_factor * a.person_height)))
    print(f"\nsilhouette self-overlap under rigid alignment (IoU vs the centre frame):")
    print("  |dt| strided " + " ".join(f"{d:>5}" for d in sorted(iou) if d <= 12))
    print("  IoU          " + " ".join(f"{iou[d]:>5.2f}" for d in sorted(iou) if d <= 12))
    half = next((d for d in sorted(iou) if iou[d] < 0.5 * iou[0]), None)
    if half:
        print(f"  IoU halves at |dt| = {half} strided frames = {half * NTH_FRAME} real frames "
              f"=> a window wider than T = {2 * half + 1} is adding differently-posed bodies. "
              f"Occluder clearing needs T >= {1 + 2 * np.percentile(runs, 75) / vx:.1f} (w=p75), "
              f"so the two constraints "
              + ("CONFLICT" if 2 * half + 1 < 1 + 2 * np.percentile(runs, 75) / vx
                 else "are compatible") + ".")

    if panels:
        strip = np.hstack([cv2.resize(panels[T], (240, 240)) for T in sorted(panels)])
        bar = np.zeros((28, strip.shape[1]), np.uint8)
        for i, T in enumerate(sorted(panels)):
            cv2.putText(bar, f'T={T}', (8 + 240 * i, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        out = f'{a.out_dir}/08_buffer_depth_fused.png'
        cv2.imwrite(out, np.vstack([bar, strip]))
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
