"""Kill test for the scale-robustness claim in docs/deepsort_blob_scoring_compatibility.md.

Runs the EXISTING blob tracker (`track_blobs`/`score_and_fit`, logic untouched - only the
already-hardcoded Kalman variances were exposed as `_Track.P_VAR/Q_VAR/R_VAR`, same
defaults) on synthetic multi-scale KTH + `generate_occlusion_branch` sequences, two ways:

  (a) per-scale-correct constants: MAX_DIST/EXPECTED_HEIGHT/MERGE_RADIUS measured from that
      bucket's own ground truth, Kalman P/Q/R scaled by scale^2 (they are pixel-variance
      constants).
  (b) one fixed bucket's constants (REF_SCALE) applied to every bucket - today's actual
      deployment pattern (see tracking/eval/eval_nfo.py: three absolute-pixel constants
      measured once, on one dataset at one resolution).

If (b) doesn't degrade meaningfully against (a), the learned-scorer motivation dies here.

Scale is simulated by resizing the whole frame (0.5x/1x/2x of kth_processed's 224x224), so
person pixel height moves ~60/119/238px while the person's *fraction* of the frame stays
constant. Residuals are reported in frame-normalized units (as eval_nfo.py does), which is
what makes the buckets comparable at all. Caveat: a real camera-distance change would keep
the frame size fixed and shrink the person inside it; that needs person/background
compositing, and is not what this test does.

Occlusion: one STATIC branch mask per sequence, restricted to the union of the sequence's
real per-frame GT boxes (utils/bb_utils-style normalized boxes from groundtruth.txt), with
occluded pixels filled from the sequence's own per-pixel temporal median (i.e. the
background showing through). Static-and-background-valued is deliberate and load-bearing:
this pipeline detects people by MOG2 background subtraction, so an occluder regenerated
per frame inside the *moving* GT box would be new content every frame - MOG2 would report
it as extra foreground instead of hiding the person, and no occlusion would occur. A
static occluder gets absorbed into the background model, so the person's mask actually
fragments when they walk behind it - the intended failure mode.

Usage:
    python -m tracking.eval.kill_test_scale [--frozen-preprocess]

--frozen-preprocess keeps min_area/morph kernel sizes at their defaults in every bucket
instead of scaling them with the bucket (default: scaled, in BOTH arms, so the (a)/(b)
contrast isolates exactly the constants the doc names).
"""
import os
import sys

import cv2
import numpy as np

from tracking.core import blob_tracker as bt
from tracking.core.track_sequence import track_windows_in_sequence
from utils.occlusion_utils import generate_occlusion_branch

ROOT = 'data/kth_processed'
SEQS = ['person01_walking_d1_uncomp_gt', 'person02_walking_d1_uncomp_gt', 'person03_walking_d1_uncomp_gt',
        'person01_jogging_d1_uncomp_gt', 'person02_jogging_d1_uncomp_gt', 'person03_jogging_d1_uncomp_gt']

SEQ_SIZE = 7      # matches config/train_config.py
NTH_FRAME = 2
SPAN = (SEQ_SIZE // 2) * NTH_FRAME

SCALES = [0.5, 1.0, 2.0]
REF_SCALE = 1.0   # the bucket arm (b) calibrates on and then deploys everywhere

OCC_DENSITY = 0.35   # fraction of the GT-union box covered by branches. NOT calibrated
                     # against real NFO occluder coverage (documented gap in the doc).
HIT_THRESHOLD = 0.1  # eval pipeline's max_dist_error, in frame-normalized units

# defaults at scale 1.0, i.e. kth_processed's native 224x224
BASE_MIN_AREA = 50.0
BASE_CLOSE_K = 6
BASE_OPEN_K = 4
BASE_P_VAR, BASE_Q_VAR, BASE_R_VAR = 50.0, 2.0, 9.0


def load_sequence(name):
    """-> ([T, H, W] uint8 frames, {frame_idx: (x, y, w, h) normalized GT box})."""
    d = os.path.join(ROOT, name)
    jpgs = sorted(f for f in os.listdir(d) if f.endswith('_or.jpg'))
    frames = np.stack([cv2.imread(os.path.join(d, f), 0) for f in jpgs], axis=0)
    boxes = {}
    for line in open(os.path.join(d, 'groundtruth.txt')):
        v = [float(x) for x in line.strip().split(',')]
        if v[1] >= 0:  # -1 marks "person not present in this frame"
            boxes[int(v[0])] = tuple(v[1:5])
    boxes = {f: b for f, b in boxes.items() if f < len(frames)}
    return frames, boxes


def center_px(box, H, W):
    return (box[0] + box[2] / 2) * W, (box[1] + box[3] / 2) * H


def measure_constants(boxes, H, W):
    """The equivalent of eval_nfo.py's three constants, measured at this bucket's scale."""
    exp_h = float(np.mean([b[3] * H for b in boxes.values()]))
    disp = []
    for f in sorted(boxes):
        g = f + NTH_FRAME
        if g in boxes:
            x1, y1 = center_px(boxes[f], H, W)
            x2, y2 = center_px(boxes[g], H, W)
            disp.append(np.hypot(x2 - x1, y2 - y1))
    return dict(max_dist=float(np.percentile(disp, 99)),
                expected_height=exp_h,
                merge_radius=exp_h / 2)


def build_bucket(frames_native, boxes, scale, seed):
    """Resize to the bucket, then composite one static, GT-union-restricted branch occluder.
    -> ([T, H, W] uint8 occluded frames, mean per-frame coverage of the real GT box)."""
    Hn, Wn = frames_native.shape[1:]
    H, W = int(round(Hn * scale)), int(round(Wn * scale))
    frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames_native], axis=0)

    x1 = int(np.floor(min(b[0] for b in boxes.values()) * W))
    y1 = int(np.floor(min(b[1] for b in boxes.values()) * H))
    x2 = int(np.ceil(max(b[0] + b[2] for b in boxes.values()) * W))
    y2 = int(np.ceil(max(b[1] + b[3] for b in boxes.values()) * H))
    union = (max(x1, 0), max(y1, 0), min(x2, W), min(y2, H))
    occ = generate_occlusion_branch((H, W), density=OCC_DENSITY, bbox=union, seed=seed)

    background = np.median(frames, axis=0).astype(np.uint8)  # person passes through, so median ~ background
    frames[:, occ] = background[occ]

    coverage = []
    for f, b in boxes.items():
        bx1, by1 = int(b[0] * W), int(b[1] * H)
        bx2, by2 = int((b[0] + b[2]) * W), int((b[1] + b[3]) * H)
        sub = occ[max(by1, 0):by2, max(bx1, 0):bx2]
        if sub.size:
            coverage.append(sub.mean())
    return frames, float(np.mean(coverage))


def run_arm(frames, boxes, const, kf, preprocess):
    """One (bucket, constants) run. -> (residuals list, n_no_track, n_windows)."""
    T, H, W = frames.shape
    centers = [c for c in sorted(boxes) if SPAN <= c < T - SPAN]
    bt._Track.P_VAR, bt._Track.Q_VAR, bt._Track.R_VAR = kf
    bg_frames = int(max(5, min(30, centers[0])))  # must not exceed frames actually elapsed
    results = track_windows_in_sequence(
        frames, centers, span=SPAN, nth_frame=NTH_FRAME, bg_frames=bg_frames,
        min_area=preprocess['min_area'], close_kernel_size=preprocess['close_k'],
        open_kernel_size=preprocess['open_k'],
        max_dist=const['max_dist'], merge_radius=const['merge_radius'],
        expected_height=const['expected_height'])
    bt._Track.P_VAR, bt._Track.Q_VAR, bt._Track.R_VAR = BASE_P_VAR, BASE_Q_VAR, BASE_R_VAR

    residuals, n_no_track = [], 0
    for c in centers:
        r = results[c]
        if r is None:
            n_no_track += 1
            continue
        gx, gy = center_px(boxes[c], H, W)
        residuals.append(float(np.hypot((r['x'] - gx) / W, (r['y'] - gy) / H)))
    return residuals, n_no_track, len(centers)


def summarize(residuals, n_no_track, n_windows):
    res = np.array(residuals)
    hit = float((res < HIT_THRESHOLD).sum()) / n_windows if n_windows else 0.0  # no_track counts as a miss
    return dict(n=n_windows, no_track=n_no_track / n_windows if n_windows else 0.0,
                mean=res.mean() if len(res) else float('nan'),
                median=float(np.median(res)) if len(res) else float('nan'),
                hit=hit)


def main():
    scale_preprocess = '--frozen-preprocess' not in sys.argv
    print(f"scale buckets {SCALES}, ref bucket {REF_SCALE}, occluder density {OCC_DENSITY}, "
          f"preprocessing {'scaled per bucket' if scale_preprocess else 'frozen at defaults'}")
    print(f"sequences: {len(SEQS)}\n")

    # 'a'/'b' are the two arms the doc asks for; the 'b+X' arms are a leave-one-in ablation
    # (fixed constants everywhere EXCEPT X, which is corrected for the bucket) - needed to
    # tell whether any degradation comes from the association gate (max_dist), from
    # score_and_fit's shape term (expected_height, the thing step 2 would replace), from
    # the merge step, or from the Kalman noise scale.
    ARMS = ['a', 'b', 'b+max_dist', 'b+expected_height', 'b+merge_radius', 'b+kf']
    pooled = {s: {arm: ([], 0, 0) for arm in ARMS} for s in SCALES}
    coverages = []

    for seq_i, name in enumerate(SEQS):
        frames_native, boxes = load_sequence(name)
        Hn, Wn = frames_native.shape[1:]
        ref_const = (measure_constants(boxes, int(round(Hn * REF_SCALE)), int(round(Wn * REF_SCALE))),
                     (BASE_P_VAR * REF_SCALE ** 2, BASE_Q_VAR * REF_SCALE ** 2, BASE_R_VAR * REF_SCALE ** 2))
        line = [f"{name.replace('_uncomp_gt', ''):26s}"]
        for scale in SCALES:
            frames, cov = build_bucket(frames_native, boxes, scale, seed=seq_i)
            coverages.append(cov)
            H, W = frames.shape[1:]
            const_a = measure_constants(boxes, H, W)
            kf_a = (BASE_P_VAR * scale ** 2, BASE_Q_VAR * scale ** 2, BASE_R_VAR * scale ** 2)
            if scale_preprocess:
                pre = dict(min_area=BASE_MIN_AREA * scale ** 2,
                           close_k=max(2, int(round(BASE_CLOSE_K * scale))),
                           open_k=max(1, int(round(BASE_OPEN_K * scale))))
            else:
                pre = dict(min_area=BASE_MIN_AREA, close_k=BASE_CLOSE_K, open_k=BASE_OPEN_K)

            const_ref, kf_ref = ref_const
            arm_cfg = {'a': (const_a, kf_a), 'b': (const_ref, kf_ref), 'b+kf': (const_ref, kf_a)}
            for key in ('max_dist', 'expected_height', 'merge_radius'):
                arm_cfg[f'b+{key}'] = ({**const_ref, key: const_a[key]}, kf_ref)
            for arm in ARMS:
                const, kf = arm_cfg[arm]
                r, nnt, nw = run_arm(frames, boxes, const, kf, pre)
                pr, pnt, pnw = pooled[scale][arm]
                pooled[scale][arm] = (pr + r, pnt + nnt, pnw + nw)
            line.append(f"{scale}x done")
        print('  '.join(line))

    print(f"\nmean occluder coverage of the real per-frame GT box: {np.mean(coverages):.3f}\n")
    print(f"{'bucket':>7} {'person_px':>9} {'arm':>18} {'n':>5} {'no_track':>9} {'resid_mean':>11} "
          f"{'resid_med':>10} {f'hit@{HIT_THRESHOLD}':>10}")
    rows = {}
    for scale in SCALES:
        for arm in ARMS:
            s = summarize(*pooled[scale][arm])
            rows[(scale, arm)] = s
            label = {'a': 'per-scale', 'b': f'fixed{REF_SCALE}x'}.get(arm, arm)
            print(f"{scale:>6}x {224 * 0.53 * scale:>9.0f} {label:>18} {s['n']:>5} "
                  f"{100 * s['no_track']:>8.1f}% {s['mean']:>11.4f} {s['median']:>10.4f} "
                  f"{100 * s['hit']:>9.1f}%")

    print("\ndegradation of (b) fixed-constants vs (a) per-scale-correct:")
    for scale in SCALES:
        a, b = rows[(scale, 'a')], rows[(scale, 'b')]
        print(f"  {scale}x: hit@{HIT_THRESHOLD} {100 * a['hit']:.1f}% -> {100 * b['hit']:.1f}% "
              f"({100 * (b['hit'] - a['hit']):+.1f}pp) | resid_med {a['median']:.4f} -> {b['median']:.4f} "
              f"| no_track {100 * a['no_track']:.1f}% -> {100 * b['no_track']:.1f}%")

    print("\nablation - fraction of the (a)-(b) hit-rate gap recovered by correcting ONE constant:")
    for scale in SCALES:
        if scale == REF_SCALE:
            continue
        gap = rows[(scale, 'a')]['hit'] - rows[(scale, 'b')]['hit']
        parts = []
        for arm in ARMS[2:]:
            recovered = (rows[(scale, arm)]['hit'] - rows[(scale, 'b')]['hit'])
            frac = recovered / gap if abs(gap) > 1e-9 else float('nan')
            parts.append(f"{arm[2:]}={100 * rows[(scale, arm)]['hit']:.1f}% ({frac:+.0%} of gap)")
        print(f"  {scale}x (gap {100 * gap:+.1f}pp): " + ', '.join(parts))

    # sanity check: at the reference bucket the two arms ARE the same constants, so any
    # difference there would mean the harness itself is non-deterministic
    a, b = rows[(REF_SCALE, 'a')], rows[(REF_SCALE, 'b')]
    assert abs(a['hit'] - b['hit']) < 1e-12 and abs(a['mean'] - b['mean']) < 1e-12, \
        f"ref bucket arms diverged - harness bug: {a} vs {b}"
    print(f"\nsanity: ref bucket {REF_SCALE}x identical across arms (as it must be)")


if __name__ == '__main__':
    main()
