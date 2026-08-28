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

Occlusion: one branch mask per sequence, restricted to the union of the sequence's real
per-frame GT boxes (normalized boxes from groundtruth.txt), appearance = OCC_DARKEN * that
pixel's temporal-median background, so the occluder is dark foliage-like but keeps the
background's texture. The occluder must be *fixed in the image* (not regenerated inside
the moving GT box each frame): this pipeline finds people by MOG2 background subtraction,
so a per-frame-regenerated occluder would be new content every frame and MOG2 would report
it as extra foreground instead of hiding the person - no occlusion would happen at all. A
frame-fixed occluder is absorbed into the background model, so the person's mask really
does fragment when they walk behind it.

--sway adds the one thing a fully static occluder cannot provide: a *moving non-person
object*. The branch mask is sheared about its base by +-N px (N scales with the bucket),
so branch tips swing while the base stays put. This matters because score_and_fit's shape
term exists specifically to reject a smoothly-moving-but-wrong candidate (its docstring:
a swaying-foliage track outscored the real person on NFO), and no static-occluder test can
put such a candidate in front of it.

Arms, beyond the (a)/(b) pair the kill test needs:
  b+X            leave-one-in ablation: fixed constants except X, corrected for the bucket
  a-no_height    per-scale-correct, score_and_fit's height term switched off entirely
  a-wrong_height per-scale-correct EXCEPT expected_height, left at the reference bucket's
                 value - the clean probe of whether the scoring term is scale-sensitive
                 when association is healthy (b+expected_height cannot tell you, because
                 there a wrong max_dist has already destroyed the tracks)
  scale_rel      every constant (including min_area/kernels/P/Q/R) derived from one
                 measured h_ref, with coefficients fixed across all buckets
  canon          resize the input so the measured h_ref equals H_REF_0, then run with the
                 SAME fixed constants arm (b) uses - scale invariance by canonicalizing
                 the input rather than by rescaling ~8 constants

Usage:
    python -m tracking.eval.kill_test_scale [--sway PX] [--frozen-preprocess]
    python -m tracking.eval.kill_test_scale --alpha-sweep [--sway PX]

--frozen-preprocess keeps min_area/morph kernel sizes at their defaults in every bucket
instead of scaling them with the bucket (default: scaled, in both (a) and (b), so their
contrast isolates exactly the constants the doc names). --alpha-sweep sweeps the two
load-bearing scale-relative coefficients at every bucket, to check whether one pair can
serve all scales.
"""
import os
import sys

import cv2
import numpy as np

from tracking.core import blob_tracker as bt
from tracking.core.preprocess import estimate_person_height
from tracking.core.track_sequence import track_windows_in_sequence, scale_relative_params
from utils.occlusion_utils import generate_occlusion_branch, sway_masks

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
OCC_DARKEN = 0.45    # occluder appearance = 0.45 * that pixel's background, so the occluder
                     # is visibly dark foliage-like but keeps the background's own texture
                     # (a partial fix for the doc's "flat-color occlusion" gap). Static
                     # occluders get absorbed by MOG2 either way; swaying ones must be
                     # visually distinct from the background or they generate no motion.
DIST_PAD = 0.5       # width of the canvas strip added for the separated distractor, as a
                     # fraction of frame width (only used with --distractor)
SWAY_PERIOD = 40.0   # frames per full sway cycle; window span is 13 frames, so a window
                     # sees ~1/3 of a cycle - i.e. sway looks like consistent drift inside
                     # a window, which is what makes it a plausible competing track.
HIT_THRESHOLD = 0.1  # eval pipeline's max_dist_error, in frame-normalized units

# The scale-relative parameterization tested by the 'scale_rel' arm now lives in
# tracking/core/track_sequence.py (ALPHA_* and scale_relative_params) and the person-height
# estimator in tracking/core/preprocess.py, so the experiment and the production path cannot
# drift apart. --alpha-sweep still sweeps the two load-bearing coefficients here.
H_REF_0 = 120.0          # canonical person height for the 'canon' arm: the measured mean at
                         # the 1x bucket, i.e. what canon resizes every bucket to match.

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


def build_bucket(frames_native, boxes, scale, seed, sway_px=0.0, distractor_px=0.0):
    """Resize to the bucket, then composite the branch occluder(s).

    Three modes:
      sway_px=0, distractor_px=0  - one static occluder over the person's path. Occlusion
                                    only, no competing detections (MOG2 absorbs it).
      sway_px>0                   - that same occluder sways. Adds distractors, but also
                                    WEAKENS the occlusion (moving occluder pixels raise the
                                    background model's variance, so the person leaks
                                    through) - the two effects are confounded.
      distractor_px>0             - the separated design: the occluder over the person's
                                    path stays perfectly static, and an INDEPENDENT swaying
                                    branch cluster of full person height is placed in a
                                    canvas strip added beside the frame, where it can never
                                    intersect the person. Occlusion strength and distractor
                                    presence become independent variables.

    Why the added strip: in KTH the person traverses the whole frame width, so the only
    person-free region inside the original frame is the thin band above their path (~30% of
    a person height here) - a distractor that small is filtered out or trivially out-scored,
    which makes the experiment a null by construction. Extending the canvas gives a
    full-height distractor with zero overlap. Residuals stay normalized by the ORIGINAL
    frame size so padded and unpadded runs remain directly comparable, and the person's
    pixel coordinates are untouched by the padding (only the normalization denominator would
    have changed, and that is what norm_hw pins down).

    -> dict(frames, boxes, coverage, dist_h, norm_hw). `boxes` is returned because padding
    changes normalized coordinates; callers must use the returned copy."""
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
    occ_appearance = (background * OCC_DARKEN).astype(np.uint8)

    def gt_box_coverage(mask, f):
        b = boxes[f]
        sub = mask[max(int(b[1] * H), 0):int((b[1] + b[3]) * H), max(int(b[0] * W), 0):int((b[0] + b[2]) * W)]
        return sub.mean() if sub.size else None

    coverage = []
    if sway_px > 0:
        for t, m in enumerate(sway_masks(occ, len(frames), sway_px, SWAY_PERIOD)):
            frames[t][m] = occ_appearance[m]
            if t in boxes:
                coverage.append(gt_box_coverage(m, t))
    else:
        frames[:, occ] = occ_appearance[occ]
        coverage = [gt_box_coverage(occ, f) for f in boxes]

    coverage = [c for c in coverage if c is not None]
    out = dict(frames=frames, boxes=boxes, coverage=float(np.mean(coverage)), dist_h=0,
               norm_hw=(H, W))
    if distractor_px <= 0:
        return out

    # Independent swaying distractor in an added canvas strip: the occluder above is
    # untouched and still perfectly static, so occlusion strength is identical to the
    # static run, and the ONLY thing that changed is that competing moving detections now
    # exist. Denser than the occluder so its blobs survive min_area at every bucket.
    # ponytail: the strip is edge-replicated background (streaky but perfectly static, which
    # is all MOG2 needs). Sample real background texture here if the strip ever has to look
    # right rather than just behave right.
    pad = int(round(DIST_PAD * W))
    frames = np.pad(frames, ((0, 0), (0, 0), (0, pad)), mode='edge')
    W2 = W + pad
    strip_bg = np.median(frames[:, :, W:], axis=0).astype(np.uint8)
    strip_appearance = (strip_bg * OCC_DARKEN).astype(np.uint8)

    dist_h = union[3] - union[1]
    dmask_full = generate_occlusion_branch((H, W2), density=0.5,
                                           bbox=(W, union[1], W2, union[3]), seed=seed + 777)
    for t, m in enumerate(sway_masks(dmask_full, len(frames), distractor_px, SWAY_PERIOD)):
        strip = m[:, W:]
        frames[t, :, W:][strip] = strip_appearance[strip]

    # padding is on the right, so the person's PIXEL position is unchanged; only the
    # normalized coordinates shift, and residuals are still normalized by norm_hw = (H, W)
    out['frames'] = frames
    out['boxes'] = {f: (b[0] * W / W2, b[1], b[2] * W / W2, b[3]) for f, b in boxes.items()}
    out['dist_h'] = dist_h
    return out


def run_arm(frames, boxes, const, kf, preprocess, norm_hw=None):
    """One (bucket, constants) run. -> (residuals list, n_no_track, n_windows).

    norm_hw: (H, W) to normalize residuals by, defaulting to the frames' own size. Pinned
    explicitly so runs whose canvas was widened for a distractor stay comparable to runs
    that were not."""
    T, H, W = frames.shape
    norm_h, norm_w = norm_hw if norm_hw is not None else (H, W)
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
        residuals.append(float(np.hypot((r['x'] - gx) / norm_w, (r['y'] - gy) / norm_h)))
    return residuals, n_no_track, len(centers)


def summarize(residuals, n_no_track, n_windows):
    res = np.array(residuals)
    hit = float((res < HIT_THRESHOLD).sum()) / n_windows if n_windows else 0.0  # no_track counts as a miss
    return dict(n=n_windows, no_track=n_no_track / n_windows if n_windows else 0.0,
                mean=res.mean() if len(res) else float('nan'),
                median=float(np.median(res)) if len(res) else float('nan'),
                hit=hit)


def scale_relative_config(h_ref):
    """All constants from ONE measured number, with coefficients identical at every scale -
    including the segmentation front-end, which must not get oracle scale info either or the
    arm is cheating. Coefficients live in tracking/core/track_sequence.py so the test and the
    production path cannot drift apart."""
    kwargs, kf = scale_relative_params(h_ref)
    const = {k: kwargs[k] for k in ('max_dist', 'expected_height', 'merge_radius')}
    pre = dict(min_area=kwargs['min_area'], close_k=kwargs['close_kernel_size'],
               open_k=kwargs['open_kernel_size'])
    return const, kf, pre


def arg_value(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main():
    scale_preprocess = '--frozen-preprocess' not in sys.argv
    sway_base = arg_value('--sway', 0.0)  # px of branch-tip sway at the 1x bucket
    dist_base = arg_value('--distractor', 0.0)  # px of sway for the separated distractor
    print(f"scale buckets {SCALES}, ref bucket {REF_SCALE}, occluder density {OCC_DENSITY}, "
          f"distractor sway {dist_base}px@1x, "
          f"darken {OCC_DARKEN}, sway {sway_base}px@1x (period {SWAY_PERIOD}), "
          f"preprocessing {'scaled per bucket' if scale_preprocess else 'frozen at defaults'}")
    print(f"sequences: {len(SEQS)}\n")

    # 'a'/'b' are the two arms the doc asks for; the 'b+X' arms are a leave-one-in ablation
    # (fixed constants everywhere EXCEPT X, which is corrected for the bucket) - needed to
    # tell whether any degradation comes from the association gate (max_dist), from
    # score_and_fit's shape term (expected_height, the thing step 2 would replace), from
    # the merge step, or from the Kalman noise scale.
    # 'a-no_height' drops score_and_fit's shape term entirely at per-scale-correct
    # constants: does that term earn its place at all? 'scale_rel' is the proposed fix -
    # every constant derived from one measured scale proxy, same coefficients everywhere.
    # 'a-wrong_height' is the clean probe for step 2's motivation: everything correct for the
    # bucket EXCEPT score_and_fit's expected_height, which is left at the reference bucket's
    # value. The b+X ablation cannot answer this, because there a wrong max_dist has already
    # destroyed the tracks before scoring ever runs.
    # 'canon' is the second proposed fix and needs no new constants at all: resize the input
    # so the measured h_ref matches H_REF_0, then run the pipeline with the SAME fixed
    # constants arm (b) uses. Scale invariance by canonicalizing the input instead of
    # rescaling ~8 constants.
    ABLATION = ['b+max_dist', 'b+expected_height', 'b+merge_radius', 'b+kf']
    ARMS = ['a', 'b'] + ABLATION + ['a-no_height', 'a-wrong_height', 'scale_rel', 'canon']
    pooled = {s: {arm: ([], 0, 0) for arm in ARMS} for s in SCALES}
    coverages, dist_heights, h_refs = [], [], {s: [] for s in SCALES}

    for seq_i, name in enumerate(SEQS):
        frames_native, boxes = load_sequence(name)
        Hn, Wn = frames_native.shape[1:]
        ref_const = (measure_constants(boxes, int(round(Hn * REF_SCALE)), int(round(Wn * REF_SCALE))),
                     (BASE_P_VAR * REF_SCALE ** 2, BASE_Q_VAR * REF_SCALE ** 2, BASE_R_VAR * REF_SCALE ** 2))
        line = [f"{name.replace('_uncomp_gt', ''):26s}"]
        for scale in SCALES:
            built = build_bucket(frames_native, boxes, scale, seed=seq_i,
                                 sway_px=sway_base * scale, distractor_px=dist_base * scale)
            frames, boxes_b, norm_hw = built['frames'], built['boxes'], built['norm_hw']
            coverages.append(built['coverage'])
            dist_heights.append(built['dist_h'])
            H, W = frames.shape[1:]
            first_center = next(c for c in sorted(boxes) if c >= SPAN)
            h_ref = estimate_person_height(frames, bg_frames=int(max(5, min(30, first_center))))
            h_refs[scale].append(h_ref)
            const_a = measure_constants(boxes_b, H, W)
            kf_a = (BASE_P_VAR * scale ** 2, BASE_Q_VAR * scale ** 2, BASE_R_VAR * scale ** 2)
            if scale_preprocess:
                pre = dict(min_area=BASE_MIN_AREA * scale ** 2,
                           close_k=max(2, int(round(BASE_CLOSE_K * scale))),
                           open_k=max(1, int(round(BASE_OPEN_K * scale))))
            else:
                pre = dict(min_area=BASE_MIN_AREA, close_k=BASE_CLOSE_K, open_k=BASE_OPEN_K)

            const_ref, kf_ref = ref_const
            base_pre = dict(min_area=BASE_MIN_AREA, close_k=BASE_CLOSE_K, open_k=BASE_OPEN_K)
            arm_cfg = {'a': (const_a, kf_a, pre, None), 'b': (const_ref, kf_ref, pre, None),
                       'b+kf': (const_ref, kf_a, pre, None),
                       'a-no_height': ({**const_a, 'expected_height': None}, kf_a, pre, None),
                       'a-wrong_height': ({**const_a, 'expected_height': const_ref['expected_height']},
                                          kf_a, pre, None),
                       'scale_rel': scale_relative_config(h_ref) + (None,),
                       'canon': (const_ref, kf_ref, base_pre, H_REF_0 / h_ref)}
            for key in ('max_dist', 'expected_height', 'merge_radius'):
                arm_cfg[f'b+{key}'] = ({**const_ref, key: const_a[key]}, kf_ref, pre, None)
            for arm in ARMS:
                const, kf, arm_pre, resize = arm_cfg[arm]
                arm_frames = frames if resize is None else np.stack(
                    [cv2.resize(f, (int(round(W * resize)), int(round(H * resize)))) for f in frames], axis=0)
                arm_norm = norm_hw if resize is None else tuple(int(round(v * resize)) for v in norm_hw)
                r, nnt, nw = run_arm(arm_frames, boxes_b, const, kf, arm_pre, norm_hw=arm_norm)
                pr, pnt, pnw = pooled[scale][arm]
                pooled[scale][arm] = (pr + r, pnt + nnt, pnw + nw)
            line.append(f"{scale}x done")
        print('  '.join(line))

    print(f"\nmean occluder coverage of the real per-frame GT box: {np.mean(coverages):.3f}")
    if dist_base > 0:
        print(f"separated distractor: swaying branch band above the GT path, mean height "
              f"{np.mean(dist_heights):.0f}px (0 = no room in that sequence), zero overlap with "
              f"the person by construction")
    print("measured scale proxy h_ref (p95 foreground-component height): "
          + ', '.join(f"{s}x={np.mean(v):.1f}px" for s, v in h_refs.items()) + "\n")
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
        for arm in ABLATION:
            recovered = (rows[(scale, arm)]['hit'] - rows[(scale, 'b')]['hit'])
            frac = recovered / gap if abs(gap) > 1e-9 else float('nan')
            parts.append(f"{arm[2:]}={100 * rows[(scale, arm)]['hit']:.1f}% ({frac:+.0%} of gap)")
        print(f"  {scale}x (gap {100 * gap:+.1f}pp): " + ', '.join(parts))

    print("\ntwo scale-invariance candidates vs the two baselines (hit@0.1 per bucket):")
    print(f"{'bucket':>7} {'per-scale(a)':>13} {'fixed(b)':>10} {'scale_rel':>10} {'canon':>10}")
    for scale in SCALES:
        print(f"{scale:>6}x " + ' '.join(f"{100 * rows[(scale, arm)]['hit']:>12.1f}%"
                                         for arm in ('a', 'b', 'scale_rel', 'canon')))
    for arm in ('a', 'b', 'scale_rel', 'canon'):
        hits = [rows[(s, arm)]['hit'] for s in SCALES]
        print(f"  spread across buckets, {arm}: {100 * (max(hits) - min(hits)):.1f}pp "
              f"(worst bucket {100 * min(hits):.1f}%)")

    print("\ndoes score_and_fit's height term earn its place, and is it scale-sensitive?")
    for scale in SCALES:
        print(f"  {scale}x: correct height {100 * rows[(scale, 'a')]['hit']:.1f}% | "
              f"no height term {100 * rows[(scale, 'a-no_height')]['hit']:.1f}% | "
              f"height wrong by {REF_SCALE / scale:.1f}x, rest correct "
              f"{100 * rows[(scale, 'a-wrong_height')]['hit']:.1f}%")

    # sanity check: at the reference bucket the two arms ARE the same constants, so any
    # difference there would mean the harness itself is non-deterministic
    a, b = rows[(REF_SCALE, 'a')], rows[(REF_SCALE, 'b')]
    assert abs(a['hit'] - b['hit']) < 1e-12 and abs(a['mean'] - b['mean']) < 1e-12, \
        f"ref bucket arms diverged - harness bug: {a} vs {b}"
    print(f"\nsanity: ref bucket {REF_SCALE}x identical across arms (as it must be)")


def alpha_sweep():
    """Sweep the two load-bearing coefficients (max_dist and merge_radius, both as multiples
    of the measured h_ref) at every bucket. The point is not the best value: it is whether
    the SAME pair is best at every bucket, which is what "calibrate once, deploy at any
    scale" requires. If the argmax moves with the bucket, h_ref-relative constants are not
    actually a scale-invariant parameterization and no amount of fitting will fix it."""
    sway_base = arg_value('--sway', 0.0)
    md_alphas = [0.15, 0.25, 0.40]
    mr_alphas = [0.5, 0.75, 1.0, 1.5]
    seqs = SEQS[:3]
    print(f"alpha sweep on {len(seqs)} sequences, sway {sway_base}px@1x\n")
    pooled = {}
    for seq_i, name in enumerate(seqs):
        frames_native, boxes = load_sequence(name)
        for scale in SCALES:
            built = build_bucket(frames_native, boxes, scale, seed=seq_i,
                                 sway_px=sway_base * scale,
                                 distractor_px=arg_value('--distractor', 0.0) * scale)
            frames, boxes_b, norm_hw = built['frames'], built['boxes'], built['norm_hw']
            first_center = next(c for c in sorted(boxes) if c >= SPAN)
            h_ref = estimate_person_height(frames, bg_frames=int(max(5, min(30, first_center))))
            const, kf, pre = scale_relative_config(h_ref)
            for md in md_alphas:
                for mr in mr_alphas:
                    r, nnt, nw = run_arm(frames, boxes_b,
                                         {**const, 'max_dist': md * h_ref, 'merge_radius': mr * h_ref},
                                         kf, pre, norm_hw=norm_hw)
                    pr, pnt, pnw = pooled.get((scale, md, mr), ([], 0, 0))
                    pooled[(scale, md, mr)] = (pr + r, pnt + nnt, pnw + nw)
        print(f"{name} done")

    print(f"\nhit@{HIT_THRESHOLD}, rows = (max_dist alpha, merge alpha), columns = bucket")
    print(f"{'md':>5} {'merge':>6} " + ' '.join(f"{s:>7}x" for s in SCALES) + "   mean  worst")
    best = {}
    for md in md_alphas:
        for mr in mr_alphas:
            hits = [summarize(*pooled[(s, md, mr)])['hit'] for s in SCALES]
            best[(md, mr)] = hits
            print(f"{md:>5} {mr:>6} " + ' '.join(f"{100 * h:>7.1f}%" for h in hits)
                  + f" {100 * np.mean(hits):>6.1f} {100 * min(hits):>6.1f}")
    for i, s in enumerate(SCALES):
        arg = max(best, key=lambda k: best[k][i])
        print(f"best pair at {s}x: md={arg[0]} merge={arg[1]} ({100 * best[arg][i]:.1f}%)")
    arg = max(best, key=lambda k: np.mean(best[k]))
    print(f"best pair on mean-over-buckets: md={arg[0]} merge={arg[1]} "
          f"(per bucket: {', '.join(f'{100 * h:.1f}%' for h in best[arg])})")


if __name__ == '__main__':
    alpha_sweep() if '--alpha-sweep' in sys.argv else main()
