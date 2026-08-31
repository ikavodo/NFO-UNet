"""Frame-at-a-time (streaming) wrapper around the offline windowed tracker, plus a demo
that plays a video file through it as a fake stream and draws the merged-blob box.

    python -m tracking.stream.stream --video data/ido_walk.mkv --person-height 421

Everything scale-dependent comes from one measured person height via
tracking.core.track_sequence.scale_relative_params - no absolute pixel constant appears
below. The tracking itself is not reimplemented: step() only maintains the ring buffer and
the single persistent MOG2, then hands the window to the same _result_from_detections the
offline evaluator uses, so the streaming and offline paths cannot silently diverge.

On the readout frame (--readout):

  'center' emits for the frame SPAN=6 behind the newest, i.e. the buffer's center. That is
  what the offline tracker does. When the winning track has no detection at the readout
  frame - the occlusion case this project exists for - the position comes from evaluating a
  least-squares line at that frame, and for an OLS fit over window indices t_i the
  prediction variance is Var[x(t*)] = s^2 * (1/n + (t* - tbar)^2 / S_tt), minimal exactly
  at t* = tbar. For n=7, t=0..6: S_tt=28, so the center costs 1/7 and the newest frame
  1/7 + 9/28, a standard-error ratio of sqrt(0.4643/0.1429) = 1.80x. Asymptotically the
  edge readout is sqrt(1 + 3(n-1)/(n+1)) -> 2x noisier than the center.

  'newest' emits for the newest frame instead: zero steady-state latency, that same fit
  evaluated at the edge of its support. Kept so the trade can be measured rather than
  asserted - run the demo twice and compare the printed jitter.
"""
import argparse
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from tracking.core.preprocess import refine_mask, filter_by_shape, estimate_person_height
from tracking.core.blob_tracker import detect_blobs, _Track
from tracking.core.track_window import _result_from_detections
from tracking.core.track_sequence import scale_relative_params

SEQ_SIZE, NTH_FRAME = 7, 2
SPAN = (SEQ_SIZE // 2) * NTH_FRAME      # 6 - frames of lookahead for a centered readout
BUFFER = 2 * SPAN + 1                   # 13 - so a window exists for *every* input frame


@dataclass
class Result:
    frame_index: int                    # absolute index of the frame this describes
    frame: np.ndarray                   # that frame, greyscale
    x: float | None
    y: float | None
    box: tuple | None                   # merged multi-blob box (x1, y1, x2, y2)
    extrapolated: bool                  # readout frame had no detection in the winning track
    score: float | None
    winner: dict | None
    detections: list = ()      # HOG boxes in full-frame coords, filled in by run()


class StreamPipeline:
    """Ring buffer of the last BUFFER frames. step(frame) returns a Result once the buffer
    is full, else None."""

    def __init__(self, person_height: float, bg_frames: int = 30, var_threshold: float = 16.0,
                 readout: str = 'center', min_solidity: float = 0.1, max_age: int = 6,
                 min_track_length: int = 3, max_detections: int = 40):
        assert readout in ('center', 'newest'), readout
        self.kw, kalman = scale_relative_params(person_height)
        # ponytail: _Track's covariances are class attributes, so this is process-global.
        # Fine for one pipeline, or several at the same person height (which is what the
        # readout comparison runs); make them instance state if two scales ever need to be
        # live at once.
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = kalman
        self.readout, self.min_solidity = readout, min_solidity
        self.max_age, self.min_track_length = max_age, min_track_length
        self.max_detections = max_detections
        # one persistent subtractor: preprocess.foreground_mask builds a fresh MOG2 per
        # call, which is exactly what a stream must not do
        self.mog = cv2.createBackgroundSubtractorMOG2(history=bg_frames,
                                                      varThreshold=var_threshold,
                                                      detectShadows=False)
        self.frames, self.dets = deque(maxlen=BUFFER), deque(maxlen=BUFFER)
        self.seen = 0

    def step(self, frame: np.ndarray) -> Result | None:
        mask = self.mog.apply(frame)[None]
        mask = refine_mask(mask, self.kw['close_kernel_size'], self.kw['open_kernel_size'])
        mask = filter_by_shape(mask, min_area=self.kw['min_area'], min_solidity=self.min_solidity)
        dets = detect_blobs(mask, min_area=self.kw['min_area'])[0]
        if len(dets) > self.max_detections:
            # Hungarian assignment is O(n^3); bound it against a frame where MOG2 goes wild
            dets = sorted(dets, key=lambda d: -d['area'])[:self.max_detections]
        self.frames.append(frame)
        self.dets.append(dets)
        self.seen += 1
        if len(self.frames) < BUFFER:
            return None

        window = [self.dets[i] for i in range(0, BUFFER, NTH_FRAME)]
        center_t = SEQ_SIZE // 2 if self.readout == 'center' else SEQ_SIZE - 1
        buf_i = center_t * NTH_FRAME
        r = _result_from_detections(window, self.min_track_length, self.kw['expected_height'],
                                    0.5, self.kw['max_dist'], self.max_age,
                                    self.kw['merge_radius'], center_t=center_t, return_box=True)
        return Result(frame_index=self.seen - 1 - (BUFFER - 1 - buf_i),
                      frame=self.frames[buf_i],
                      x=r and r['x'], y=r and r['y'], box=r and r['box'],
                      extrapolated=bool(r and r['extrapolated']),
                      score=r and r['score'], winner=r and r['winner'])


# --------------------------------------------------------------------------- demo

def frames_from_video(path: str, scale: float = 1.0):
    """Yield greyscale frames one at a time - the fake stream."""
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cannot open {path}"
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield grey
    cap.release()


def annotate(result: Result, fps: float) -> np.ndarray:
    vis = cv2.cvtColor(result.frame, cv2.COLOR_GRAY2BGR)
    for (dx1, dy1, dx2, dy2), conf in result.detections:
        cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (255, 0, 255), 2)
        cv2.putText(vis, f"HOG {conf:.2f}", (dx1, max(14, dy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    if result.box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in result.box)
        colour = (0, 165, 255) if result.extrapolated else (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(vis, (int(round(result.x)), int(round(result.y))), 4, colour, -1)
    label = f"f{result.frame_index}  {fps:5.1f} fps"
    if result.x is None:
        label += "  no track"
    else:
        label += f"  score {result.score:.0f}"
        if result.extrapolated:
            label += "  fitted readout"
    if result.detections:
        label += f"  |  {len(result.detections)} HOG"

    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


_HOG = None


def hog_detector():
    """OpenCV's built-in HOG people detector, built once. No model download, no GPU."""
    global _HOG
    if _HOG is None:
        if not hasattr(cv2, 'HOGDescriptor'):
            raise RuntimeError(
                f"cv2.HOGDescriptor is absent from OpenCV {cv2.__version__}: HOG was removed "
                f"in OpenCV 5. Run this with an OpenCV 4 interpreter (on this machine: "
                f"~/miniconda3/envs/spacejam/bin/python, cv2 4.10.0), or pass --no-hog. "
                f"Everything except the detector overlay works on either.")
        _HOG = cv2.HOGDescriptor()
        _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _HOG


def detect_person(frame: np.ndarray, centre, person_height: float, pad: float = 0.25,
                  target_px: float = 128.0, win_stride=(8, 8), scale_step: float = 1.05,
                  hit_threshold: float = 0.0):
    """HOG on the CENTRE FRAME, restricted to a scale-relative crop around the tracker's
    merged box. Returns [((x1, y1, x2, y2), confidence), ...] in FULL-FRAME coordinates, so
    the detector can be drawn straight on top of the merged box.

    target_px replaces the "2x upscale, mandatory" rule from the earlier notes. HOG's
    detection window is 64x128, so what governs whether it fires is the person's height in the
    image it is handed RELATIVE TO 128 - not an absolute factor measured once on one dataset at
    one person size. The crop is resized so person_height maps to target_px, which makes the
    setting scale-free the same way every other parameter here is. Restricting to the tracker's
    own box is what makes this cheap and keeps HOG off the foliage; box=None falls back to the
    whole frame.

    The crop is PERSON-SHAPED, not the merged box: HOG's window has a fixed 1:2 aspect, and
    the merged box does not (measured on ido_walk its median height is 309px against a 421px
    person, so it both under-covers and has an arbitrary aspect). So only the box's CENTRE is
    used, and the window is (0.9 + 2*pad) x (1.5 + 2*pad) person heights around it.

    weights from detectMultiScale are SVM decision values, not probabilities - report them as
    the detector's own margin and do not call them a probability. hit_threshold is that same
    margin's accept cutoff; lowering it below 0 surfaces weaker detections rather than none.
    """
    h, w = frame.shape
    if centre is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        cx, cy = centre
        hw, hh = (0.45 + pad) * person_height, (0.75 + pad) * person_height
        x1, y1 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
        x2, y2 = min(w, int(round(cx + hw))), min(h, int(round(cy + hh)))
    if x2 - x1 < 16 or y2 - y1 < 32:
        return []
    crop = frame[y1:y2, x1:x2]
    k = target_px / person_height
    if abs(k - 1.0) > 1e-3:
        crop = cv2.resize(crop, None, fx=k, fy=k,
                          interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    if crop.shape[0] < 128 or crop.shape[1] < 64:
        return []
    rects, weights = hog_detector().detectMultiScale(crop, winStride=win_stride,
                                                     padding=(8, 8), scale=scale_step,
                                                     hitThreshold=hit_threshold)
    return [((int(x1 + rx / k), int(y1 + ry / k),
              int(x1 + (rx + rw) / k), int(y1 + (ry + rh) / k)), float(c))
            for (rx, ry, rw, rh), c in zip(rects, np.ravel(weights))]


def jitter(xs_by_index: dict) -> float:
    """RMS discrete acceleration of the reported x, pooled over consecutive frame triples.
    A person walking has near-zero true acceleration, so this is dominated by estimation
    noise - the observable the readout-variance argument in the module docstring predicts.
    Takes an index->x map (rather than a list) so gaps, e.g. untracked frames or a
    restriction to the person-present interval, cannot be silently treated as adjacent."""
    ks = sorted(xs_by_index)
    a = [xs_by_index[k] - 2 * xs_by_index[k + 1] + xs_by_index[k + 2]
         for i, k in enumerate(ks) if k + 1 in xs_by_index and k + 2 in xs_by_index]
    return float(np.sqrt(np.mean(np.square(a)))) if a else float('nan')


def in_ranges(f: int, ranges) -> bool:
    return not ranges or any(lo <= f <= hi for lo, hi in ranges)


def run(video: str, person_height: float, scale: float = 0.5, readout: str = 'center',
        out_dir: str = 'images/stream', tag: str = '', display: bool = False,
        src_fps: float = 24.0, present=(), out_fps: float = None,
        hog: bool = True, hog_target: float = 128.0, hog_pad: float = 0.25,
        hog_thresh: float = 0.0) -> dict:
    pipe = StreamPipeline(person_height=person_height, readout=readout)
    if person_height / 7.5 < 40:
        print(f"note: head is ~{person_height / 7.5:.0f}px at this scale; face-level tasks "
              f"are not viable, person detection is the realistic downstream task")

    writer, results, t_start, shown, compute, hog_time = None, [], time.perf_counter(), 0, 0.0, 0.0
    for frame in frames_from_video(video, scale):
        t0 = time.perf_counter()
        r = pipe.step(frame)
        compute += time.perf_counter() - t0
        if r is None:
            continue
        if hog:
            t1 = time.perf_counter()
            centre = None if r.x is None else (r.x, r.y)
            r.detections = detect_person(r.frame, centre, person_height, pad=hog_pad,
                                         target_px=hog_target, hit_threshold=hog_thresh)
            hog_time += time.perf_counter() - t1
        results.append(r)
        fps = len(results) / (time.perf_counter() - t_start)
        vis = annotate(r, fps)
        if writer is None:
            frame_h, frame_w = h, w = vis.shape[:2]
            writer = cv2.VideoWriter(f"{out_dir}/{tag or readout}_stream.mp4",
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     out_fps or src_fps, (w, h))
        assert vis.shape[:2] == (frame_h, frame_w), (
            f"frame {r.frame_index} is {vis.shape[:2]} but the writer was opened at "
            f"{(frame_h, frame_w)}; cv2.VideoWriter would drop it silently")
        writer.write(vis)
        if display:
            cv2.imshow('stream', vis)
            # pace against a monotonic start, so playback does not drift by however long
            # the tracker took on each frame
            behind = 1000 * (shown + 1) / src_fps - 1000 * (time.perf_counter() - t_start)
            if cv2.waitKey(max(1, int(behind))) == 27:
                break
        shown += 1
    if writer is not None:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    wall = time.perf_counter() - t_start
    boxed = [r for r in results if r.box is not None]
    extrap = [r for r in results if r.extrapolated]
    montage([r for r in boxed if in_ranges(r.frame_index, present)],
            f"{out_dir}/{tag or readout}_montage.png")
    xs = {r.frame_index: r.x for r in results if r.x is not None and in_ranges(r.frame_index, present)}
    stats = dict(readout=readout, emitted=len(results), boxed=len(boxed),
                 extrapolated=len(extrap),
                 # compute cost, never the paced interval - with --display the two differ
                 ms_per_frame=1000 * compute / max(pipe.seen, 1),
                 fps=pipe.seen / compute, wall_fps=shown / wall,
                 jitter_px=jitter(xs), jitter_n=len(xs),
                 latency_frames=SPAN if readout == 'center' else 0)
    if hog:
        fired = [r for r in results if r.detections]
        best = [max(c for _, c in r.detections) for r in fired]
        stats.update(hog_fired=len(fired) / max(len(results), 1),
                     hog_fired_when_boxed=sum(1 for r in boxed if r.detections) / max(len(boxed), 1),
                     hog_mean_best_conf=float(np.mean(best)) if best else float('nan'),
                     hog_ms_per_frame=1000 * hog_time / max(len(results), 1))
    print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in stats.items()))
    return stats


def montage(results: list[Result], path: str, n: int = 6, cols: int = 3, width: int = 480):
    """Contact sheet of n evenly spaced annotated results - the visual trace of the run."""
    if not results:
        return
    picks = [results[i] for i in np.linspace(0, len(results) - 1, n).astype(int)]
    tiles = []
    for r in picks:
        vis = annotate(r, float('nan'))
        tiles.append(cv2.resize(vis, (width, int(width * vis.shape[0] / vis.shape[1]))))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack([r for r in rows if r.shape == rows[0].shape]))
    print(f"wrote {path}")


def compare_readouts(video: str, person_height: float, scale: float = 0.5,
                     out_dir: str = 'images/stream', present=()) -> dict:
    """Paired center-vs-newest readout comparison, one variable changed.

    Both pipelines are fed the same frames, so their ring buffers are identical at every
    step, and score_and_fit does not take the readout index at all - so both arms pick the
    SAME winning track from the SAME window. The only difference is which frame the
    position is read out at, which is exactly the quantity in question. Results are aligned
    by the frame each one describes, so the centered arm's 6-frame lookahead is accounted
    for rather than compared away.
    """
    pipes = {ro: StreamPipeline(person_height, readout=ro) for ro in ('center', 'newest')}
    xs = {ro: {} for ro in pipes}
    extrap = {ro: set() for ro in pipes}
    for frame in frames_from_video(video, scale):
        for ro, pipe in pipes.items():
            r = pipe.step(frame)
            if r is not None and r.x is not None:
                xs[ro][r.frame_index] = r.x
                if r.extrapolated:
                    extrap[ro].add(r.frame_index)
    common = [f for f in sorted(set(xs['center']) & set(xs['newest'])) if in_ranges(f, present)]
    dx = np.array([abs(xs['center'][f] - xs['newest'][f]) for f in common])
    out = dict(n=len(common),
               extrap_center=len(extrap['center'] & set(common)) / max(len(common), 1),
               extrap_newest=len(extrap['newest'] & set(common)) / max(len(common), 1),
               median_dx=float(np.median(dx)) if len(dx) else float('nan'),
               p90_dx=float(np.percentile(dx, 90)) if len(dx) else float('nan'))
    keep = set(common)
    for ro in pipes:
        sel = {f: xs[ro][f] for f in keep}
        out[f'jitter_{ro}'] = jitter(sel)
        # the fit-variance argument only bites where the readout frame had no detection
        fb = extrap[ro] & keep
        out[f'jitter_{ro}_fallback'] = jitter({f: xs[ro][f] for f in sel
                                               if {f, f + 1, f + 2} & fb})
    print("  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()))
    _plot_comparison(xs, present, f'{out_dir}/05_readout_comparison.png')
    return out


def _plot_comparison(xs, present, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    for ro, style in (('center', '-'), ('newest', '--')):
        ks = sorted(xs[ro])
        ax.plot(ks, [xs[ro][k] for k in ks], style, lw=1.2,
                label=f"readout={ro} (latency {SPAN if ro == 'center' else 0} frames)")
    for lo, hi in present:
        ax.axvspan(lo, hi, color='g', alpha=.12)
    ax.set_xlabel('frame'); ax.set_ylabel('reported x [px]')
    ax.set_title('Centered vs newest-frame readout, same windows, same winning tracks '
                 '(green = person present)')
    ax.legend(); plt.tight_layout(); plt.savefig(path, dpi=110)
    print(f'wrote {path}')


def measure_presence(video: str, scale: float = 0.5, out_dir: str = 'images/stream',
                     min_run: int = 5, area_frac: float = 0.01, diff_thresh: int = 30):
    """Person-present frame ranges, from a temporal-median background - independent of the
    tracker, so it can scope the tracker's evaluation without circularity. Valid only
    because the camera is static; it is a scoping aid, not ground truth."""
    fr = np.stack(list(frames_from_video(video, scale)))
    bg = np.median(fr, axis=0).astype(np.int16)
    k = np.ones((9, 9), np.uint8)
    frac = np.array([cv2.morphologyEx((np.abs(f.astype(np.int16) - bg) > diff_thresh).astype(np.uint8),
                                      cv2.MORPH_CLOSE, k).mean() for f in fr])
    runs, start = [], None
    for i, p in enumerate(frac > area_frac):
        if p and start is None:
            start = i
        elif not p and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(frac) - 1))
    print(f'{len(fr)} frames, foreground > {area_frac:.0%} of frame on '
          f'{(frac > area_frac).sum()} of them; present runs: '
          + ' '.join(f'{a}:{b}' for a, b in runs))
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video', default='data/ido_walk.mkv')
    p.add_argument('--scale', type=float, default=0.5, help='resize factor applied to every frame')
    p.add_argument('--person-height', type=float, default=None,
                   help='person height in pixels AFTER --scale; estimated from the footage if omitted')
    p.add_argument('--readout', choices=('center', 'newest'), default='center')
    p.add_argument('--out-dir', default='images/stream')
    p.add_argument('--tag', default='', help='output filename prefix (defaults to --readout)')
    p.add_argument('--display', action='store_true', help='cv2.imshow paced at the source fps')
    p.add_argument('--no-hog', dest='hog', action='store_false',
                   help='skip the HOG person detector overlay')
    p.add_argument('--hog-target', type=float, default=128.0,
                   help="resize the crop so the person is this tall before HOG; HOG's own "
                        "window is 64x128, so this is the scale-free version of an upscale factor")
    p.add_argument('--hog-pad', type=float, default=0.25,
                   help='margin around the person-shaped crop fed to HOG, in person heights')
    p.add_argument('--hog-thresh', type=float, default=0.0,
                   help="HOG's SVM margin cutoff; below 0 surfaces weaker detections")
    p.add_argument('--out-fps', type=float, default=None,
                   help='frame rate stamped on the written mp4; below the source rate this '
                        'gives slow motion for frame-by-frame inspection (e.g. 6 = 4x slow). '
                        'Does not affect --display pacing or any measurement.')
    p.add_argument('--compare', action='store_true',
                   help='paired center-vs-newest readout comparison instead of a single run')
    p.add_argument('--measure-presence', action='store_true',
                   help='print person-present frame ranges and exit')
    p.add_argument('--present', nargs='*', default=[], metavar='LO:HI',
                   help='restrict the reported statistics to these frame ranges')
    a = p.parse_args()
    present = [tuple(int(v) for v in r.split(':')) for r in a.present]

    if a.measure_presence:
        measure_presence(a.video, a.scale, a.out_dir)
        return

    height = a.person_height
    if height is None:
        probe = np.stack([f for i, f in enumerate(frames_from_video(a.video, a.scale)) if i % 4 == 0][:60])
        height = estimate_person_height(probe)
        print(f"estimated person height {height:.1f}px from {len(probe)} probe frames "
              f"(pass --person-height to override; the estimator is unreliable on footage "
              f"where the person is absent for most of the probe)")
    if a.compare:
        compare_readouts(a.video, height, scale=a.scale, out_dir=a.out_dir, present=present)
    else:
        run(a.video, height, scale=a.scale, readout=a.readout, out_dir=a.out_dir, tag=a.tag,
            display=a.display, present=present, out_fps=a.out_fps, hog=a.hog,
            hog_target=a.hog_target, hog_pad=a.hog_pad, hog_thresh=a.hog_thresh)


if __name__ == '__main__':
    main()
