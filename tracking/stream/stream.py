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
import itertools
import os
import threading
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
    smooth: tuple = None       # ((x, y), (w, h), coasting) from Smoother, filled by run()


class StreamPipeline:
    """Ring buffer of the last BUFFER frames. step(frame) returns a Result once the buffer
    is full, else None."""

    def __init__(self, person_height: float, bg_frames: int = 30, var_threshold: float = 16.0,
                 readout: str = 'center', min_solidity: float = 0.1, max_age: int = 6,
                 min_track_length: int = 3, max_detections: int = 40,
                 suppress_warmup: bool = True):
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
        # MOG2's learning rate is ~1/history, so before `history` real frames have elapsed the
        # background model is under-adapted and the mask is full of spurious foreground. That is
        # the dominant source of early false positives, and track_sequence's own docstring
        # already says any window before bg_frames have elapsed is under-adapted. Suppressing
        # output until then costs the first bg_frames-1-SPAN emissions and introduces no new
        # constant.
        self.warmup = bg_frames if suppress_warmup else 0
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
        if len(self.frames) < BUFFER or self.seen < self.warmup:
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


class Smoother:
    """Holt's linear (double) exponential smoothing of the tracker's output, with a validation
    gate and coast.

    WHY NOT A PLAIN EMA. A first-order EMA with time constant tau lags a target moving at v by
    v*tau. Measured on ido_walk the person moves ~8px per frame, so a 0.15s half-life (tau ~ 5
    frames at 24fps) would cost ~40px of systematic lag on a 421px person - about 10% of a body
    height - bought with the jitter it removes. Holt's method carries a TREND term as well as a
    level, so it is unbiased for constant velocity, which is exactly the motion model the
    tracker already assumes:

        level_t = a*z_t + (1-a)*(level_{t-1} + trend_{t-1})
        trend_t = b*(level_t - level_{t-1}) + (1-b)*trend_{t-1}

    WHY A GATE. The failure this is actually for - the tracker jumping to a foliage blob for a
    handful of frames - is not something a linear low-pass can reject: it averages the outlier
    in and spreads it over MORE frames than it arrived in. So a measurement further than
    jump_max person-heights from the prediction is rejected and the filter coasts on its own
    trend, for at most hold_s seconds; past that the jump is accepted, because the person may
    genuinely have been re-acquired somewhere else. That is the validation-gate-and-coast of
    classical target tracking (Blackman and Popoli), applied to the OUTPUT rather than to
    association - track_blobs already gates association with max_dist.

    Both smoothing constants are half-lives IN SECONDS, converted with the real frame interval,
    so behaviour is identical at 24 fps and at whatever a webcam delivers. A per-frame alpha
    would be a hidden frame-rate constant - the same class of mistake as a hidden pixel
    constant, which this codebase spent a lot of effort removing.

    Latency: a half-life h adds a group delay of roughly h to the reported position, on top of
    the tracker's SPAN=6 frames of lookahead. At the 0.15s default and 24 fps that is ~3.6
    frames, so the end-to-end budget is ~10 frames (0.4s), not 6.
    """

    def __init__(self, person_height: float, fps: float, halflife_s: float = 0.15,
                 trend_halflife_s: float = 0.5, size_halflife_s: float = 0.4,
                 jump_max: float = 0.75, hold_s: float = 0.35,
                 init_m: int = 3, init_n: int = 5):
        dt = 1.0 / max(fps, 1e-6)
        self.a = 1.0 - 0.5 ** (dt / max(halflife_s, 1e-6))
        self.b = 1.0 - 0.5 ** (dt / max(trend_halflife_s, 1e-6))
        self.c = 1.0 - 0.5 ** (dt / max(size_halflife_s, 1e-6))
        self.jump_max = jump_max * person_height
        self.hold = max(1, int(round(hold_s * fps)))
        self.init_m, self.init_n = init_m, init_n
        self.reset()

    def reset(self):
        self.level = None
        self.trend = np.zeros(2)
        self.size = None
        self.misses = 0
        self.init_buf = deque(maxlen=self.init_n)

    def update(self, xy, wh):
        """xy: (x, y) or None. wh: (w, h) or None. Returns ((x, y) | None, (w, h) | None,
        coasting)."""
        if self.level is None:
            # M-of-N track initiation (Blackman and Popoli): do not declare a lock on a single
            # measurement. An isolated warm-up false positive would otherwise BECOME the lock,
            # and then the real person arrives as a large jump and gets gated out for hold_s.
            # Requiring init_m of the last init_n measurements to agree within the gate rejects
            # one-off spikes without needing a magnitude threshold on the score.
            if xy is None:
                return None, None, False
            self.init_buf.append((np.array(xy, float), wh))
            pts = np.array([q for q, _ in self.init_buf])
            med = np.median(pts, axis=0)
            close = [i for i, q in enumerate(pts)
                     if float(np.hypot(*(q - med))) <= self.jump_max]
            if len(close) < self.init_m:
                return None, None, False
            sel = pts[close]
            self.level = sel[-1]
            # seed the trend from the consistent run, so the filter starts with the right
            # velocity instead of spending its first half-life catching up
            self.trend = (sel[-1] - sel[0]) / max(len(sel) - 1, 1)
            last_wh = self.init_buf[close[-1]][1]
            if last_wh is not None:
                self.size = np.array(last_wh, float)
            return self._out(False)

        pred = self.level + self.trend
        gated = xy is None or float(np.hypot(*(np.array(xy, float) - pred))) > self.jump_max
        if gated:
            self.misses += 1
            if self.misses > self.hold:
                self.reset()
                return self.update(xy, wh)
            self.level = pred          # coast on the trend
            return self._out(True)

        self.misses = 0
        prev, z = self.level, np.array(xy, float)
        self.level = self.a * z + (1.0 - self.a) * pred
        self.trend = self.b * (self.level - prev) + (1.0 - self.b) * self.trend
        if wh is not None:
            w = np.array(wh, float)
            self.size = w if self.size is None else self.c * w + (1.0 - self.c) * self.size
        return self._out(False)

    def _out(self, coasting):
        return (tuple(self.level),
                None if self.size is None else tuple(self.size),
                coasting)


# --------------------------------------------------------------------------- demo

def _open_camera(index, width, height, fps, fourcc, auto_exposure):
    # cv2.CAP_V4L2 EXPLICITLY. Opening a PATH like '/dev/video2' without naming the backend
    # routes to FFMPEG, which silently ignores every V4L2 property: measured on the C920,
    # VideoCapture('/dev/video2') negotiates 640x480 with fourcc \x00 and backend FFMPEG, while
    # VideoCapture('/dev/video2', cv2.CAP_V4L2) gives 1280x720 @30 MJPG. Opening by integer index
    # happens to pick V4L2 anyway, so this bug only appears with the path form - which is the
    # form anyone reading `v4l2-ctl --list-devices` will naturally reach for. Measured end to
    # end on the live app: 8.9 fps via FFMPEG against 24.1 fps via V4L2.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    got = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"camera {index!r}: backend {cap.getBackendName()}, negotiated "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"@ {cap.get(cv2.CAP_PROP_FPS):g}fps, fourcc "
          f"{''.join(chr((got >> 8 * i) & 0xFF) for i in range(4))}", flush=True)
    return cap


def webcam_frames(index=0, scale: float = 1.0, width: int = 1280, height: int = 720,
                  fps: float = 30.0, fourcc: str = 'MJPG', auto_exposure: float = 0.25,
                  reconnect: bool = True, reconnect_wait: float = 2.0):
    """Newest-frame-only webcam source: a daemon thread writes a single slot and the consumer
    takes whatever is there. A fixed-latency pipeline must never accumulate a backlog - if the
    consumer falls behind, the right thing is to DROP frames, not to queue them.

    `index` may be a device index or a path like '/dev/video0'. Note that a UVC camera usually
    exposes TWO nodes - a capture node and a metadata node - so the second /dev/videoN is
    typically not a second camera. `v4l2-ctl --list-devices` says which is which.

    FOURCC IS SET FIRST AND IT MATTERS. Read off the actual hardware with v4l2-ctl:

        Logitech HD Pro Webcam C920   MJPG 1920x1080@30   YUYV 1920x1080@5, 2304x1536@2
        Integrated_Webcam_FHD         MJPG 1920x1080@30   YUYV 1920x1080@5, 1280x720@10

    YUYV is what the V4L2 backend often negotiates by default, so requesting MJPG is the
    difference between a live demo and a slideshow - and it is nothing to do with the tracker,
    which costs 2.5ms/frame. The negotiated settings are printed rather than assumed, because
    V4L2 silently substitutes whatever it can do instead of failing.

    Auto-exposure and auto white balance are switched off on purpose: auto-gain shifts global
    brightness, and MOG2 reads a global brightness shift as everything-is-foreground. Static
    camera only, for the same reason. auto_exposure=0.25 is the V4L2 "manual" magic value on most
    drivers; some want 1, and some ignore it - pass a different value if exposure still drifts.
    """
    # cv2.CAP_V4L2 EXPLICITLY. Opening a PATH like '/dev/video2' without naming the backend
    # routes to FFMPEG, which silently ignores every V4L2 property: measured on the C920,
    # VideoCapture('/dev/video2') negotiates 640x480 with fourcc \x00 and backend FFMPEG, while
    # VideoCapture('/dev/video2', cv2.CAP_V4L2) gives 1280x720 @30 MJPG. Opening by integer index
    # happens to pick V4L2 anyway, so this bug only appears with the path form - which is the
    # form anyone reading `v4l2-ctl --list-devices` will naturally reach for.
    opened_once = False
    while True:
        cap = _open_camera(index, width, height, fps, fourcc, auto_exposure)
        if cap is None:
            assert opened_once, (
                f"cannot open camera {index!r}. `v4l2-ctl --list-devices` lists the capture "
                f"nodes; a UVC camera's second node is usually metadata, not a second camera")
            print(f"camera gone; retrying in {reconnect_wait:g}s", flush=True)
            time.sleep(reconnect_wait)
            continue
        opened_once = True
        slot, stop = [None], threading.Event()

        def reader():
            while not stop.is_set():
                ok, f = cap.read()
                if not ok:
                    stop.set()
                    break
                slot[0] = f

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            while not stop.is_set():
                f, slot[0] = slot[0], None
                if f is None:
                    time.sleep(0.002)
                    continue
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                yield g if scale == 1.0 else cv2.resize(g, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_AREA)
        finally:
            # also runs on GeneratorExit when the consumer stops, and the exception then
            # propagates past the reconnect loop, so closing the generator does NOT reopen
            stop.set()
            t.join(timeout=1.0)
            cap.release()
        if not reconnect:
            return
        # A USB reset or an unplugged camera must not end an installation that is meant to run
        # for weeks, so reopen instead of falling out of the generator.
        print(f"camera read failed; reconnecting in {reconnect_wait:g}s", flush=True)
        time.sleep(reconnect_wait)


def frames_from_source(path, scale: float = 1.0):
    """A video file, a directory of greyscale jpgs (the NFO layout), or a webcam index."""
    if isinstance(path, int) or (isinstance(path, str) and path.isdigit()):
        yield from webcam_frames(int(path), scale)
    elif os.path.isdir(path):
        for n in sorted(f for f in os.listdir(path) if f.endswith('.jpg')):
            g = cv2.imread(os.path.join(path, n), 0)
            yield g if scale == 1.0 else cv2.resize(g, None, fx=scale, fy=scale,
                                                    interpolation=cv2.INTER_AREA)
    else:
        yield from frames_from_video(path, scale)


def load_gt(path: str):
    """nfo_processed groundtruth.txt: `frame,x,y,w,h`, normalised. Returns {frame: (x,y,w,h)}."""
    gt = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) == 5:
                gt[int(parts[0])] = tuple(float(v) for v in parts[1:])
    return gt


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
    if result.box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in result.box)
        colour = (0, 165, 255) if result.extrapolated else (0, 255, 0)
        thin = 1 if result.smooth else 2      # raw box recedes once a smoothed one is drawn
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, thin)
        cv2.circle(vis, (int(round(result.x)), int(round(result.y))), 3, colour, -1)
    if result.smooth:
        (sxy, swh, coasting) = result.smooth
        if sxy is not None:
            sc = (0, 255, 255) if coasting else (255, 255, 0)
            if swh is not None:
                cv2.rectangle(vis, (int(sxy[0] - swh[0] / 2), int(sxy[1] - swh[1] / 2)),
                              (int(sxy[0] + swh[0] / 2), int(sxy[1] + swh[1] / 2)), sc, 2)
            cv2.circle(vis, (int(round(sxy[0])), int(round(sxy[1]))), 5, sc, -1)
    label = f"f{result.frame_index}  {fps:5.1f} fps"
    if result.x is None:
        label += "  no track"
    else:
        label += f"  score {result.score:.0f}"
        if result.extrapolated:
            label += "  fitted readout"
    if result.smooth and result.smooth[2]:
        label += "  COASTING"
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


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


def bootstrap_person_height(frames: np.ndarray, bg_frames: int = 30) -> float:
    """Measure the person height from a CONTIGUOUS block of frames.

    Contiguity is not incidental. estimate_person_height works off MOG2 foreground, so
    sub-sampling the frames before handing them over makes every moving object look bigger
    between consecutive samples and inflates the estimate badly - measured against NFO ground
    truth, a stride-4 probe of 60 frames returned +38%/+177%/+105%/+149% on seq1-4, while the
    same 240 frames taken contiguously return +33%/+17%/-5%/+4%. The tracker is flat over
    0.75x-1.5x of the true height (measured: hit@0.1 >= 0.96 across that band on all four
    sequences), so contiguous is inside the usable band and strided is not.

    It still needs the person to be present and moving somewhere in the block: on a
    person-absent window of ido_walk.mkv it returns 266px, measuring leaf motion. So treat a
    wildly implausible answer as a signal to pass --person-height, not as a measurement.
    """
    return float(estimate_person_height(frames, bg_frames=bg_frames))


def run(video, person_height: float = None, scale: float = 0.5, readout: str = 'center',
        out_dir: str = 'images/stream', tag: str = '', display: bool = False,
        src_fps: float = 24.0, present=(), out_fps: float = None, smooth: bool = True,
        halflife: float = 0.15, trend_halflife: float = 0.5, jump_max: float = 0.75,
        gt_path: str = None, probe_frames: int = 240, montage_all: bool = False,
        suppress_warmup: bool = True, init_m: int = 1) -> dict:
    source = frames_from_source(video, scale)
    warmup = []
    if person_height is None:
        for f in source:
            warmup.append(f)
            if len(warmup) >= probe_frames:
                break
        assert warmup, f"no frames from {video}"
        person_height = bootstrap_person_height(np.stack(warmup))
        h_frame = warmup[0].shape[0]
        print(f"bootstrapped person height {person_height:.0f}px from {len(warmup)} contiguous "
              f"frames ({person_height / h_frame:.0%} of frame height)")
        if not 0.05 * h_frame <= person_height <= 0.95 * h_frame:
            print(f"  WARNING: that is implausible for a person; the estimator measures any "
                  f"large moving object, so pass --person-height explicitly")
    pipe = StreamPipeline(person_height=person_height, readout=readout,
                          suppress_warmup=suppress_warmup)
    sm = Smoother(person_height, src_fps, halflife_s=halflife,
                  trend_halflife_s=trend_halflife, jump_max=jump_max,
                  init_m=init_m) if smooth else None
    gt = load_gt(gt_path) if gt_path else None
    if person_height / 7.5 < 40:
        print(f"note: head is ~{person_height / 7.5:.0f}px at this scale; face-level tasks "
              f"are not viable, person detection is the realistic downstream task")

    writer, results, t_start, shown, compute = None, [], time.perf_counter(), 0, 0.0
    # the probe frames are fed through the pipeline too, so they double as MOG2's warm-up
    # instead of being consumed and thrown away
    for frame in itertools.chain(warmup, source):
        t0 = time.perf_counter()
        r = pipe.step(frame)
        compute += time.perf_counter() - t0
        if r is None:
            continue
        if sm is not None:
            wh = None if r.box is None else (r.box[2] - r.box[0], r.box[3] - r.box[1])
            r.smooth = sm.update(None if r.x is None else (r.x, r.y), wh)
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
    montage([r for r in (results if montage_all else boxed)
             if in_ranges(r.frame_index, present)],
            f"{out_dir}/{tag or readout}_montage.png")
    keep = lambda r: in_ranges(r.frame_index, present)
    xs = {r.frame_index: r.x for r in results if r.x is not None and keep(r)}
    sxs = {r.frame_index: r.smooth[0][0] for r in results
           if r.smooth and r.smooth[0] is not None and keep(r)}
    stats = dict(readout=readout, emitted=len(results), boxed=len(boxed),
                 extrapolated=len(extrap),
                 # compute cost, never the paced interval - with --display the two differ
                 ms_per_frame=1000 * compute / max(pipe.seen, 1),
                 fps=pipe.seen / compute, wall_fps=shown / wall,
                 jitter_px=jitter(xs), jitter_n=len(xs),
                 latency_frames=SPAN if readout == 'center' else 0)
    if sm is not None:
        stats.update(jitter_smoothed=jitter(sxs), jitter_n_smoothed=len(sxs),
                     coasting=sum(1 for r in results if r.smooth and r.smooth[2]) / max(len(results), 1))
    if gt is not None:
        H, W = results[0].frame.shape[:2]
        for tag, get in (('raw', lambda r: (r.x, r.y) if r.x is not None else None),
                         ('smoothed', lambda r: r.smooth[0] if r.smooth else None)):
            res = []
            for r in results:
                p = get(r)
                if p is None or r.frame_index not in gt:
                    continue
                g = gt[r.frame_index]
                res.append(float(np.hypot((p[0] - (g[0] + g[2] / 2) * W) / W,
                                          (p[1] - (g[1] + g[3] / 2) * H) / H)))
            if res:
                res = np.array(res)
                stats[f'resid_{tag}'] = float(res.mean())
                stats[f'hit_{tag}'] = float((res <= 0.1).mean())
                stats[f'n_{tag}'] = len(res)
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
    p.add_argument('--video', default='data/ido_walk.mkv',
                   help='video file, directory of greyscale jpgs, or a webcam index (e.g. 0)')
    p.add_argument('--no-smooth', dest='smooth', action='store_false',
                   help='report the raw per-frame tracker output with no smoothing')
    p.add_argument('--halflife', type=float, default=0.15,
                   help='position half-life in SECONDS (frame-rate independent)')
    p.add_argument('--trend-halflife', type=float, default=0.5,
                   help="half-life of Holt's trend term, in seconds")
    p.add_argument('--jump-max', type=float, default=0.75,
                   help='reject-and-coast gate on the smoother, in person heights')
    p.add_argument('--src-fps', type=float, default=24.0,
                   help='source frame rate; sets the smoother constants and --display pacing')
    p.add_argument('--gt', default=None,
                   help='nfo_processed groundtruth.txt to score residual/hit against')
    p.add_argument('--scale', type=float, default=0.5, help='resize factor applied to every frame')
    p.add_argument('--person-height', type=float, default=None,
                   help='person height in pixels AFTER --scale; bootstrapped from the first '
                        '--probe-frames contiguous frames if omitted')
    p.add_argument('--init-m', type=int, default=1,
                   help='M-of-5 track initiation for the smoother; 1 disables it (lock on the '
                        'first measurement)')
    p.add_argument('--montage-all', action='store_true',
                   help='sample the montage from every emission, including ones with no box')
    p.add_argument('--no-warmup-suppression', dest='suppress_warmup', action='store_false',
                   help="emit during MOG2's warm-up too (shows the early false positives)")
    p.add_argument('--probe-frames', type=int, default=240,
                   help='contiguous frames used to bootstrap the person height; they are then '
                        'fed through the pipeline as well, so they are not wasted')
    p.add_argument('--readout', choices=('center', 'newest'), default='center')
    p.add_argument('--out-dir', default='images/stream')
    p.add_argument('--tag', default='', help='output filename prefix (defaults to --readout)')
    p.add_argument('--display', action='store_true', help='cv2.imshow paced at the source fps')
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

    if a.compare:
        height = a.person_height
        if height is None:
            probe = np.stack([f for _, f in zip(range(a.probe_frames),
                                                frames_from_source(a.video, a.scale))])
            height = bootstrap_person_height(probe)
            print(f"bootstrapped person height {height:.0f}px")
        compare_readouts(a.video, height, scale=a.scale, out_dir=a.out_dir, present=present)
    else:
        run(a.video, a.person_height, scale=a.scale, readout=a.readout, out_dir=a.out_dir,
            tag=a.tag, probe_frames=a.probe_frames,
            display=a.display, present=present, out_fps=a.out_fps, smooth=a.smooth,
            halflife=a.halflife, trend_halflife=a.trend_halflife, jump_max=a.jump_max,
            src_fps=a.src_fps, gt_path=a.gt, montage_all=a.montage_all,
            suppress_warmup=a.suppress_warmup, init_m=a.init_m)


if __name__ == '__main__':
    main()
