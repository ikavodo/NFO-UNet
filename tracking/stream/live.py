"""Live person tracker on a webcam. Real-time, static camera, no model download, no GPU.

    python -m tracking.stream.live                      # default camera
    python -m tracking.stream.live --camera 1
    python -m tracking.stream.live --source data/ido_walk.mkv    # a file, paced like a camera
    python -m tracking.stream.live --yolo               # + a side-by-side YOLO detector panel

ESC or q quits. `--source` exists so the whole application can be exercised without a camera
attached - it is the same code path, only the frame producer differs.

WHAT IT DOES DIFFERENTLY FROM tracking.stream.stream

stream.py is the measurement harness: it writes an mp4 and a montage, scores against ground
truth, and reports jitter. This is the application. It displays, it does not record unless
asked, and it has to survive the two things a live camera imposes that a file does not:

1. THE PERSON HEIGHT MUST BE DISCOVERED WITH NOBODY TO ASK. Every scale-dependent parameter
   comes from one measured person height (scale_relative_params), and the measurement needs the
   person present and moving. On a file you can probe a fixed window; live you cannot, because
   the operator IS the person and has not walked in yet. So this keeps a rolling buffer and
   retries the estimate every probe_frames//4 frames, accepting the first estimate that is both
   plausible (5-95% of frame height) and STABLE - within stable_tol of the previous attempt.
   Two independent attempts agreeing is much harder to fake with one moving curtain than a
   single plausible number is.

   Measured tolerance, so the bar is not arbitrary: sweeping --person-height against NFO ground
   truth, hit@0.1 stays >= 0.96 across 0.75x-1.5x of the true height and only collapses past 3x.
   The estimator itself, on contiguous frames, lands +33%/+17%/-5%/+4% on NFO seq1-4. So
   "stable within 25%" is inside the flat band by construction.

2. THE FRAME RATE IS NOT DECLARED. The Smoother's constants are half-lives in seconds and need
   a real frame interval, so fps is MEASURED over the bootstrap rather than assumed. Passing a
   wrong fps would silently rescale the smoothing.

Camera notes, both load-bearing and both already in webcam_frames(): a daemon thread writes a
single slot so a slow consumer DROPS frames instead of building a backlog, and auto-exposure and
auto white balance are disabled because auto-gain shifts global brightness and MOG2 reads that
as everything-is-foreground. Static camera only, for the same reason.
"""
import argparse
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from tracking.stream.stream import (BUFFER, SPAN, Smoother, StreamPipeline, annotate,
                                    bootstrap_person_height, frames_from_video, webcam_frames)


# yolo11m, not yolo11n: on this footage the nano model returned ~0% real recall and 85% false
# positives on a STATIC shelf object, while the medium model gave 5-26% real recall and 0% false
# positives. A panel showing the nano model would misrepresent what a detector baseline can do.
YOLO_WEIGHTS = 'data/yolo11m.pt'


def hud(vis, lines, colour=(255, 255, 255)):
    """Both passes at the SAME thickness, offset by a pixel, i.e. a drop shadow rather than an
    outline. cv2's Hershey glyph ADVANCE depends on thickness - this line measures 498px at
    thickness 1 and 521px at 2 - so a black outline drawn thicker than the white fill drifts
    progressively rightward and the text renders as two visibly diverging copies. It read as an mp4
    compression ghost; it was in the raw array all along.
    """
    for i, text in enumerate(lines):
        y = 26 + 26 * i
        cv2.putText(vis, text, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return vis


def pick_aligned(pending, target):
    """Newest buffered detection whose frame index is NOT NEWER than `target`, or None.

    Offline, YOLO runs on every frame and the tracker's index is always in the buffer, so pairing is
    a dict lookup. Live it cannot: YOLO costs ~48ms against the tracker's ~8ms, so it sees roughly
    every other frame and the exact index is usually absent. Never reaching FORWARD is the
    load-bearing half - a detector shown a later frame than the tracker is being flattered, and the
    person crosses ~9px/frame here, so a few frames of leak is most of a body width of unearned
    agreement.
    """
    older = [k for k in pending if k <= target]
    return (max(older), pending[max(older)]) if older else None


class YoloWorker:
    """YOLO on a background thread, newest-frame-wins, so the TRACKER never waits for it.

    Not an optimisation - a correctness requirement. Run YOLO inline and the loop drops to ~20fps,
    and because webcam_frames keeps a single slot and discards what the consumer misses, the tracker
    would then receive an IRREGULARLY STRIDED stream. Both its constant-velocity model (nth_frame=2
    assumes uniform spacing) and MOG2's 1/history learning rate assume a fixed frame interval, so
    that shows up as degraded tracking, not merely a slower demo. Threaded, the main loop stays
    tracker-only at ~8ms, consumes the camera at full rate, and YOLO simply skips frames it cannot
    reach - which is the stage that is allowed to skip.
    """

    def __init__(self, weights: str, conf: float, device: str):
        os.environ.setdefault('YOLO_VERBOSE', 'False')
        from ultralytics import YOLO
        from tracking.eval.yolo_vs_tracker import yolo_boxes
        self._infer, self._model = yolo_boxes, YOLO(weights)
        self._conf, self._device = conf, device
        self._job = None                  # (index, frame): newest submission only, older dropped
        self._out = {}                    # index -> [(box, conf), ...]
        self._lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._ms, self._n, self._skipped = 0.0, 0, 0
        # Warm the model HERE, on the main thread, before the loop starts. The first CUDA predict
        # costs seconds (kernel autotune, cuDNN algorithm selection), and left in the worker it
        # would blank the panel for the whole first stretch of footage - a 6.4s test clip completed
        # ZERO inferences before this - as well as landing that one-off compile cost in the
        # reported ms/frame. Startup already blocks for the height bootstrap, so this is free.
        self._infer(self._model, np.zeros((288, 288), np.uint8), self._conf, self._device)
        self._thread = threading.Thread(target=self._loop, daemon=True, name='yolo')
        self._thread.start()

    def submit(self, index: int, frame):
        with self._lock:
            self._skipped += self._job is not None      # overwritten before it was ever started
            self._job = (index, frame)
        self._wake.set()

    def _loop(self):
        while not self._stop.is_set():
            if not self._wake.wait(0.05):
                continue
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            t0 = time.perf_counter()
            boxes = self._infer(self._model, job[1], self._conf, self._device)
            with self._lock:
                self._out[job[0]] = boxes
                self._ms += time.perf_counter() - t0
                self._n += 1

    def take(self, target: int, keep: int):
        """Pair with the tracker's readout, then evict what can never be asked for again."""
        with self._lock:
            hit = pick_aligned(self._out, target)
            for stale in [k for k in self._out if k < target - keep]:
                del self._out[stale]
            return hit

    def stats(self):
        with self._lock:
            return (1000 * self._ms / self._n if self._n else float('nan'), self._n, self._skipped)

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)


def bar(vis, text):
    """Title along the BOTTOM: the top of a live panel is already spoken for by annotate's
    per-frame label and the HUD, and a top bar would paint over both."""
    h = vis.shape[0]
    cv2.rectangle(vis, (0, h - 30), (vis.shape[1] - 1, h - 1), (0, 0, 0), -1)
    cv2.putText(vis, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def yolo_panel(grey, boxes, title):
    vis = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    for (x1, y1, x2, y2), s in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f'{s:.2f}', (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255), 2)
    return bar(vis, title)


def bootstrap(source, probe_frames: int, stable_tol: float, display: bool,
              motion_thresh: int = 20, min_motion: float = 0.001):
    """Consume frames until the person height is both plausible and stable across two attempts.

    Returns (height, buffered_frames, measured_fps). The buffered frames are handed back so the
    caller can replay them through the pipeline - they are MOG2's warm-up, and throwing them
    away would mean starting the background model from zero after the bootstrap.

    A MOTION GATE IS REQUIRED, and plausible-plus-stable is not enough on its own. MOG2's first
    frame is entirely foreground because there is no model yet, so estimate_person_height on a
    completely STATIC stack returns roughly 80% of the frame height - measured, 73px on a 90px
    frame - and that answer is perfectly "stable" across retries because it is deterministic. It
    is also indistinguishable from a real large person by size alone: the C920 run measured a
    genuine 294px on a 360px frame, 82%. So the estimate is only attempted when the probe window
    actually contains frame-to-frame change, which a static scene cannot fake. Without this an
    installation started on an empty room locks a bogus height and then tracks with parameters
    scaled to it.
    """
    # BOUNDED. Only the last probe_frames are ever used, and this loop runs until the height
    # converges - which may be never, if nobody walks in. An unbounded list here grows at
    # frame_bytes * fps: 6.9 MB/s at 640x360 grayscale and 30fps, i.e. 25 GB/hour. A deque with
    # maxlen is the whole fix.
    buf = deque(maxlen=probe_frames)
    motion = deque(maxlen=probe_frames)          # per-frame changed-pixel fraction
    seen, t0, retry = 0, time.perf_counter(), max(1, probe_frames // 4)
    prev = prev_frame = None
    for frame in source:
        if prev_frame is not None:
            motion.append(float((np.abs(frame.astype(np.int16) - prev_frame) > motion_thresh)
                                .mean()))
        prev_frame = frame
        buf.append(frame)
        seen += 1
        h_frame = frame.shape[0]
        moving = len(motion) > 1 and float(np.median(motion)) > min_motion
        if seen >= probe_frames and seen % retry == 0 and moving:
            est = bootstrap_person_height(np.stack(buf))
            plausible = 0.05 * h_frame <= est <= 0.95 * h_frame
            stable = prev is not None and abs(est - prev) <= stable_tol * max(est, prev)
            if plausible and stable:
                fps = seen / max(time.perf_counter() - t0, 1e-6)
                return float(est), deque(buf), fps
            prev = est
        if display:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            msg = [f"BOOTSTRAP  {seen} frames seen, need {probe_frames}",
                   "walk across the frame so the person height can be measured"]
            if len(motion) > 1 and not moving:
                msg.append(f"no motion detected (median {np.median(motion):.4f} of pixels "
                           f"changing, need {min_motion}) - not measuring yet")
            if prev is not None:
                msg.append(f"last estimate {prev:.0f}px ({prev / h_frame:.0%} of frame) "
                           f"- need two agreeing within {stable_tol:.0%}")
            cv2.imshow('live tracker', hud(vis, msg, (0, 255, 255)))
            if cv2.waitKey(1) in (27, ord('q')):
                return None, deque(buf), 0.0
    return None, deque(buf), 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', default='0',
                   help="device index or path, e.g. 0 or /dev/video0. A UVC camera's second "
                        "/dev/videoN is usually its metadata node, not a second camera")
    p.add_argument('--cam-size', default='1280x720', help='requested capture size')
    p.add_argument('--cam-fps', type=float, default=30.0)
    p.add_argument('--fourcc', default='MJPG',
                   help='MJPG is usually the only format offering 30fps above VGA; YUYV often '
                        'caps at 5-10fps at high resolution')
    p.add_argument('--auto-exposure', type=float, default=0.25,
                   help='V4L2 manual-exposure magic value; some drivers want 1')
    p.add_argument('--source', default=None,
                   help='video file to use instead of the camera, paced to its own frame rate')
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--person-height', type=float, default=None,
                   help='skip the bootstrap and use this (pixels, after --scale)')
    p.add_argument('--probe-frames', type=int, default=120)
    p.add_argument('--stable-tol', type=float, default=0.25,
                   help='two consecutive height estimates must agree within this fraction')
    p.add_argument('--halflife', type=float, default=0.15)
    p.add_argument('--jump-max', type=float, default=0.75)
    p.add_argument('--min-confidence', type=float, default=None, metavar='P',
                   help='presence gate in CALIBRATED PROBABILITY, which is the same gate as '
                        '--min-score but in units a human can reason about: 0.90 == score 8.1, '
                        '0.75 == 4.2, 0.50 == 2.0 under the fit in '
                        'tracking/core/score_calibration.json. Overrides --min-score if both are '
                        'given. Fit it with: python -m tracking.eval.calibrate_score')
    p.add_argument('--min-score', type=float, default=0.0,
                   help='presence gate: report nothing when the winning track scores below this')
    p.add_argument('--yolo', nargs='?', const=YOLO_WEIGHTS, default=None, metavar='WEIGHTS',
                   help='add a side-by-side YOLO person-detector panel (bare flag uses '
                        f'{os.path.basename(YOLO_WEIGHTS)}). Runs on a background thread so the '
                        'tracker keeps full frame rate; YOLO skips frames it cannot reach.')
    p.add_argument('--yolo-conf', type=float, default=0.25)
    p.add_argument('--yolo-device', default='cuda', help="'cuda' or 'cpu'")
    p.add_argument('--record', default=None, help='also write an annotated mp4 here')
    p.add_argument('--no-display', dest='display', action='store_false')
    p.add_argument('--status-every', type=float, default=60.0,
                   help='seconds between one-line status reports (frames, fps, RSS)')
    p.add_argument('--max-frames', type=int, default=0,
                   help='stop after this many emitted frames; 0 = run indefinitely')
    p.add_argument('--loop', action='store_true',
                   help='restart --source when it ends, for soak-testing the indefinite path')
    a = p.parse_args()

    if a.source:
        cap = cv2.VideoCapture(a.source)
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        source = (_looped_file(a.source, a.scale) if a.loop
                  else frames_from_video(a.source, a.scale))
        print(f"source {a.source} at {file_fps:g} fps, paced like a camera"
              + (", looping" if a.loop else ""), flush=True)
    else:
        cam = int(a.camera) if a.camera.isdigit() else a.camera
        cw, ch = (int(v) for v in a.cam_size.lower().split('x'))
        source = webcam_frames(cam, a.scale, width=cw, height=ch, fps=a.cam_fps,
                              fourcc=a.fourcc, auto_exposure=a.auto_exposure)
        file_fps = None
        print("auto-exposure and auto-WB disabled, static camera assumed")

    height, warm, fps = a.person_height, [], file_fps or 30.0
    if height is None:
        height, warm, fps = bootstrap(source, a.probe_frames, a.stable_tol, a.display)
        if height is None:
            print("quit during bootstrap")
            cv2.destroyAllWindows()
            return
        print(f"bootstrapped person height {height:.0f}px over {len(warm)} frames, "
              f"measured {fps:.1f} fps")
    if file_fps:
        fps = file_fps

    if a.min_confidence is not None:
        from tracking.core.calibration import load, score_for_confidence
        a.min_score = score_for_confidence(a.min_confidence, load())
        print(f'gate: P(present) >= {a.min_confidence:.2f}  ==  score >= {a.min_score:.2f}')

    pipe = StreamPipeline(person_height=height, min_score=a.min_score)
    sm = Smoother(height, fps, halflife_s=a.halflife, jump_max=a.jump_max)
    writer = None
    shown, t_start = 0, time.perf_counter()
    # fps over a trailing window, not since start: a cumulative average stops reflecting the
    # current rate within minutes, and this is meant to run for weeks
    recent = deque(maxlen=int(max(fps, 1) * 2))
    last_status = t_start
    disp_fps = fps

    yolo = None if a.yolo is None else YoloWorker(a.yolo, a.yolo_conf, a.yolo_device)
    fed = 0

    try:
        for frame in _chain(warm, source):
            if yolo is not None:
                yolo.submit(fed, frame)
            fed += 1
            r = pipe.step(frame)
            if r is None:
                continue
            wh = None if r.box is None else (r.box[2] - r.box[0], r.box[3] - r.box[1])
            r.smooth = sm.update(None if r.x is None else (r.x, r.y), wh)
            vis = annotate(r, disp_fps)
            hud(vis, ["", ""] * (yolo is not None) + ["", f"person {height:.0f}px   latency {SPAN} frames "
                          f"({1000 * SPAN / fps:.0f} ms)   {'TRACKING' if r.x is not None else 'searching'}"])
            if yolo is not None:
                bar(vis, 'blob + Kalman tracker')
                ms, _, _ = yolo.stats()
                hit = yolo.take(r.frame_index, 2 * BUFFER)
                # "no answer yet" is NOT "an answer of nothing" - say which, or an empty panel
                # reads as a confident non-detection when it is really a cold model
                if hit is None:
                    left = yolo_panel(r.frame, [], f'YOLO {os.path.basename(a.yolo)}   warming up')
                else:
                    j, det = hit
                    left = yolo_panel(r.frame, det, f'YOLO {os.path.basename(a.yolo)} '
                                      f'conf>={a.yolo_conf:g}   f{j} ({r.frame_index - j:+d} vs '
                                      f'tracker)   {ms:.0f} ms {a.yolo_device}   '
                                      f'{str(len(det)) + " person" if det else "nothing"}')
                vis = np.hstack([left, vis])
            shown += 1
            now = time.perf_counter()
            recent.append(now)
            if len(recent) > 1:
                disp_fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
            if now - last_status >= a.status_every:
                extra = ''
                if yolo is not None:
                    ms, n, skipped = yolo.stats()
                    extra = f"  yolo {ms:.0f} ms x{n} ({100 * skipped / max(n + skipped, 1):.0f}% frames skipped)"
                print(f"[{(now - t_start) / 3600:6.2f}h] {shown} frames  {disp_fps:5.1f} fps  "
                      f"rss {rss_mb():6.1f} MB{extra}", flush=True)
                last_status = now
            if a.max_frames and shown >= a.max_frames:
                break
            if a.record:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (w, h))
                writer.write(vis)
            if a.display:
                cv2.imshow('live tracker', vis)
                # a live camera is already rate-limited by the camera; only a FILE needs pacing,
                # and never pace by more than the frame interval or the display falls behind
                delay = 1 if file_fps is None else max(
                    1, int(1000 * shown / file_fps - 1000 * (time.perf_counter() - t_start)))
                if cv2.waitKey(delay) in (27, ord('q')):
                    break
    finally:
        if yolo is not None:
            yolo.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    print(f"{shown} frames displayed in {wall:.1f}s = {shown / max(wall, 1e-6):.1f} fps"
          + (f"; wrote {a.record}" if a.record else ""))


def _looped_file(path: str, scale: float):
    """Repeat a file forever - the soak-test stand-in for a camera that never stops."""
    while True:
        n = 0
        for f in frames_from_video(path, scale):
            n += 1
            yield f
        if n == 0:
            raise RuntimeError(f"{path} yielded no frames")


def _chain(buffered, source):
    """The bootstrap frames first (they are MOG2's warm-up), then the live stream.

    Frames are POPPED as they are yielded rather than iterated, so the warm-up block is freed
    during the replay instead of being pinned for the lifetime of the process by this
    generator's own frame.
    """
    while buffered:
        yield buffered.popleft()
    for f in source:
        yield f


def rss_mb() -> float:
    """Current resident set size, MB. Cheap enough to print once a minute, and the only honest
    way to claim a long run is not leaking."""
    try:
        with open('/proc/self/statm') as fh:
            return int(fh.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 20
    except (OSError, IndexError, ValueError):
        return float('nan')


if __name__ == '__main__':
    main()
