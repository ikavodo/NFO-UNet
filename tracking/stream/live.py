"""Live person tracker on a webcam. Real-time, static camera, no model download, no GPU.

    python -m tracking.stream.live                      # default camera
    python -m tracking.stream.live --camera 1
    python -m tracking.stream.live --source data/ido_walk.mkv    # a file, paced like a camera

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
import time

import cv2
import numpy as np

from tracking.stream.stream import (BUFFER, SPAN, Smoother, StreamPipeline, annotate,
                                    bootstrap_person_height, frames_from_video, webcam_frames)


def hud(vis, lines, colour=(255, 255, 255)):
    for i, text in enumerate(lines):
        cv2.putText(vis, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 1)
    return vis


def bootstrap(source, probe_frames: int, stable_tol: float, display: bool):
    """Consume frames until the person height is both plausible and stable across two attempts.

    Returns (height, buffered_frames, measured_fps). The buffered frames are handed back so the
    caller can replay them through the pipeline - they are MOG2's warm-up, and throwing them
    away would mean starting the background model from zero after the bootstrap.
    """
    buf, attempts, t0, retry = [], [], time.perf_counter(), max(1, probe_frames // 4)
    prev = None
    for frame in source:
        buf.append(frame)
        h_frame = frame.shape[0]
        if len(buf) >= probe_frames and len(buf) % retry == 0:
            est = bootstrap_person_height(np.stack(buf[-probe_frames:]))
            plausible = 0.05 * h_frame <= est <= 0.95 * h_frame
            stable = prev is not None and abs(est - prev) <= stable_tol * max(est, prev)
            attempts.append((est, plausible, stable))
            if plausible and stable:
                fps = len(buf) / max(time.perf_counter() - t0, 1e-6)
                return float(est), buf, fps
            prev = est
        if display:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            msg = [f"BOOTSTRAP  {len(buf)}/{probe_frames} frames",
                   "walk across the frame so the person height can be measured"]
            if prev is not None:
                msg.append(f"last estimate {prev:.0f}px ({prev / h_frame:.0%} of frame) "
                           f"- need two agreeing within {stable_tol:.0%}")
            cv2.imshow('live tracker', hud(vis, msg, (0, 255, 255)))
            if cv2.waitKey(1) in (27, ord('q')):
                return None, buf, 0.0
    return None, buf, 0.0


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
    p.add_argument('--record', default=None, help='also write an annotated mp4 here')
    p.add_argument('--no-display', dest='display', action='store_false')
    a = p.parse_args()

    if a.source:
        cap = cv2.VideoCapture(a.source)
        file_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cap.release()
        source = frames_from_video(a.source, a.scale)
        print(f"source {a.source} at {file_fps:g} fps, paced like a camera")
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

    pipe = StreamPipeline(person_height=height)
    sm = Smoother(height, fps, halflife_s=a.halflife, jump_max=a.jump_max)
    writer = None
    shown, t_start, last_report = 0, time.perf_counter(), time.perf_counter()
    disp_fps = fps

    try:
        for frame in _chain(warm, source):
            r = pipe.step(frame)
            if r is None:
                continue
            wh = None if r.box is None else (r.box[2] - r.box[0], r.box[3] - r.box[1])
            r.smooth = sm.update(None if r.x is None else (r.x, r.y), wh)
            vis = annotate(r, disp_fps)
            hud(vis, ["", f"person {height:.0f}px   latency {SPAN} frames "
                          f"({1000 * SPAN / fps:.0f} ms)   {'TRACKING' if r.x is not None else 'searching'}"])
            shown += 1
            now = time.perf_counter()
            if now - last_report >= 1.0:
                disp_fps = shown / (now - t_start)
                last_report = now
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
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    print(f"{shown} frames displayed in {wall:.1f}s = {shown / max(wall, 1e-6):.1f} fps"
          + (f"; wrote {a.record}" if a.record else ""))


def _chain(buffered, source):
    """The bootstrap frames first (they are MOG2's warm-up), then the live stream."""
    for f in buffered:
        yield f
    for f in source:
        yield f


if __name__ == '__main__':
    main()
