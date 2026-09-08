"""Split screen: a COCO-trained YOLO person detector against the blob/Kalman tracker.

    python -m tracking.eval.yolo_vs_tracker --clip walk_noisy1
    python -m tracking.eval.yolo_vs_tracker --clip ido_walk --weights /path/to/yolo11n.pt

Left panel is YOLO, right panel is the tracker, same frame and same scale, so the only difference
is the method. Writes <clip>_yolo_vs_tracker.mp4 plus a montage, and prints recall and
false-positive rate for BOTH methods against hand-labelled presence intervals.

WHY THIS IS WORTH RUNNING AT ALL, given the repo already says not to. Three prior derivations
found COCO detectors returning 0.000 confidence on this project's footage - the deleted
gen_data/score_frames_yolo.py on NFO with yolov8n, then torchvision Faster R-CNN and YOLOv8n on
NFO, KTH and every crop variant. But all of those were 195px-or-smaller people in outdoor foliage.
These clips are a different regime: an 840px well-lit indoor person at full resolution, which is
the one case where a COCO detector should be comfortable. So it was untested rather than settled.

WEIGHTS come from whatever is already on the machine - no download. yolo11n.pt lives in the
sibling YOLO-26-CAM project.

READ recall AND false positives together. A detector that fires on most frames looks good on
recall alone; the interesting question on this footage is whether it fires on the PERSON.
"""
import argparse
import os
import time

import cv2
import numpy as np

from tracking.stream.stream import (BUFFER, SPAN, Smoother, StreamPipeline, annotate,
                                    bootstrap_person_height, frames_from_source)

DEFAULT_WEIGHTS = '/home/akovi/PycharmProjects/YOLO-26-CAM/models/yolo11n.pt'
PRESENT = {
    'ido_walk':        [(31, 194), (278, 391)],
    'ido_walk_frozen': [],
    'walk_noisy1':     [(65, 195), (353, 453)],
    'walk_noisy2':     [(70, 160), (270, 373)],
}


def select_montage(rows, mode: str, want: int = 4):
    """Which rows go in the montage - and 'evenly spaced across the whole clip' is the wrong
    default the moment the point is to demonstrate one specific outcome.

    'spread'        every row, evenly spaced - what the clip looks like overall
    'tracker-wins'  only rows where ground truth says present, the tracker found a box, YOLO
                    found nothing, and the box is fully inside the frame - the frames that
                    actually make the tracker's case. A frame where both miss, where the tracker
                    fires while absent, or where the box is clipped at the frame edge (a real
                    detection of a partially-visible person, but reads as broken rendering in a
                    montage meant to make a clean visual case) is not a clean win.

    Rows: (frame_index, vis, yolo_fired, tracker_has_box, gt_present, box_in_bounds). box_in_bounds
    is meaningless when tracker_has_box is False - a miss is excluded for being a miss, not for its
    unset bounds flag. Empty in, empty out - no silent fallback to 'spread' when nothing qualifies,
    because that would show frames the caller explicitly said were not what they asked for.
    """
    pool = rows if mode == 'spread' else [r for r in rows if r[4] and r[3] and not r[2] and r[5]]
    if not pool:
        return []
    n = min(want, len(pool))
    return [pool[i] for i in np.linspace(0, len(pool) - 1, n).astype(int)]


def yolo_boxes(model, grey, conf, device=None):
    """Person boxes as [((x1,y1,x2,y2), conf), ...]. The greyscale frame is replicated to 3
    channels because that is what a COCO-trained network expects, and a previous investigation
    confirmed greyscaling is not itself the cause of failure - the same models still find 4 people
    at 0.87-0.999 in a greyscaled stock photo."""
    r = model.predict(np.repeat(grey[:, :, None], 3, axis=2), classes=[0], conf=conf,
                      device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xy = r.boxes.xyxy.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()
    return [((int(a), int(b), int(c), int(d)), float(s)) for (a, b, c, d), s in zip(xy, cf)]


def panel(grey, title, boxes=None, result=None, fired=None):
    vis = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR) if result is None else annotate(result, float('nan'))
    for (x1, y1, x2, y2), s in (boxes or []):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f'{s:.2f}', (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 255), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(vis, title + ('' if fired is None else f'   {fired}'), (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--clip', default='walk_noisy1')
    p.add_argument('--weights', default=DEFAULT_WEIGHTS)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--conf', type=float, default=0.25, help="ultralytics' own default")
    p.add_argument('--min-score', type=float, default=5.0, help='tracker presence gate')
    p.add_argument('--src-fps', type=float, default=24.0)
    p.add_argument('--device', default='cuda', help="'cuda' or 'cpu'")
    p.add_argument('--readout', choices=('center', 'newest'), default='center',
                   help="'newest' makes the tracker causal, which removes the SPAN offset entirely "
                        "and with it the need for the YOLO buffer - both then answer about the same "
                        "newest frame. Costs +11%% jitter and 27%% more fitted readouts, measured.")
    p.add_argument('--select', choices=('spread', 'tracker-wins'), default='spread',
                   help="'tracker-wins' picks only frames where ground truth says present, the "
                        "tracker found a box, and YOLO found nothing - i.e. the frames that "
                        "demonstrate the tracker's advantage, rather than a generic sample of the "
                        "clip. Written to <clip>_tracker-wins.png instead of the default montage.")
    p.add_argument('--out-dir', default='images/stream')
    a = p.parse_args()
    assert os.path.exists(a.weights), f'no weights at {a.weights}'

    os.environ.setdefault('YOLO_VERBOSE', 'False')
    from ultralytics import YOLO
    model = YOLO(a.weights)

    frames = list(frames_from_source(f'data/{a.clip}.mkv', a.scale))
    h = bootstrap_person_height(np.stack(frames[:240]))
    pipe = StreamPipeline(h, min_score=a.min_score, readout=a.readout)
    hold = SPAN if a.readout == 'center' else 0
    sm = Smoother(h, a.src_fps)
    present = PRESENT.get(a.clip, [])
    inp = lambda i: any(lo <= i <= hi for lo, hi in present)

    # THE ALIGNMENT. YOLO infers on the NEWEST frame, as a real-time system must - it has no
    # reason to wait. The tracker emits for the frame SPAN=6 behind the newest, because its window
    # is centred. So YOLO's answers are buffered by frame index and the one from SPAN frames ago is
    # flushed when the tracker catches up to it. Pairing them per loop iteration instead would put
    # the panels 6 frames (250ms at 24fps) out of sync AND flatter YOLO, which would be showing a
    # later frame than the tracker.
    #
    # This is a restructure of a previously correct-by-accident version that ran YOLO on the
    # tracker's already-emitted frame. Same pixels reach YOLO either way, so the recall and
    # false-positive numbers must come out IDENTICAL - which is the check that the buffering is
    # right rather than merely plausible.
    pending, writer, rows, tally = {}, None, [], {}
    yolo_ms, track_ms, n_yolo = 0.0, 0.0, 0
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        pending[i] = yolo_boxes(model, f, a.conf, a.device)
        yolo_ms += time.perf_counter() - t0
        n_yolo += 1
        for stale in [k for k in pending if k < i - 2 * BUFFER]:
            del pending[stale]                       # bound it even if the tracker never emits
        t0 = time.perf_counter()
        r = pipe.step(f)
        track_ms += time.perf_counter() - t0
        if r is None:
            continue
        wh = None if r.box is None else (r.box[2] - r.box[0], r.box[3] - r.box[1])
        r.smooth = sm.update(None if r.x is None else (r.x, r.y), wh)
        det = pending.pop(r.frame_index, [])
        left = panel(r.frame, f'YOLO {os.path.basename(a.weights)} conf>={a.conf:g} '
                              f'(newest frame, held {hold})',
                     boxes=det, fired=f'{len(det)} person' if det else 'nothing')
        right = panel(r.frame, f'blob + Kalman tracker  min-score {a.min_score:g}', result=r,
                      fired='box' if r.box is not None else 'nothing')
        vis = np.hstack([left, right])
        if writer is None:
            hh, ww = vis.shape[:2]
            writer = cv2.VideoWriter(f'{a.out_dir}/{a.clip}_yolo_vs_tracker.mp4',
                                     cv2.VideoWriter_fourcc(*'mp4v'), a.src_fps, (ww, hh))
        writer.write(vis)
        if present:
            k = 'PRESENT' if inp(r.frame_index) else 'absent'
            t = tally.setdefault(k, [0, 0, 0])
            t[0] += 1
            t[1] += bool(det)
            t[2] += r.box is not None
        # margin, not a bare >=0/<=width check: a box that reaches exactly to the frame edge is a
        # genuine detection of a person walking OUT of frame (verified: (729,116,960,351) and
        # (743,78,960,540) on a 960x540 frame, both landing on x2==960 exactly) - correct, but the
        # visible portion is a sliver against the boundary with nothing beyond it, which is what
        # reads as "clipped" in a montage meant to show a clean win. 12px keeps genuinely
        # boundary-touching boxes out without discarding anything merely near an edge.
        margin = 12
        box_in_bounds = (r.box is not None and r.box[0] >= margin and r.box[1] >= margin
                        and r.box[2] <= r.frame.shape[1] - margin
                        and r.box[3] <= r.frame.shape[0] - margin)
        rows.append((r.frame_index, vis, bool(det), r.box is not None,
                    bool(present) and inp(r.frame_index), box_in_bounds))
    if writer is not None:
        writer.release()

    picks = select_montage(rows, a.select)
    # same predicate select_montage applies internally, asked for with want=len(rows) so nothing
    # is left out - duplicating a boolean expression here would drift out of sync with it silently,
    # as happened once already when box_in_bounds was added to the filter but not to this line
    pool_size = len(rows) if a.select == 'spread' else len(select_montage(rows, a.select, want=len(rows)))
    suffix = '_yolo_vs_tracker' if a.select == 'spread' else f'_{a.select}'
    out_png = f'{a.out_dir}/{a.clip}{suffix}.png'
    if not picks:
        print(f'no frames satisfy --select {a.select} for {a.clip}; not writing {out_png}')
    else:
        w = 900
        tiles = [cv2.resize(v, (w, int(w * v.shape[0] / v.shape[1]))) for _, v, *_ in picks]
        cv2.imwrite(out_png, np.vstack(tiles))
        print(f'{len(picks)}/{pool_size} qualifying frames shown in {out_png}')

    print(f'{a.clip}: person height {h:.0f}px, {len(rows)} emitted frames, readout '
          f'{a.readout}, YOLO held {hold} frames (buffer depth {hold + 1})')
    per = lambda ms: 1000 * ms / max(n_yolo, 1)
    print(f'  timing: YOLO {per(yolo_ms):5.1f} ms/frame on {a.device}, tracker '
          f'{per(track_ms):4.1f} ms/frame CPU  ->  serial {per(yolo_ms) + per(track_ms):5.1f} ms '
          f'({1000 / (per(yolo_ms) + per(track_ms)):4.1f} fps), '
          f'parallel {max(per(yolo_ms), per(track_ms)):5.1f} ms '
          f'({1000 / max(per(yolo_ms), per(track_ms)):4.1f} fps)')
    if tally:
        print(f"{'segment':>9}{'frames':>8}{'YOLO fires':>12}{'tracker box':>13}")
        for k in ('PRESENT', 'absent'):
            if k in tally:
                n, y, t = tally[k]
                print(f'{k:>9}{n:>8}{100 * y / n:>11.0f}%{100 * t / n:>12.0f}%')
        if 'PRESENT' in tally and 'absent' in tally:
            np_, yp, tp = tally['PRESENT']
            na, ya, ta = tally['absent']
            print(f'\n  YOLO    recall {100*yp/np_:.0f}%  false-positive {100*ya/na:.0f}%'
                  f'   -> {"ANTI-CORRELATED with truth" if ya/na > yp/np_ else "correlated"}')
            print(f'  tracker recall {100*tp/np_:.0f}%  false-positive {100*ta/na:.0f}%')
    print(f'wrote {a.out_dir}/{a.clip}_yolo_vs_tracker.mp4 and {out_png}')


if __name__ == '__main__':
    main()
