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

import cv2
import numpy as np

from tracking.stream.stream import (Smoother, StreamPipeline, annotate, bootstrap_person_height,
                                    frames_from_source)

DEFAULT_WEIGHTS = '/home/akovi/PycharmProjects/YOLO-26-CAM/models/yolo11n.pt'
PRESENT = {
    'ido_walk':        [(31, 194), (278, 391)],
    'ido_walk_frozen': [],
    'walk_noisy1':     [(65, 195), (353, 453)],
    'walk_noisy2':     [(70, 160), (270, 373)],
}


def yolo_boxes(model, grey, conf):
    """Person boxes as [((x1,y1,x2,y2), conf), ...]. The greyscale frame is replicated to 3
    channels because that is what a COCO-trained network expects, and a previous investigation
    confirmed greyscaling is not itself the cause of failure - the same models still find 4 people
    at 0.87-0.999 in a greyscaled stock photo."""
    r = model.predict(np.repeat(grey[:, :, None], 3, axis=2), classes=[0], conf=conf,
                      verbose=False)[0]
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
    p.add_argument('--out-dir', default='images/stream')
    a = p.parse_args()
    assert os.path.exists(a.weights), f'no weights at {a.weights}'

    os.environ.setdefault('YOLO_VERBOSE', 'False')
    from ultralytics import YOLO
    model = YOLO(a.weights)

    frames = list(frames_from_source(f'data/{a.clip}.mkv', a.scale))
    h = bootstrap_person_height(np.stack(frames[:240]))
    pipe = StreamPipeline(h, min_score=a.min_score)
    sm = Smoother(h, a.src_fps)
    present = PRESENT.get(a.clip, [])
    inp = lambda i: any(lo <= i <= hi for lo, hi in present)

    writer, rows, tally = None, [], {}
    for f in frames:
        r = pipe.step(f)
        if r is None:
            continue
        wh = None if r.box is None else (r.box[2] - r.box[0], r.box[3] - r.box[1])
        r.smooth = sm.update(None if r.x is None else (r.x, r.y), wh)
        det = yolo_boxes(model, r.frame, a.conf)
        left = panel(r.frame, f'YOLO {os.path.basename(a.weights)} conf>={a.conf:g}',
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
        rows.append((r.frame_index, vis, bool(det), r.box is not None))
    if writer is not None:
        writer.release()

    picks = [rows[i] for i in np.linspace(0, len(rows) - 1, 4).astype(int)]
    w = 900
    tiles = [cv2.resize(v, (w, int(w * v.shape[0] / v.shape[1]))) for _, v, _, _ in picks]
    out_png = f'{a.out_dir}/{a.clip}_yolo_vs_tracker.png'
    cv2.imwrite(out_png, np.vstack(tiles))

    print(f'{a.clip}: person height {h:.0f}px, {len(rows)} emitted frames')
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
