"""Build a TensorRT engine for the YOLO panel, verify it against the .pt, and time both.

    python -m tracking.stream.export_trt                        # build data/yolo11m.engine
    python -m tracking.stream.export_trt --weights data/yolo11m.pt --imgsz 640
    python -m tracking.stream.export_trt --check                 # just report engine vs environment

Then point the live app at it - nothing else changes, ultralytics loads a .engine like a .pt:

    python -m tracking.stream.live --camera /dev/video2 --yolo data/yolo11m.engine

AN ENGINE IS NOT A PORTABLE FILE. This is the single most important thing about this script. TRT
compiles kernels for one compute capability, one TRT version, one CUDA version, and bakes the input
resolution in. An engine built on this laptop (sm_120 Blackwell, TRT 11.2, CUDA 13) will NOT
deserialize on the Zotac unless the Zotac matches all of it, and the error TRT prints on mismatch
names none of the four. So the engine CANNOT be prepared here and shipped: only yolo11m.pt and this
script travel, and the build runs on the machine that will run the inference. The sidecar .json
written next to the engine exists to turn that cryptic crash into a sentence.

WHY BOTHER AT ALL, given the tracker beats YOLO on this footage (90%/1% against 17%/0% on
walk_noisy1)? Only for the demo panel. TRT does not make the detector better, it makes the
side-by-side affordable - the honest framing is presentation cost, not detection quality.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

FIELDS = {'compute_cap': 'GPU compute capability', 'trt': 'TensorRT version',
          'cuda': 'CUDA version', 'imgsz': 'input size', 'half': 'FP16 flag',
          'pt_sha': 'source checkpoint'}


def engine_is_stale(fingerprint, env):
    """None if the engine is usable here, else a sentence naming every field that moved.

    Reports ALL mismatches rather than the first: someone moving a build between machines has
    usually changed the GPU and the TRT version together, and fixing one at a time wastes a
    multi-minute rebuild per attempt.
    """
    if not fingerprint:
        return ('no fingerprint sidecar next to the engine, so it cannot be checked against this '
                'machine - rebuild it')
    bad = [f'{label} was {fingerprint.get(key)!r} at build time, this machine has {env[key]!r}'
           for key, label in FIELDS.items() if fingerprint.get(key) != env[key]]
    return None if not bad else 'engine does not match this environment: ' + '; '.join(bad)


def sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()[:12]


def environment(weights, imgsz, half):
    import tensorrt
    import torch
    cap = torch.cuda.get_device_capability()
    return {'compute_cap': f'{cap[0]}.{cap[1]}', 'trt': tensorrt.__version__,
            'cuda': torch.version.cuda, 'imgsz': imgsz, 'half': half, 'pt_sha': sha(weights),
            'gpu': torch.cuda.get_device_name(0)}


def side_path(engine):
    return os.path.splitext(engine)[0] + '.engine.json'


def load_fingerprint(engine):
    p = side_path(engine)
    return json.load(open(p)) if os.path.exists(p) else None


def bench(model, frames, device, conf, warmup=10, runs=60):
    """Median, not mean: the first calls after a load include autotune and cuDNN selection, and one
    2-second outlier would swamp a mean over 60 runs and overstate the steady-state cost."""
    from tracking.eval.yolo_vs_tracker import yolo_boxes
    for i in range(warmup):
        yolo_boxes(model, frames[i % len(frames)], conf, device)
    ms, boxes = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        boxes.append(yolo_boxes(model, frames[i % len(frames)], conf, device))
        ms.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ms)), boxes


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / union if union > 0 else 0.0


def agreement(a, b, iou_min=0.5):
    """Do the two models find the same people? Greedy IoU matching, NOT zip over sorted lists.

    The first version of this zipped the two frames' box lists after sorting them by coordinate,
    which pairs unrelated boxes the moment the counts differ - model A's only box against model B's
    first of two - and duly reported a "worst corner shift" of 222px that was entirely an artefact
    of the pairing. FP16 does not move a box by a third of the frame. Match first, then measure, and
    report what failed to match separately with its confidence, because a detection that appears in
    one model and not the other is a different phenomenon from one that shifted.
    """
    same_count = sum(len(x) == len(y) for x, y in zip(a, b))
    shifts, only_a, only_b = [], [], []
    for x, y in zip(a, b):
        free = list(range(len(y)))
        for bx, cf in x:
            best = max(free, key=lambda j: iou(bx, y[j][0]), default=None)
            if best is not None and iou(bx, y[best][0]) >= iou_min:
                free.remove(best)
                shifts.append(max(abs(np.array(bx) - np.array(y[best][0]))))
            else:
                # record the BEST IoU it did reach: a box that moved past the match threshold is a
                # shifted detection, not a lost one, and only this number separates the two
                only_a.append((cf, iou(bx, y[best][0]) if best is not None else 0.0))
        only_b += [(y[j][1], max((iou(bx, y[j][0]) for bx, _ in x), default=0.0)) for j in free]
    return same_count, len(a), shifts, only_a, only_b


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default='data/yolo11m.pt')
    p.add_argument('--imgsz', type=int, default=640, help='baked into the engine, not a runtime arg')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--no-half', dest='half', action='store_false', help='build FP32 instead of FP16')
    p.add_argument('--clip', default='data/ido_walk.mkv', help='frames to verify and time against')
    p.add_argument('--check', action='store_true',
                   help='only report whether an existing engine matches this machine')
    p.add_argument('--force', action='store_true', help='rebuild even if the engine is current')
    a = p.parse_args()

    engine = os.path.splitext(a.weights)[0] + '.engine'
    env = environment(a.weights, a.imgsz, a.half)
    print(f"{env['gpu']}  sm_{env['compute_cap']}  TRT {env['trt']}  CUDA {env['cuda']}")

    if a.check:
        if not os.path.exists(engine):
            print(f'no engine at {engine} - build one with this script ON THIS MACHINE')
            return
        why = engine_is_stale(load_fingerprint(engine), env)
        print(f'{engine}: ' + (why or 'matches this environment, safe to load'))
        return

    os.environ.setdefault('YOLO_VERBOSE', 'False')
    from ultralytics import YOLO
    from tracking.stream.stream import frames_from_source

    fresh = os.path.exists(engine) and not engine_is_stale(load_fingerprint(engine), env)
    if fresh and not a.force:
        print(f'{engine} already matches this environment; --force to rebuild')
    else:
        t0 = time.perf_counter()
        YOLO(a.weights).export(format='engine', imgsz=a.imgsz, half=a.half, device=0, verbose=False)
        json.dump(env, open(side_path(engine), 'w'), indent=1)
        print(f'built {engine} in {time.perf_counter() - t0:.0f}s '
              f'({os.path.getsize(engine) / 1e6:.0f} MB), fingerprint -> {side_path(engine)}')

    frames = list(frames_from_source(a.clip, 0.5))[60:120]
    pt_ms, pt_boxes = bench(YOLO(a.weights), frames, 'cuda', a.conf)
    tr_ms, tr_boxes = bench(YOLO(engine, task='detect'), frames, None, a.conf)
    print(f'\n  .pt      {pt_ms:6.1f} ms/frame  (median of 60, CUDA)')
    print(f'  .engine  {tr_ms:6.1f} ms/frame  ->  {pt_ms / tr_ms:.2f}x')

    same, n, shifts, only_pt, only_tr = agreement(pt_boxes, tr_boxes)
    print(f'\n  agreement over {n} frames: identical box count on {same}/{n}')
    if shifts:
        print(f'  {len(shifts)} boxes matched at IoU>=0.5: median corner shift '
              f'{np.median(shifts):.1f}px, worst {max(shifts):.1f}px')
    for label, items in (('.pt only', only_pt), ('.engine only', only_tr)):
        for cf, best in sorted(items, reverse=True):
            print(f'  1 detection {label:12s} conf {cf:.2f} (threshold {a.conf:g}), '
                  f'best IoU against the other model {best:.2f}'
                  + ('  <- shifted, not lost' if best > 0.1 else '  <- absent entirely'))
    if not only_pt and not only_tr:
        print('  every detection matched: FP16 changed no decision at this threshold')


if __name__ == '__main__':
    main()
