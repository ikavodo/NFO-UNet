"""PanAf500 pre-filter for the method's scope (2026-11-11): apes that MOVE, since the integral image only recovers pixels
hidden behind static foliage when the ape moves relative to it. GT annotations only, no masks of any compared method.

Motion per frame (centred 9-frame difference, per ape track):
    v_t = |c(t+4) − c(t−4)| / (8 h_t)  +  |log h(t+4) − log h(t−4)| / 8
The first term is image-plane translation in body heights per frame. The second is scale change: an ape walking towards
or away from the camera barely translates but its box grows/shrinks (log-height change per frame).
A frame moves if v_t > V_MOVE (0.01: sitting/standing p90 ≈ 0.006, walking median 0.015, measured 2026-11-11);
a frame counts only if its GT behaviour is also a moving class (MOV; box jitter alone otherwise passes ~90
sitting/standing/hanging videos), and a track qualifies with ≥ MIN_MOVING such frames (48 = 2 s). This is a coarse pre-filter; the user's visual triage of
the contact sheets is the actual occlusion label.

    PYTHONPATH=. python -m benchmark.panaf.filter_moving      # -> results/panaf/candidates.csv, images/panaf/triage/*.jpg
"""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get('PANAF_ROOT', '../PanAf20K/PanAf500')).resolve()
V_MOVE, MIN_MOVING, PER_SHEET, NSAMP, HT = 0.01, 48, 10, 6, 110
MOV = {'walking', 'running', 'climbing_up', 'climbing_down', 'sitting_on_back'}   # labels of a body that translates


def tracks(js):
    tr = defaultdict(dict)
    for a in js['annotations']:
        for d in a['detections']:
            tr[d['ape_id']][a['frame_id']] = (d['bbox'], d['behaviour'])
    return tr


def moving_frames(fr, labels=None):
    labels = MOV if labels is None else labels
    ks = sorted(fr)
    b = np.array([fr[k][0] for k in ks], float)
    c, h = (b[:, :2] + b[:, 2:]) / 2, np.maximum(b[:, 3] - b[:, 1], 1)
    out = []
    for i in range(4, len(ks) - 4):
        if ks[i + 4] - ks[i - 4] == 8:
            v = np.linalg.norm(c[i + 4] - c[i - 4]) / (8 * h[i]) + abs(np.log(h[i + 4] / h[i - 4])) / 8
            if v > V_MOVE and fr[ks[i]][1] in labels:
                out.append(ks[i])
    return out


def strip(path, fr, ks):
    want = {k - 1: k for k in ks}                      # frame_id is 1-indexed
    cap, i, got = cv2.VideoCapture(str(path)), 0, {}
    while len(got) < len(want):
        ok, im = cap.read()
        if not ok: break
        if i in want:
            x1, y1, x2, y2 = fr[want[i]][0]
            cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, 0.8 * max(x2 - x1, y2 - y1)
            H, W = im.shape[:2]
            c = im[int(max(0, cy - s)):int(min(H, cy + s)), int(max(0, cx - s)):int(min(W, cx + s))]
            c = cv2.resize(c, (HT, HT))
            # GT box outline, in crop coordinates
            sx, sy = HT / max(1, min(W, cx + s) - max(0, cx - s)), HT / max(1, min(H, cy + s) - max(0, cy - s))
            cv2.rectangle(c, (int((x1 - max(0, cx - s)) * sx), int((y1 - max(0, cy - s)) * sy)),
                          (int((x2 - max(0, cx - s)) * sx), int((y2 - max(0, cy - s)) * sy)), (0, 255, 0), 1)
            got[i] = c
        i += 1
    return np.hstack([got.get(k - 1, np.zeros((HT, HT, 3), np.uint8)) for k in ks])


def main(labels=None, tag='triage'):
    """labels None: the moving classes (the candidate filter). STATIC_LAB: apes LABELLED static whose box moves."""
    rows, strips = [], []
    for f in sorted(ROOT.glob('annotations/*/*.json')):
        js = json.load(open(f))
        best = None
        for aid, fr in tracks(js).items():
            mv = moving_frames(fr, labels)
            if len(mv) >= MIN_MOVING and (best is None or len(mv) > len(best[2])):
                best = (aid, fr, mv)
        if best is None:
            continue
        aid, fr, mv = best
        beh = Counter(fr[k][1] for k in mv)
        rows.append(dict(video=js['video'], split=f.parent.name, ape_id=aid, n_moving=len(mv),
                         moving_behaviours=' '.join(f'{k}:{v}' for k, v in beh.most_common()), occluded=''))
        ks = [mv[int(round(x))] for x in np.linspace(0, len(mv) - 1, NSAMP)]
        s = strip(ROOT / 'videos' / f'{js["video"]}.mp4', fr, ks)
        bar = np.zeros((18, s.shape[1], 3), np.uint8)
        cv2.putText(bar, f'{len(rows):3d} {js["video"]} {f.parent.name} ape {aid}: {beh.most_common(2)}', (3, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        strips.append(np.vstack([bar, s]))
        os.makedirs(f'images/panaf/{tag}/strips', exist_ok=True)
        cv2.imwrite(f'images/panaf/{tag}/strips/{js["video"]}.jpg', s, [cv2.IMWRITE_JPEG_QUALITY, 85])   # one per video, for the triage page
    os.makedirs('results/panaf', exist_ok=True); os.makedirs(f'images/panaf/{tag}', exist_ok=True)
    with open('results/panaf/candidates.csv' if tag == 'triage' else f'results/panaf/{tag}.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for i in range(0, len(strips), PER_SHEET):
        cv2.imwrite(f'images/panaf/{tag}/sheet_{i // PER_SHEET:02d}.jpg', np.vstack(strips[i:i + PER_SHEET]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f'{len(rows)} candidate videos ->', Counter(r['split'] for r in rows), f'; {(len(strips) + PER_SHEET - 1) // PER_SHEET} sheets')


STATIC_LAB = {'sitting', 'standing', 'hanging'}

if __name__ == '__main__':
    import sys
    main(STATIC_LAB, 'static_moving') if sys.argv[1:] == ['static'] else main()
