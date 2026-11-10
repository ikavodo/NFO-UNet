"""Survey CDnet 2014 videos for targets where Layer-SAM's assumptions hold and SAM2 masks should miss visible pixels.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tools.cdnet_survey [CDNET_ROOT]

CDnet GT labels: 0 static, 50 shadow, 85 outside ROI, 170 unknown, 255 motion. Per sampled GT frame (temporal ROI,
every STRIDE-th frame) the motion pixels are grouped into objects by dilation (MERGE px), then per object:
n_frag = raw connected components inside it (fragmentation: a person behind branches -> many), fill = motion px / box
area, unk = unknown (170) px / box area. Per video: share of frames with exactly one object, medians over those frames.
Output: results/benchmark/cdnet/survey.csv and one GT/frame montage per video in images/benchmark/cdnet/.
"""
import csv, glob, os, sys
import cv2
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else '../CDnet_2014/dataset'
STRIDE, MERGE, MIN_PX = 5, 15, 30
OUT, IMG = 'results/benchmark/cdnet', 'images/benchmark/cdnet'
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MERGE + 1, 2 * MERGE + 1))


def objects(gt):
    mot = (gt == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(mot, K))
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        m = mot * (lab == i)
        if m.sum() < MIN_PX:
            continue
        nf = cv2.connectedComponents(m)[0] - 1
        box = gt[y:y + h, x:x + w]
        out.append(dict(area=int(m.sum()), n_frag=nf, fill=m.sum() / (w * h), unk=(box == 170).mean(), box=(x, y, w, h)))
    return out


rows = []
for vid in sorted(glob.glob(f'{ROOT}/*/*/')):
    cat, name = vid.rstrip('/').split('/')[-2:]
    if not os.path.exists(f'{vid}/temporalROI.txt'):
        print('skip (no temporalROI):', cat, name); continue
    a, b = map(int, open(f'{vid}/temporalROI.txt').read().split())
    per, show = [], None
    for k in range(a, b + 1, STRIDE):
        gt = cv2.imread(f'{vid}/groundtruth/gt{k:06d}.png', 0)
        if gt is None:
            continue
        obs = objects(gt)
        per.append(obs)
        if len(obs) == 1 and (show is None or obs[0]['n_frag'] > show[1]):
            show = (k, obs[0]['n_frag'])
    one = [o[0] for o in per if len(o) == 1]
    med = lambda key: float(np.median([o[key] for o in one])) if one else float('nan')
    frame0 = cv2.imread(f'{vid}/input/in{a:06d}.jpg', 0)
    rows.append(dict(category=cat, video=name, res=f'{frame0.shape[1]}x{frame0.shape[0]}', roi_frames=b - a + 1,
                     sampled=len(per), pct_empty=round(100 * sum(len(o) == 0 for o in per) / max(len(per), 1)),
                     pct_single=round(100 * len(one) / max(len(per), 1)),
                     pct_multi=round(100 * sum(len(o) > 1 for o in per) / max(len(per), 1)),
                     med_area=round(med('area')), med_frag=med('n_frag'), med_fill=round(med('fill'), 2),
                     med_unk=round(med('unk'), 3), max_frag_frame=show[0] if show else ''))
    if show:   # visual trace: the single-object frame with the most fragments, input | GT
        k = show[0]
        im = cv2.imread(f'{vid}/input/in{k:06d}.jpg'); gt = cv2.imread(f'{vid}/groundtruth/gt{k:06d}.png')
        m = np.hstack([im, gt]); s = 900 / m.shape[1]
        cv2.imwrite(f'{IMG}/{cat}_{name}_{k}.jpg', cv2.resize(m, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(rows[-1], flush=True)

with open(f'{OUT}/survey.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
