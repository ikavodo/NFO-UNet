#!/usr/bin/env python3
"""PanAf500 GT-only motion audit (2026-11-11): does the behaviour label follow box motion, which videos are static,
and how many tracks move enough for the integral image (needs ape-vs-occluder motion). -> renders/motion_stats.csv
Speed v = |centre(t+4) - centre(t-4)| / 8 / box height (body heights per frame, centred 9-frame difference)."""
import json, csv, collections
from pathlib import Path
import numpy as np
ROOT = Path(__import__('os').environ.get('PANAF_ROOT', '../PanAf20K/PanAf500')).resolve()   # run from the NFO-UNet root
STATIC = {'sitting', 'standing', 'sitting_on_back'}
per_label, rows, trans = collections.defaultdict(list), [], collections.Counter()
for f in sorted(ROOT.glob('annotations/*/*.json')):
    d = json.load(open(f)); tr = collections.defaultdict(dict)
    for a in d['annotations']:
        for det in a['detections']:
            tr[det['ape_id']][a['frame_id']] = (det['bbox'], det['behaviour'])
    nb = nstat = 0; moving_tracks = 0; labs = collections.Counter()
    for aid, fr in tr.items():
        ks = sorted(fr); b = np.array([fr[k][0] for k in ks], float); lab = [fr[k][1] for k in ks]
        c = (b[:, :2] + b[:, 2:]) / 2; h = np.maximum(b[:, 3] - b[:, 1], 1); w = np.maximum(b[:, 2] - b[:, 0], 1)
        for i in range(4, len(ks) - 4):
            if ks[i + 4] - ks[i - 4] == 8:
                per_label[lab[i]].append(np.linalg.norm(c[i + 4] - c[i - 4]) / 8 / h[i])
        for i in range(1, len(lab)):
            if lab[i] != lab[i - 1]: trans[(lab[i - 1], lab[i])] += 1
        ext = np.ptp(c[:, 0]) / np.median(w) if len(ks) > 1 else 0
        moving_tracks += ext > 1.0
        nb += len(lab); nstat += sum(l in STATIC for l in lab); labs.update(lab)
    rows.append(dict(video=d['video'], split=f.parent.name, n_tracks=len(tr), n_boxes=nb, frac_static=round(nstat / max(nb, 1), 3),
                     moving_tracks=moving_tracks, labels=' '.join(f'{k}:{v}' for k, v in labs.most_common())))
with open(Path("results/panaf") / 'motion_stats.csv', 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('speed per label (body heights/frame): n, median, p90, frac > 0.02')
for l, v in sorted(per_label.items(), key=lambda x: -len(x[1])):
    v = np.array(v); print(f'  {l:18s} {len(v):6d}  {np.median(v):.4f}  {np.percentile(v, 90):.4f}  {np.mean(v > 0.02):.2f}')
print('top label transitions:', trans.most_common(10))
n = len(rows); vids = [r for r in rows if r['n_boxes']]
fs = np.array([r['frac_static'] for r in vids])
print(f'videos {n} with boxes {len(vids)}: frac_static >= 0.95: {np.sum(fs >= 0.95)}, >= 0.8: {np.sum(fs >= 0.8)}, '
      f'any moving track (x-extent > 1 box width): {sum(r["moving_tracks"] > 0 for r in vids)}')
for s in ('train', 'validation', 'test'):
    v = [r for r in vids if r['split'] == s]; print(f'  {s}: {len(v)} videos, {sum(r["moving_tracks"] > 0 for r in v)} with a moving track')
ex = [r for r in vids if r['frac_static'] >= 0.99 and r['moving_tracks'] == 0]
print('examples, >= 99% sitting/standing and no moving track:', [(r['video'], r['split'], r['labels']) for r in ex[:6]])
