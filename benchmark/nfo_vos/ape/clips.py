"""One training clip per source video: the ape's most-moving 40-frame window (simplified 2026-11-13, user: no sliding
windows; the integral image needs motion, so only moving apes are used).

Ape sources: PanAf500 videos labelled clear (PanAf20K labels/consensus.csv) in the train split, not in the eval subset,
not leaked (results/ape/leakage.csv). The 80 PanAf-SBR train-only videos join once their sbr100 labels exist
(PanAf20K labels/sbr100_occlusion_*.csv, clear only). Boxes and silhouettes: PanAf-SBR annotations (its boxes equal the
PanAf500 boxes; masks = SBR RLE). Per ape track, windows of N = 40 frames every STRIDE = 8 frames, kept if:
  - >= 36 frames have a box AND a non-empty mask (the ape is present),
  - no other ape's box overlaps R by more than 5% of R in any window frame (other apes would be unlabelled
    background inside the planted region), R = union of the window's boxes + 10% margin, clipped to the frame.
Recorded per window (no filtering on them): motion = centre path length / median box width, median box area, the
share of frames whose box touches the frame border, R. Training samples a video uniformly, then a window, so long
videos do not dominate.
    python -m benchmark.nfo_vos.ape.clips   -> results/ape/clip_index.csv
"""
import json
from collections import defaultdict
import numpy as np, pandas as pd
from pycocotools import mask as M
from benchmark.nfo_vos.ape import PANAF, SBR_ROOT, OUT

N, STRIDE, MIN_PRESENT, MARGIN, MAX_OTHER = 40, 4, 36, 0.10, 0.05
OK = ('clear', 'partial')   # partial apes get extra planted occlusion on top; occluded videos are all in the eval subset
MIN_MOTION = 0.5            # centre path over the 40 frames, in median box widths


def sources():
    lab = pd.read_csv(PANAF / 'labels' / 'consensus.csv').set_index('video').occlusion
    leak = pd.read_csv(OUT / 'leakage.csv').set_index('video')
    ann = {p.stem: p for p in SBR_ROOT.glob('annotations*/train/*.json')}
    v = [x for x in lab[lab.isin(OK)].index if x in ann and not leak.in_eval.get(x, False) and not leak.leaked.get(x, False)]
    extra = sorted(PANAF.glob('labels/sbr100_occlusion_*.csv'))
    if extra:
        s = pd.read_csv(extra[0]); v += [x for x in s[s.occlusion.isin(OK)].video if x in ann and x not in lab.index
                                         and not leak.leaked.get(x, False)]
    return v, ann, len(extra) > 0


def windows(video, path):
    js = json.load(open(path)); H, W = None, None
    tr = defaultdict(dict)
    for a in js['annotations']:
        for d in a['detections']:
            s = d['segmentation']
            if not s:   # some SBR detections carry no mask: treat as absent
                continue
            H, W = s['size']
            area = M.area({'size': s['size'], 'counts': s['counts'].encode() if isinstance(s['counts'], str) else s['counts']})
            tr[str(d['ape_id'])][a['frame_id']] = (np.array(d['bbox'], float), area)
    out = []
    for aid, t in tr.items():
        fr = sorted(t)
        for s0 in range(fr[0], fr[-1] - N + 2, STRIDE):
            ks = [k for k in range(s0, s0 + N) if k in t and t[k][1] > 0]
            if len(ks) < MIN_PRESENT:
                continue
            B = np.stack([t[k][0] for k in ks]); bw = np.median(B[:, 2] - B[:, 0]); bh = np.median(B[:, 3] - B[:, 1])
            x1, y1, x2, y2 = B[:, 0].min() - MARGIN * bw, B[:, 1].min() - MARGIN * bh, B[:, 2].max() + MARGIN * bw, B[:, 3].max() + MARGIN * bh
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H); RA = max((x2 - x1) * (y2 - y1), 1)
            other = 0.0
            for o, to in tr.items():
                if o == aid:
                    continue
                for k in range(s0, s0 + N):
                    if k in to:
                        b = to[k][0]; ix = max(0, min(x2, b[2]) - max(x1, b[0])); iy = max(0, min(y2, b[3]) - max(y1, b[1]))
                        other = max(other, ix * iy / RA)
            c = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
            out.append(dict(video=video, ape_id=aid, start=s0, n_present=len(ks), other_overlap=other,
                            motion=float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum() / max(bw, 1)),
                            box_area=float(bw * bh), border=float(((B[:, 0] <= 1) | (B[:, 1] <= 1) | (B[:, 2] >= W - 1) | (B[:, 3] >= H - 1)).mean()),
                            R=[round(float(v), 1) for v in (x1, y1, x2, y2)], H=H, W=W))
    return out


def main():
    v, ann, has_sbr = sources()
    T = pd.DataFrame([w for x in v for w in windows(x, ann[x])])
    T = T[T.other_overlap <= MAX_OTHER]
    best = T.loc[T.groupby('video').motion.idxmax()]                       # one clip per video: its most-moving window
    lab = pd.read_csv(PANAF / 'labels' / 'consensus.csv').set_index('video').occlusion
    for f in sorted(PANAF.glob('labels/sbr100_occlusion_*.csv')):
        lab = pd.concat([lab, pd.read_csv(f).set_index('video').occlusion]); lab = lab[~lab.index.duplicated()]
    best['occlusion'] = best.video.map(lab); best['keep'] = best.motion >= MIN_MOTION
    best.to_csv(OUT / 'clip_index.csv', index=False)
    print(f'source videos (clear+partial, train, non-eval, non-leaked): {len(v)} (sbr100 labels {"used" if has_sbr else "missing"}); '
          f'with a valid window: {len(best)}')
    for m in (0.3, 0.5, 1.0):
        k = best[best.motion >= m]; print(f'  motion >= {m}: {len(k)} clips ({dict(k.occlusion.value_counts())})')
    print(f'kept (>= {MIN_MOTION}): {int(best.keep.sum())} -> results/ape/clip_index.csv')


if __name__ == '__main__':
    main()
