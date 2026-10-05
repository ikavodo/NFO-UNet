"""Diagnostic: how much of each pseudo-mask lies on the static occluder layer?

    python -m gen_data.nfo_pseudo_masks.check_occluder_overlap

The camera is static, so occluders (trunks, branches, bushes) are part of the empty-scene
background. Occluder layer O = pixels of the median person-free background that are darker than
its Otsu threshold - the same dark-is-occluder convention find_clear_regions uses per column
(nfo_visibility.py:74), here applied per pixel. A modal person mask should mostly avoid O, so

    occ_frac = |M & O| / |M|

is a mask-error proxy. It is not zero for a correct mask: dark clothing, thin branches blurred
at 224, and the ground under the feet all land in O. That is why the *_sammask.png
distribution is the control the *_sammask_ext.png frames are judged against, not zero.

Second column, bg_agree = fraction of M where |frame - background| < BG_TOL: mask pixels that
look like the empty scene (an occluder in front OR wall behind), whatever its intensity.
Person pixels that happen to match the background colour also count, so it is an upper bound.

Person-free frames sit more than PERSON_MARGIN frames from every GT segment: annotation stops
while the person is still partly visible (gen_nfo_pseudo_masks.extrapolate_edge_boxes).
"""
import csv
import glob
import os

import cv2
import numpy as np

from gen_data.nfo_pseudo_masks.nfo_segment_utils import find_segments
from utils.bb_utils import parse_bbs

IN_DIR = 'data/nfo_processed'
IMG_DIR = 'images/nfo_occluder_overlap'
CSV_PATH = 'results/nfo_pseudo_masks/occluder_overlap.csv'
SEQS = ['seq1', 'seq2', 'seq3', 'seq4']
PERSON_MARGIN = 40
BG_SAMPLES = 60
BG_TOL = 10
N_WORST = 8


def person_free_frames(bbs, n_frames, margin=PERSON_MARGIN):
    segs = find_segments(bbs)
    return [i for i in range(n_frames)
            if all(i < s - margin or i > e + margin for s, e in segs)]


def background(seq_dir, free):
    sample = free[::max(1, len(free) // BG_SAMPLES)][:BG_SAMPLES]
    return np.median(np.stack([cv2.imread(os.path.join(seq_dir, f'{i:05d}_or.jpg'), 0)
                               for i in sample]), axis=0).astype(np.uint8)


def occluder_layer(bg):
    thresh, _ = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bg < thresh


def score(mask, frame, bg, occ):
    n = mask.sum()
    if n == 0:
        return float('nan'), float('nan'), 0
    agree = np.abs(frame.astype(np.int16) - bg.astype(np.int16)) < BG_TOL
    return (mask & occ).sum() / n, (mask & agree).sum() / n, int(n)


def overlay(frame, mask, bad):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    vis[mask & ~bad] = (0.5 * vis[mask & ~bad] + (0, 0, 127)).astype(np.uint8)   # red: mask off O
    vis[mask & bad] = (0, 255, 255)                                                # yellow: counted against M
    return vis


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    rows = []
    for seq in SEQS:
        seq_dir = os.path.join(IN_DIR, f'{seq}_gt')
        bbs = parse_bbs(os.path.join(seq_dir, 'groundtruth.txt'))
        n_frames = max(bbs) + 1
        free = person_free_frames(bbs, n_frames)
        bg = background(seq_dir, free)
        occ = occluder_layer(bg)
        cv2.imwrite(os.path.join(IMG_DIR, f'{seq}_occluder_layer.png'),
                    np.hstack([bg, (occ * 255).astype(np.uint8)]))
        print(f'{seq}: {len(free)} person-free frames, occluder layer = {occ.mean():.1%} of frame')

        for kind, tag in (('main', 'sammask'), ('ext', 'sammask_ext')):
            for p in sorted(glob.glob(os.path.join(seq_dir, f'*_{tag}.png'))):
                idx = int(os.path.basename(p)[:5])
                mask = cv2.imread(p, 0) > 127
                frame = cv2.imread(os.path.join(seq_dir, f'{idx:05d}_or.jpg'), 0)
                of, ba, n = score(mask, frame, bg, occ)
                rows.append(dict(seq=seq, kind=kind, idx=idx, n_px=n, occ_frac=of, bg_agree=ba))

        # worst ext frames per metric, for the user to judge. yellow = the pixels the metric
        # counts against the mask (on O, or matching the background), red = the rest of the mask
        for key in ('occ_frac', 'bg_agree'):
            ext = sorted([r for r in rows if r['seq'] == seq and r['kind'] == 'ext' and r['n_px']],
                         key=lambda r: -r[key])[:N_WORST]
            tiles = []
            for r in ext:
                frame = cv2.imread(os.path.join(seq_dir, f'{r["idx"]:05d}_or.jpg'), 0)
                mask = cv2.imread(os.path.join(seq_dir, f'{r["idx"]:05d}_sammask_ext.png'), 0) > 127
                bad = occ if key == 'occ_frac' else \
                    np.abs(frame.astype(np.int16) - bg.astype(np.int16)) < BG_TOL
                t = overlay(frame, mask, bad)
                cv2.putText(t, f'f{r["idx"]} {r[key]:.2f}', (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1)
                tiles.append(t)
            if tiles:
                cv2.imwrite(os.path.join(IMG_DIR, f'{seq}_worst_ext_{key}.png'), np.hstack(tiles))

    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f'\n{"seq":5} {"kind":5} {"n":>5} {"empty":>5} | occ_frac p50 p90 p99 max | '
          f'bg_agree p50 p90 | ext > main p99 (occ / bg)')
    for seq in SEQS:
        main_of = np.array([r['occ_frac'] for r in rows if r['seq'] == seq and r['kind'] == 'main'
                            and r['n_px']])
        main_ba = np.array([r['bg_agree'] for r in rows if r['seq'] == seq and r['kind'] == 'main'
                            and r['n_px']])
        p99, p99b = np.percentile(main_of, 99), np.percentile(main_ba, 99)
        for kind in ('main', 'ext'):
            rs = [r for r in rows if r['seq'] == seq and r['kind'] == kind]
            of = np.array([r['occ_frac'] for r in rs if r['n_px']])
            ba = np.array([r['bg_agree'] for r in rs if r['n_px']])
            n_empty = sum(1 for r in rs if not r['n_px'])
            q = np.percentile(of, [50, 90, 99]) if len(of) else [np.nan] * 3
            qb = np.percentile(ba, [50, 90]) if len(ba) else [np.nan] * 2
            over = f'{int((of > p99).sum())} / {int((ba > p99b).sum())}' if kind == 'ext' else ''
            print(f'{seq:5} {kind:5} {len(rs):5} {n_empty:5} | {q[0]:.2f} {q[1]:.2f} {q[2]:.2f} '
                  f'{of.max() if len(of) else np.nan:.2f} | {qb[0]:.2f} {qb[1]:.2f} | {over}')
    print(f'\nwrote {CSV_PATH} and {IMG_DIR}/')


if __name__ == '__main__':
    main()
