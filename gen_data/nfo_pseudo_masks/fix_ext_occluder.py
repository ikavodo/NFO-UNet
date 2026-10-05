"""Remove occluder pixels from *_sammask_ext.png masks: keep a mask pixel only if most of its
neighbourhood differs from the empty-scene background.

    python -m gen_data.nfo_pseudo_masks.fix_ext_occluder            # dry run: stats + montages
    python -m gen_data.nfo_pseudo_masks.fix_ext_occluder --apply    # write *_sammask_extfix.png, FLAGGED only

Failure it targets (check_occluder_overlap.py, eyeballed): extension masks that cover the person
AND attached occluder/background - mostly one connected component, so dropping whole components
cannot separate them. Per pixel instead:
    C  = M & (|I_t - B| >= BG_TOL)                changed (non-background) mask pixels
    rho(x) = box_k(C)(x) / box_k(M)(x)            local changed fraction, within the mask only
    M' = M & (rho >= RHO), minus components < MIN_AREA px
The box average, not the raw C, keeps camouflaged person pixels (clothing that matches the
background) whose neighbours changed; occluder regions are static scene, so rho stays low there.

Applied to the user-flagged frames ONLY. As a global filter it is not safe: on the main masks
(control) median IoU(before, after) is 0.98/1.00/0.91/1.00 for seq1-4, but p10 is 0.79 (seq1)
and 0.58 (seq3, judged fine by eye) - it also cuts camouflaged parts of masks that are fine.
"""
import argparse
import os

import cv2
import numpy as np

from gen_data.nfo_pseudo_masks.check_occluder_overlap import BG_TOL, IN_DIR, background, person_free_frames
from utils.bb_utils import parse_bbs

K, RHO, MIN_AREA = 7, 0.5, 10
OUT_DIR = 'images/nfo_occluder_overlap'
FLAGGED = {'seq1': list(range(6, 11)) + list(range(1805, 1815)), 'seq2': list(range(1087, 1093)),
           'seq3': [], 'seq4': list(range(1331, 1337))}   # eyeballed by the user, 2026-10-05


def fix(mask, frame, bg, k=K, rho=RHO, min_area=MIN_AREA):
    changed = (mask & (np.abs(frame.astype(np.int16) - bg.astype(np.int16)) >= BG_TOL)).astype(np.float32)
    num = cv2.boxFilter(changed, -1, (k, k), normalize=False)
    den = cv2.boxFilter(mask.astype(np.float32), -1, (k, k), normalize=False)
    keep = mask & (num >= rho * np.maximum(den, 1))
    n, lab, st, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            keep[lab == i] = False
    return keep


# Static occluder patches the pixel test cannot remove (they only half-match the median
# background, e.g. foliage that moves in wind). The camera is static, so the patch is cut by
# LOCATION: region R = union of the non-person components in the frames where they are separate
# (seq, frames to cut, frames that define R, column x: components centred left of x are patch).
REGION_CUTS = [('seq4', range(1022, 1031), [1022, 1023, 1027, 1028, 1029], 215)]   # user, 2026-10-05


def patch_region(d, defining, x_split):
    R = None
    for idx in defining:
        m = cv2.imread(os.path.join(d, f'{idx:05d}_sammask_ext.png'), 0) > 127
        n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        for k in range(1, n):
            if cen[k, 0] < x_split:
                R = (lab == k) if R is None else (R | (lab == k))
    return cv2.dilate(R.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def apply_region_cuts(write=True):
    """Returns {seq: [(frame, old, new, label)]} for the REGION_CUTS frames; writes *_extfix."""
    out = {}
    for seq, frames, defining, x_split in REGION_CUTS:
        d = os.path.join(IN_DIR, f'{seq}_gt')
        R = patch_region(d, defining, x_split)
        for idx in frames:
            m = cv2.imread(os.path.join(d, f'{idx:05d}_sammask_ext.png'), 0) > 127
            m2 = m & ~R
            if write:
                cv2.imwrite(os.path.join(d, f'{idx:05d}_sammask_extfix.png'), (m2 * 255).astype(np.uint8))
            I = cv2.imread(os.path.join(d, f'{idx:05d}_or.jpg'), 0)
            out.setdefault(seq, []).append((I, m, m2, f'{seq} f{idx}'))
    return out


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def tile(frame, before, after, label):
    v = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    removed = before & ~after
    v[after] = (0.5 * v[after] + (0, 127, 0)).astype(np.uint8)      # green: kept
    v[removed] = (0, 0, 255)                                        # red: removed
    cv2.putText(v, label, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return v


def write_video(path, pairs, scale=3, fps=3):
    """pairs: [(frame, old_mask, new_mask, idx)] -> side-by-side mp4, old (left) | fixed (right)."""
    h, w = pairs[0][0].shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (2 * w * scale, h * scale))
    for frame, old, new, idx in pairs:
        sides = []
        for m, name in ((old, 'old'), (new, 'fixed')):
            v = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            v[m] = (0.5 * v[m] + (0, 0, 127)).astype(np.uint8)
            v = cv2.resize(v, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            cv2.putText(v, f'{idx} {name} {m.sum()}px', (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)
            sides.append(v)
        vw.write(np.hstack(sides))
    vw.release()


def review_unflagged(n_top=20):
    """Old-vs-fixed video of the n_top UNFLAGGED ext frames (all sequences pooled) where the fix
    removes the largest share of mask area. Writes nothing to data/; the list goes to a CSV."""
    cands = []
    for seq, flagged in FLAGGED.items():
        d = os.path.join(IN_DIR, f'{seq}_gt')
        bbs = parse_bbs(os.path.join(d, 'groundtruth.txt'))
        bg = background(d, person_free_frames(bbs, max(bbs) + 1))
        for f in sorted(x for x in os.listdir(d) if x.endswith('_sammask_ext.png')):
            idx = int(f[:5])
            m = cv2.imread(os.path.join(d, f), 0) > 127
            if idx in flagged or not m.any():
                continue
            I = cv2.imread(os.path.join(d, f'{idx:05d}_or.jpg'), 0)
            m2 = fix(m, I, bg)
            cands.append((1 - m2.sum() / m.sum(), seq, idx, I, m, m2))
    cands.sort(key=lambda c: -c[0])
    top = cands[:n_top]
    with open(os.path.join(OUT_DIR, 'review_unflagged_top.csv'), 'w') as fh:
        fh.write('seq,idx,area_removed_frac\n')
        fh.writelines(f'{seq},{idx},{lost:.3f}\n' for lost, seq, idx, *_ in top)
    write_video(os.path.join(OUT_DIR, 'review_unflagged_top.mp4'),
                [(I, m, m2, f'{seq} {idx} -{lost:.0%}') for lost, seq, idx, I, m, m2 in top], fps=2)
    for lost, seq, idx, *_ in top:
        print(f'{seq} f{idx}: {lost:.0%} of area removed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--review-unflagged', type=int, default=0, metavar='N',
                    help='only write an old-vs-fixed video of the N unflagged ext frames the fix '
                         'shrinks most, then exit')
    ap.add_argument('--region-cuts', action='store_true',
                    help='only apply REGION_CUTS (writes *_extfix) + an old-vs-fixed video, then exit')
    args = ap.parse_args()
    if args.region_cuts:
        for seq, pairs in apply_region_cuts().items():
            write_video(os.path.join(OUT_DIR, f'{seq}_region_cut.mp4'), pairs)
            print(seq, [(lab, int(o.sum()), int(n.sum())) for _, o, n, lab in pairs])
        return
    if args.review_unflagged:
        review_unflagged(args.review_unflagged)
        return
    for seq, flagged in FLAGGED.items():
        d = os.path.join(IN_DIR, f'{seq}_gt')
        bbs = parse_bbs(os.path.join(d, 'groundtruth.txt'))
        bg = background(d, person_free_frames(bbs, max(bbs) + 1))
        stats = {'main': [], 'ext_ok': [], 'ext_flagged': []}
        tiles, pairs = [], []
        for tag in ('sammask', 'sammask_ext'):
            for f in sorted(x for x in os.listdir(d) if x.endswith(f'_{tag}.png')):
                idx = int(f[:5])
                m = cv2.imread(os.path.join(d, f), 0) > 127
                if not m.any():
                    continue
                I = cv2.imread(os.path.join(d, f'{idx:05d}_or.jpg'), 0)
                m2 = fix(m, I, bg)
                key = 'main' if tag == 'sammask' else ('ext_flagged' if idx in flagged else 'ext_ok')
                stats[key].append(iou(m, m2))
                if key == 'ext_flagged':
                    tiles.append(tile(I, m, m2, f'f{idx} {m.sum()}->{m2.sum()}'))
                    pairs.append((I, m, m2, f'f{idx}'))
                if args.apply and key == 'ext_flagged':
                    cv2.imwrite(os.path.join(d, f'{idx:05d}_sammask_extfix.png'), (m2 * 255).astype(np.uint8))
        print(seq, '  '.join(f'{k}: n={len(v)} IoU(before,after) p10={np.percentile(v, 10):.2f} '
                             f'p50={np.median(v):.2f}' for k, v in stats.items() if v))
        if tiles:
            rows = [np.hstack(tiles[i:i + 8] + [np.zeros_like(tiles[0])] * (8 - len(tiles[i:i + 8])))
                    for i in range(0, len(tiles), 8)]
            cv2.imwrite(os.path.join(OUT_DIR, f'{seq}_fix_flagged.png'), np.vstack(rows))
            write_video(os.path.join(OUT_DIR, f'{seq}_fix_flagged.mp4'), pairs)


if __name__ == '__main__':
    main()
