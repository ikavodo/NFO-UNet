"""Pilot trials (spec §2-§3): seq1 segments 4-5, starts every 10 frames, 51-frame windows,
admissibility D(p*) >= 2, shared box + p* prompt, covariates v(t0) and n_frag(t0), warm-up range.

    python -m benchmark.nfo_vos.trials      # writes results/benchmark/pilot/trials.json
"""
import json
import os

import cv2
import numpy as np

from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
from gen_data.nfo_pseudo_masks.nfo_segment_utils import find_segments
from gen_data.nfo_pseudo_masks.nfo_visibility import confirmed_clear_frames, default_clear_regions
from utils.bb_utils import parse_bbs

SEQ = 'seq1'
SEQ_DIR = f'data/nfo_processed/{SEQ}_gt'
NATIVE_W, NATIVE_H = 800, 600
SIZE = 224
PILOT_SEGMENTS = (4, 5)          # indices into find_segments (spec §2)
STRIDE, WINDOW = 10, 50          # window covers [t0, t0 + WINDOW] -> 51 frames
MIN_D = 2.0                      # admissibility, 224 px
GAP_MARGIN = 40                  # person-free: > 40 frames from every GT segment
OUT = 'results/benchmark/pilot/trials.json'


def to_native(x224, y224):
    """224 pixel index -> native coords through the padded square (pixel centre at i + 0.5)."""
    s = max(NATIVE_W, NATIVE_H)
    return ((x224 + 0.5) / SIZE * s - (s - NATIVE_W) / 2, (y224 + 0.5) / SIZE * s - (s - NATIVE_H) / 2)


def gt_mask(idx):
    p = os.path.join(SEQ_DIR, f'{idx:05d}_sammask.png')
    return cv2.imread(p, 0) > 127 if os.path.exists(p) else None


_BBS = None


def bbs():
    global _BBS
    if _BBS is None:
        _BBS = parse_bbs(os.path.join(SEQ_DIR, 'groundtruth.txt'))
    return _BBS


def box_area_224(idx):
    bb = bbs()[idx][0]
    return bb.w * SIZE * bb.h * SIZE


def clear_frames(seg):
    b = bbs()
    regions = default_clear_regions(SEQ_DIR, b, max(b.keys()) + 1, SIZE, SIZE)
    return [i for i in confirmed_clear_frames(b, seg['start'], seg['end'], regions, SIZE)
            if gt_mask(i) is not None]


def kappa(seg):
    """Median fill ratio |M| / |amodal GT box| of a fully visible person (confirmed-clear frames)."""
    if 'kappa' not in seg:
        seg['kappa'] = float(np.median([gt_mask(i).sum() / box_area_224(i) for i in clear_frames(seg)]))
    return seg['kappa']


def visibility(idx, seg):
    """v = |M| / (kappa * |box|): visible fraction of the expected full-body area at THIS frame's
    scale. Replaces the rev-2 |M| / median|M|, which mixed apparent size (the walker approaches or
    recedes within a segment) with visibility."""
    return float(gt_mask(idx).sum() / (kappa(seg) * box_area_224(idx)))


def pilot_segments():
    segs = find_segments(bbs())
    out = []
    for i in PILOT_SEGMENTS:
        s, e = segs[i]
        prev_end = segs[i - 1][1] if i > 0 else -GAP_MARGIN - 1
        out.append(dict(idx=i, start=s, end=e, warmup=(prev_end + GAP_MARGIN + 1, s - GAP_MARGIN - 1)))
    return out


def build_trials():
    trials = []
    for seg in pilot_segments():
        s, e = seg['start'], seg['end']
        last_gt = max(i for i in range(s, e + 1) if gt_mask(i) is not None)
        n_clear = len(clear_frames(seg))
        for t0 in range(s, last_gt - WINDOW + 1, STRIDE):
            M = gt_mask(t0)
            D = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5)
            py, px = np.unravel_index(np.argmax(D), D.shape)
            bb = bbs()[t0][0]
            n_cc = cv2.connectedComponents(M.astype(np.uint8), connectivity=8)[0] - 1
            trials.append(dict(
                id=f'{SEQ}_s{seg["idx"]}_t{t0}', seq=SEQ, seg_idx=seg['idx'], seg_start=s, seg_end=e,
                t0=t0, frames=list(range(t0, t0 + WINDOW + 1)), warmup=list(seg['warmup']),
                box_native=[float(v) for v in gt_to_native(bb, NATIVE_W, NATIVE_H)],
                box_224=[bb.x * SIZE, bb.y * SIZE, (bb.x + bb.w) * SIZE, (bb.y + bb.h) * SIZE],
                point_224=[float(px), float(py)], point_native=[float(v) for v in to_native(px, py)],
                D=float(D[py, px]), admissible=bool(D[py, px] >= MIN_D),
                v=visibility(t0, seg), n_frag=int(n_cc), kappa=kappa(seg), n_clear=n_clear))
    return trials


def main():
    trials = build_trials()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(trials, f, indent=1)
    for t in trials:
        print(f"{t['id']}: v={t['v']:.2f} n_frag={t['n_frag']} D={t['D']:.1f} "
              f"{'ok' if t['admissible'] else 'INADMISSIBLE'}")
    print(f'{sum(t["admissible"] for t in trials)}/{len(trials)} admissible -> {OUT}')


if __name__ == '__main__':
    main()
