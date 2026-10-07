"""Round-robin trial schedule (spec §3, rev. 5).

Every frame is used at most once. Within a sequence all 8 segments share one occluder field:
visibility v at the same position x correlates r ≈ 0.65–0.73 across segments of either direction
and gait (measured 2026-10-07; null r ≈ 0). So start positions are spread ACROSS segments instead
of overlapping windows on one segment:
  - each sequence has exactly 2 segments per (direction, gait). The earlier one is in the COVER
    half, the later one in the BACKUP half;
  - each (direction, gait) group gets a quarter-phase φ ∈ {0, W/4, W/2, 3W/4}, the same for both
    twins, so the backup replicates the cover's start conditions;
  - segment windows are back to back: frames φ + mW … φ + (m+1)W − 1, m = 0 … ⌊(L − φ)/W⌋ − 1.
W = 40 is the largest W with 3W/4 + W ≤ (4/7)·L_min(74), i.e. every segment keeps ≥ 1 window.

    NFO_RUN=rr python -m benchmark.nfo_vos.trials --seqs seq1            # pilot
    NFO_RUN=rr python -m benchmark.nfo_vos.trials --seqs seq1 seq2 seq3 seq4   # full run
Writes results/benchmark/$NFO_RUN/trials.json. The old overlapping-window pilot lives in
results/benchmark/pilot (its trials.json is kept; this module no longer generates it).
"""
import argparse
import functools
import json
import os

import cv2
import numpy as np

from gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks import gt_to_native
from gen_data.nfo_pseudo_masks.nfo_segment_utils import find_segments
from gen_data.nfo_pseudo_masks.nfo_visibility import confirmed_clear_frames, default_clear_regions
from utils.bb_utils import parse_bbs

RUN = os.environ.get('NFO_RUN', 'rr')
RES = f'results/benchmark/{RUN}'
IMG = f'images/benchmark/{RUN}'
OUT = f'{RES}/trials.json'
ALL_SEQS = ('seq1', 'seq2', 'seq3', 'seq4')
SEQ = 'seq1'                     # default for single-sequence callers
SEQ_DIR = f'data/nfo_processed/{SEQ}_gt'
NATIVE_W, NATIVE_H = 800, 600
SIZE = 224
W = 40
PHASE = {('R', 'walk'): 0, ('L', 'walk'): 1, ('R', 'run'): 2, ('L', 'run'): 3}   # x W/4
MIN_D = 2.0                      # admissibility, 224 px
GAP_MARGIN = 40                  # person-free: > 40 frames from every GT segment
MIN_WARMUP = 20
KAPPA_FALLBACK = 0.394           # median fill ratio when a segment has no confirmed-clear frame


def seq_dir(seq):
    return f'data/nfo_processed/{seq}_gt'


def to_native(x224, y224):
    """224 pixel index -> native coords through the padded square (pixel centre at i + 0.5)."""
    s = max(NATIVE_W, NATIVE_H)
    return ((x224 + 0.5) / SIZE * s - (s - NATIVE_W) / 2, (y224 + 0.5) / SIZE * s - (s - NATIVE_H) / 2)


def gt_mask(idx, seq=SEQ):
    p = os.path.join(seq_dir(seq), f'{idx:05d}_sammask.png')
    return cv2.imread(p, 0) > 127 if os.path.exists(p) else None


@functools.lru_cache(maxsize=None)
def bbs(seq=SEQ):
    return parse_bbs(os.path.join(seq_dir(seq), 'groundtruth.txt'))


def box_area_224(idx, seq=SEQ):
    bb = bbs(seq)[idx][0]
    return bb.w * SIZE * bb.h * SIZE


def clear_frames(seg):
    b = bbs(seg['seq'])
    regions = default_clear_regions(seq_dir(seg['seq']), b, max(b.keys()) + 1, SIZE, SIZE)
    return [i for i in confirmed_clear_frames(b, seg['start'], seg['end'], regions, SIZE)
            if gt_mask(i, seg['seq']) is not None]


def kappa(seg):
    """Median fill ratio |M| / |amodal GT box| of a fully visible person (confirmed-clear frames)."""
    if 'kappa' not in seg:
        cf = clear_frames(seg)
        seg['kappa'] = (float(np.median([gt_mask(i, seg['seq']).sum() / box_area_224(i, seg['seq']) for i in cf]))
                        if cf else KAPPA_FALLBACK)
    return seg['kappa']


def visibility(idx, seg):
    """v = |M| / (kappa * |box|): visible fraction of the expected full-body area at this frame's scale."""
    return float(gt_mask(idx, seg['seq']).sum() / (kappa(seg) * box_area_224(idx, seg['seq'])))


def _raw_segments(seq):
    b = bbs(seq)
    out = []
    for i, (s, e) in enumerate(find_segments(b)):
        L = e - s + 1
        xs = np.array([(b[k][0].x + b[k][0].w / 2) * SIZE for k in range(s, e + 1)])
        hs = np.array([b[k][0].h * SIZE for k in range(s, e + 1)])
        out.append(dict(seq=seq, idx=i, start=s, end=e, L=L, dir='R' if xs[-1] > xs[0] else 'L',
                        speed_h=float(abs(np.polyfit(np.arange(L), xs, 1)[0]) / np.median(hs))))
    return out


@functools.lru_cache(maxsize=None)
def gait_threshold():
    """2-means on speed in body heights per frame over all 32 segments; midpoint of the centres.
    Measured centres: walk 0.027, run 0.043."""
    v = np.array([s['speed_h'] for q in ALL_SEQS for s in _raw_segments(q)])
    c = np.array([v.min(), v.max()])
    for _ in range(50):
        lab = np.abs(v[:, None] - c[None]).argmin(1)
        c = np.array([v[lab == j].mean() for j in (0, 1)])
    return float(c.mean())


def _warmup(segs, i):
    """Person-free frames for the background prior (tracker-based methods only). Preferred: the
    gap right before segment i (causal). If that is shorter than MIN_WARMUP (or absent: the first
    segment of every sequence starts at frame 11-32), use the sequence's longest person-free gap.
    That is scene calibration, as a deployed static camera has seen its empty scene before.
    Returns ((lo, hi), source)."""
    gaps = []
    for j in range(len(segs) + 1):
        lo = (segs[j - 1]['end'] if j > 0 else -GAP_MARGIN - 1) + GAP_MARGIN + 1
        hi = (segs[j]['start'] if j < len(segs) else max(bbs(segs[0]['seq']).keys()) + 1) - GAP_MARGIN - 1
        gaps.append((lo, hi))
    own = gaps[i]
    if own[1] - own[0] + 1 >= MIN_WARMUP:
        return own, 'preceding'
    best = max(gaps, key=lambda g: g[1] - g[0])
    assert best[1] - best[0] + 1 >= MIN_WARMUP, f"no person-free gap of >= {MIN_WARMUP} frames in {segs[0]['seq']}"
    return best, 'scene'


def segments(seq):
    segs = _raw_segments(seq)
    thr = gait_threshold()
    for s in segs:
        s['gait'] = 'run' if s['speed_h'] > thr else 'walk'
    for i, s in enumerate(segs):
        s['warmup'], s['warmup_source'] = _warmup(segs, i)
    for dg in PHASE:
        grp = sorted((s for s in segs if (s['dir'], s['gait']) == dg), key=lambda s: s['start'])
        assert len(grp) == 2, f'{seq}: expected 2 segments for {dg}, found {len(grp)}'
        for half, s in zip(('cover', 'backup'), grp):
            s['half'], s['phase'] = half, PHASE[dg] * W // 4
    return segs


def build_trials(seqs=('seq1',)):
    trials = []
    for seq in seqs:
        for seg in segments(seq):
            for m in range((seg['L'] - seg['phase']) // W):
                t0 = seg['start'] + seg['phase'] + m * W
                M = gt_mask(t0, seq)
                if M is None or not M.any():
                    trials.append(dict(id=f"{seq}_s{seg['idx']}_t{t0}", seq=seq, t0=t0, admissible=False,
                                       reason='no GT mask at t0'))
                    continue
                D = cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5)
                py, px = np.unravel_index(np.argmax(D), D.shape)
                bb = bbs(seq)[t0][0]
                trials.append(dict(
                    id=f"{seq}_s{seg['idx']}_t{t0}", seq=seq, seg_idx=seg['idx'], seg_start=seg['start'],
                    seg_end=seg['end'], dir=seg['dir'], gait=seg['gait'], half=seg['half'],
                    phase=seg['phase'], m=m, t0=t0, frames=list(range(t0, t0 + W)),
                    warmup=list(seg['warmup']), warmup_source=seg['warmup_source'], x0=(bb.x + bb.w / 2) * SIZE,
                    box_native=[float(v) for v in gt_to_native(bb, NATIVE_W, NATIVE_H)],
                    box_224=[bb.x * SIZE, bb.y * SIZE, (bb.x + bb.w) * SIZE, (bb.y + bb.h) * SIZE],
                    point_224=[float(px), float(py)], point_native=[float(v) for v in to_native(px, py)],
                    D=float(D[py, px]), admissible=bool(D[py, px] >= MIN_D),
                    v=visibility(t0, seg), n_frag=int(cv2.connectedComponents(M.astype(np.uint8), connectivity=8)[0] - 1),
                    kappa=kappa(seg)))
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', default=['seq1'])
    a = ap.parse_args()
    trials = build_trials(tuple(a.seqs))
    os.makedirs(RES, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(trials, f, indent=1)
    ok = [t for t in trials if t['admissible']]
    for t in trials:
        print(f"{t['id']}: " + (f"{t['dir']}{t['gait'][0]} {t['half']:6s} v={t['v']:.2f} D={t['D']:.1f}"
                                 if 'v' in t else t.get('reason', '')) + ('' if t['admissible'] else '  INADMISSIBLE'))
    print(f'{len(ok)}/{len(trials)} admissible -> {OUT}')


if __name__ == '__main__':
    main()
