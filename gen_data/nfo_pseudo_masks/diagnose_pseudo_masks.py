"""Flag problematic segments from gen_nfo_pseudo_masks.py's diagnostics CSVs.

A segment-level low-IoU rate near 100% (not just the usual 30-55% partial disagreement seen in
healthy segments) reliably indicates a real failure - confirmed visually on seq1 segments 1 and 5
(near-total mask collapse under heavy occlusion, or a confidently-wrong blob locked onto
background). This just aggregates the existing diagnostics CSVs per segment and flags outliers;
it does not touch the masks themselves.

Usage:
    python3 -m gen_data.nfo_pseudo_masks.diagnose_pseudo_masks --diagnostics-dir out
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

LOW_IOU_THRESHOLD = 0.3
PROBLEMATIC_RATE = 0.7  # segment-level low-IoU rate at/above this is flagged


def load_segment_stats(csv_path):
    """Returns {segment_idx: {'n_frames': int, 'n_unlabeled': int, 'n_scored': int,
    'n_low_iou': int, 'raw_range': (min, max)}}."""
    stats = defaultdict(lambda: {'n_frames': 0, 'n_unlabeled': 0, 'n_scored': 0,
                                  'n_low_iou': 0, 'raw_min': None, 'raw_max': None})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seg = int(row['segment_idx'])
            s = stats[seg]
            s['n_frames'] += 1
            raw_idx = int(row['raw_idx'])
            s['raw_min'] = raw_idx if s['raw_min'] is None else min(s['raw_min'], raw_idx)
            s['raw_max'] = raw_idx if s['raw_max'] is None else max(s['raw_max'], raw_idx)
            n_reached = int(row['n_checkpoints_reached'])
            if n_reached == 0:
                s['n_unlabeled'] += 1
                continue
            iou_str = row['min_pairwise_iou']
            if iou_str and iou_str != 'nan':  # nan = single-checkpoint frame, no IoU to compute
                iou_val = float(iou_str)
                s['n_scored'] += 1
                if iou_val < LOW_IOU_THRESHOLD:
                    s['n_low_iou'] += 1
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics-dir', default='.')
    parser.add_argument('--low-iou-threshold', type=float, default=LOW_IOU_THRESHOLD)
    parser.add_argument('--problematic-rate', type=float, default=PROBLEMATIC_RATE)
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.diagnostics_dir, '*_pseudo_mask_diagnostics.csv')))
    if not csv_paths:
        raise RuntimeError(f'no *_pseudo_mask_diagnostics.csv found in {args.diagnostics_dir}')

    flagged = []
    for path in csv_paths:
        seq = os.path.basename(path).split('_pseudo_mask_diagnostics.csv')[0]
        stats = load_segment_stats(path)
        print(f'\n{seq}:')
        for seg in sorted(stats):
            s = stats[seg]
            unlabeled_rate = s['n_unlabeled'] / s['n_frames']
            low_iou_rate = s['n_low_iou'] / s['n_scored'] if s['n_scored'] else 0.0
            is_problematic = (low_iou_rate >= args.problematic_rate) or (unlabeled_rate >= 0.1)
            flag = ' <-- PROBLEMATIC' if is_problematic else ''
            print(f'  segment {seg} (raw {s["raw_min"]}-{s["raw_max"]}, {s["n_frames"]} frames): '
                  f'unlabeled={100*unlabeled_rate:.0f}% low_iou={100*low_iou_rate:.0f}% '
                  f'(n_scored={s["n_scored"]}){flag}')
            if is_problematic:
                flagged.append((seq, seg, s['raw_min'], s['raw_max']))

    print(f'\n{len(flagged)} problematic segment(s) flagged:')
    for seq, seg, raw_min, raw_max in flagged:
        print(f'  {seq} segment {seg} (raw {raw_min}-{raw_max})')


if __name__ == '__main__':
    main()
