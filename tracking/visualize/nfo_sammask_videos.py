"""Render each NFO sequence with its SAM2 pseudo-GT masks (*_sammask.png) overlaid, as .mp4.

    python -m tracking.visualize.nfo_sammask_videos [--seqs seq1_gt ...] [--fps 25] [--scale 2]

Every frame is rendered, not just labelled ones: green = sammask, orange = *_sammask_ext
(edge extension, NOT ground truth), magenta = *_sammask_extfix (an extension frame with occluder
pixels removed by gen_data/nfo_pseudo_masks/fix_ext_occluder.py; preferred over its _ext),
red = GT box, and frames with
no mask are tagged "no mask" so gaps between person passes stay visible instead of being cut.
Masks come from gen_data/nfo_pseudo_masks/gen_nfo_pseudo_masks.py (multi-checkpoint SAM2).
"""
import argparse
import os

import cv2
import numpy as np

from utils.bb_utils import parse_bbs

ROOT = 'data/nfo_processed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seqs', nargs='*', default=sorted(d for d in os.listdir(ROOT) if d.endswith('_gt')))
    p.add_argument('--fps', type=float, default=25.0)
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--out-dir', default='images/stream/nfo_gt_masks')
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for seq in a.seqs:
        d = os.path.join(ROOT, seq)
        bbs = parse_bbs(os.path.join(d, 'groundtruth.txt'))
        n = len([f for f in os.listdir(d) if f.endswith('_or.jpg')])
        out = os.path.join(a.out_dir, f'{seq}_sammask.mp4')
        writer, n_mask = None, 0
        for i in range(n):
            img = cv2.imread(os.path.join(d, f'{i:05d}_or.jpg'))
            h, w = img.shape[:2]
            mp = os.path.join(d, f'{i:05d}_sammask.png')
            ext = os.path.join(d, f'{i:05d}_sammask_ext.png')
            extfix = os.path.join(d, f'{i:05d}_sammask_extfix.png')
            has = os.path.exists(mp)
            colour, tag = np.array([0, 255, 0]), ''
            if not has and os.path.exists(extfix):
                mp, has, colour, tag = extfix, True, np.array([255, 0, 255]), '  fixed'
            elif not has and os.path.exists(ext):
                mp, has, colour = ext, True, np.array([0, 165, 255])
            if has:
                m = cv2.imread(mp, 0)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                sel = m > 127
                img[sel] = (0.45 * img[sel] + 0.55 * colour).astype(np.uint8)
                n_mask += 1
            vis = cv2.resize(img, (w * a.scale, h * a.scale), interpolation=cv2.INTER_NEAREST)
            for bb in bbs.get(i, []):
                if bb.x >= 0:
                    s = a.scale
                    cv2.rectangle(vis, (int(bb.x * w * s), int(bb.y * h * s)),
                                  (int((bb.x + bb.w) * w * s), int((bb.y + bb.h) * h * s)), (0, 0, 255), 1)
            cv2.putText(vis, f'{seq} f{i}' + (tag if has else '  no mask'), (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if writer is None:
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), a.fps,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        writer.release()
        print(f'{seq}: {n} frames, {n_mask} with mask -> {out}')


if __name__ == '__main__':
    main()
