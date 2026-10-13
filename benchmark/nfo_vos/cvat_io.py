"""CVAT round trip for hand masks on seq1 segment 4 (docs/hand_annotation_plan.md). 2026-10-13

    python -m benchmark.nfo_vos.cvat_io export            # -> data/cvat/seq1_seg4/{images/, prefill.xml}
    python -m benchmark.nfo_vos.cvat_io import <xml> [tag]  # CVAT-for-images-1.1 export -> seq1_gt/<idx>_<tag>.png

export: all 155 native frames (annotate only the TARGETS; the rest are for flipping) and a CVAT-for-images 1.1
annotation file pre-filling the TARGETS with the native pseudo-GT as one 'person' mask each.
import: rasterise 'person' (1) and 'void' (2, painted over person) masks and polygons to 0/1/2 PNGs at 800x600
(tag 'handmask' by default, 'handmask_scratch' for the from-scratch task), then print pseudo-GT P/R vs hand.

Mask RLE (CVAT XML <mask rle=... left top width height>): run lengths over the mask's bounding box,
row-major, alternating background/foreground and starting with background (assumed from CVAT's
dataset_manager/formats/cvat.py; verify with one round trip before annotating).
"""
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from benchmark.nfo_vos.run_sam2_video import gt_mask_native

SEQ, A, B = 'seq1', 1154, 1308
TARGETS = list(range(A, B + 1, 5))          # every 5th frame; includes the t0s 1154, 1194, 1234
SCRATCH = [1164, 1214, 1264]                # annotated from scratch first (anchoring check)
OUT = 'data/cvat/seq1_seg4'
FRAMES = f'data/nfo_final/nfo_final/{SEQ}'
GT = f'data/nfo_processed/{SEQ}_gt'


def rle_encode(m):
    ys, xs = np.nonzero(m)
    l, t, r, b = xs.min(), ys.min(), xs.max(), ys.max()
    flat = m[t:b + 1, l:r + 1].ravel().astype(np.int8)
    runs = np.diff(np.r_[0, np.flatnonzero(np.diff(flat)) + 1, flat.size])
    runs = np.r_[0, runs] if flat[0] else runs                       # first run counts background
    return ', '.join(map(str, runs)), l, t, r - l + 1, b - t + 1


def rle_decode(rle, l, t, w, h, shape=(600, 800)):
    runs = np.array([int(v) for v in rle.split(',')])
    flat = np.repeat(np.arange(len(runs)) % 2, runs).astype(bool)
    out = np.zeros(shape, bool)
    out[t:t + h, l:l + w] = flat.reshape(h, w)
    return out


def export():
    os.makedirs(f'{OUT}/images', exist_ok=True)
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = '1.1'
    for i, k in enumerate(range(A, B + 1)):
        shutil.copy(f'{FRAMES}/{k:05d}.jpg', f'{OUT}/images/{k:05d}.jpg')
        img = ET.SubElement(root, 'image', id=str(i), name=f'{k:05d}.jpg', width='800', height='600')
        m = gt_mask_native(SEQ, k) if k in TARGETS else None
        if m is not None and m.any():
            rle, l, t, w, h = rle_encode(m)
            assert (rle_decode(rle, l, t, w, h) == m).all()
            ET.SubElement(img, 'mask', label='person', source='file', occluded='0', z_order='0', rle=rle,
                          left=str(l), top=str(t), width=str(w), height=str(h))
    ET.ElementTree(root).write(f'{OUT}/prefill.xml')
    print(f'{OUT}: {B - A + 1} frames, {len(TARGETS)} prefilled targets, scratch frames {SCRATCH}')


def import_(xml, tag='handmask'):
    for img in ET.parse(xml).getroot().iter('image'):
        k = int(os.path.splitext(img.get('name'))[0])
        shapes = [m for m in img if m.tag in ('mask', 'polygon')]
        if not shapes:
            continue
        lab = np.zeros((600, 800), np.uint8)
        for want, val in (('person', 1), ('void', 2)):
            for m in shapes:
                if m.get('label') != want:
                    continue
                if m.tag == 'mask':
                    lab[rle_decode(m.get('rle'), *(int(float(m.get(a))) for a in ('left', 'top', 'width', 'height')))] = val
                else:   # polygon "x1,y1;x2,y2;..." (also what CVAT writes per frame for interpolated polygon tracks)
                    pts = np.array([[float(v) for v in xy.split(',')] for xy in m.get('points').split(';')])
                    cv2.fillPoly(lab, [np.round(pts).astype(np.int32)], val)
        p, h = gt_mask_native(SEQ, k), lab == 1
        if p is not None and (lab == p.astype(np.uint8)).all():   # untouched prefill: not a hand label
            print(f'{k}: identical to pseudo-GT, skipped (not annotated yet?)')
            continue
        cv2.imwrite(f'{GT}/{k:05d}_{tag}.png', lab)
        keep = lab != 2
        tp = (p & h & keep).sum()
        print(f'{k}: pseudo-GT precision {tp / max((p & keep).sum(), 1):.2f} recall {tp / max(h.sum(), 1):.2f}'
              f'  void {(~keep).mean() * 100:.2f}% of frame')


if __name__ == '__main__':
    export() if sys.argv[1] == 'export' else import_(*sys.argv[2:])
