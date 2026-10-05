"""Does image-level integration help at all? T4-2 (warped blob vote + visibility step, no SAM2)
against T4-3 (frame-t blob), over alignment {tracker, oracle} x buffer stride {1, 2, 3, 4}.

    python -m benchmark.nfo_vos.diag_integration      # CPU only, ~minutes

Mechanism being tested (derived): the median over N aligned frames recovers a person pixel only
if it is unoccluded in > N/2 of them. Under a static occluder of width w and person motion
delta px/frame, buffer stride s gives w < N * delta * s / 2. At 224 px delta is ~1 px/frame,
so with s = 1 and N = 7 only gaps < 3.5 px can fill. Stride widens the span, and under
independent centroid noise it leaves the end-of-buffer misalignment unchanged
(6s * SE(v) = 6 sigma / sqrt(28) for any s).

oracle: relative alignment from the GT box centres (a ceiling, not a method). The output is still
placed at the tracker's own position at t, so only the relative warp differs from 'tracker'.
"""
import os

import cv2
import json
import numpy as np
import pandas as pd

from benchmark.nfo_vos import metrics, run_t4 as R, trials as TR
from tracking.core.blob_tracker import merged_center
from tracking.core.integrate_image import crop_at, fuse, restrict_to_nearby
from tracking.eval.gt_sam_gate import project_to_frame, visibility_within_mask

N = 7
STRIDES = (1, 2, 3, 4)
ALIGN = ('tracker', 'oracle')
OUT_CSV = 'results/benchmark/pilot/diag_integration.csv'
OUT_IMG = 'images/benchmark/pilot_checks/diag_integration_refs.png'


def buffer_indices(t, stride):
    return [t - stride * j for j in range(N - 1, -1, -1) if t - stride * j >= 0]


def gt_centre(idx):
    bb = TR.bbs()[idx][0]
    return (bb.x + bb.w / 2) * TR.SIZE, (bb.y + bb.h / 2) * TR.SIZE


def run_trial(trial, P, stride, align, keep_refs=()):
    frames, masks, dets, kw = P['frames'], P['masks'], P['dets'], P['kw']
    T, H, W = frames.shape
    crop = int(np.clip(round(R.CROP_MULT * P['h0']), 60, min(H, W)))
    votes = {m: np.zeros((T, H, W), bool) for m in range(1, N + 1)}
    blob_t = np.zeros((T, H, W), bool)
    refs = {}
    for t in range(T):
        ax, ay, vx = R.readout_line(P['chain'], t, span=stride * (N - 1) + 1)
        cx, cy = merged_center(dets[t], ax, ay, kw['merge_radius'])
        B = buffer_indices(t, stride)
        if align == 'tracker':
            pos = [(cx + vx * (k - t), cy) for k in B]
        else:
            gx_t, gy_t = gt_centre(trial['frames'][t])
            pos = [(cx + gt_centre(trial['frames'][k])[0] - gx_t, cy + gt_centre(trial['frames'][k])[1] - gy_t) for k in B]
        aligned = np.stack([crop_at(frames[k], x, y, crop) for k, (x, y) in zip(B, pos)])
        ref = fuse(aligned, method='median')
        blobs = [restrict_to_nearby((masks[k] > 0).astype(np.uint8), masks[k], dets[k], x, y,
                                    kw['merge_radius']) > 0 for k, (x, y) in zip(B, pos)]
        blob_t[t] = blobs[-1]
        ab = np.stack([crop_at(b.astype(np.uint8), x, y, crop) > 0 for b, (x, y) in zip(blobs, pos)])
        for m in votes:
            S = R.blob_vote(ab, m)
            votes[m][t] = project_to_frame(visibility_within_mask(aligned, ref, S)[-1], cx, cy, crop, (H, W))
        if t in keep_refs:
            refs[t] = ref
    return votes, blob_t, refs


def jf(trial, P_stack):
    vals = []
    for dt in range(1, len(trial['frames']) - 1):          # DAVIS: first and last dropped
        G = TR.gt_mask(trial['frames'][dt])
        if G is None:
            continue
        m = metrics.frame_metrics(P_stack[dt], G)
        vals.append((m['J'] + m['F']) / 2)
    return float(np.mean(vals))


def main():
    trials = [t for t in json.load(open(TR.OUT)) if t['admissible']]
    rows, ref_tiles = [], {}
    show = trials[5]                                        # one trial for the visual trace
    for trial in trials:
        P = R.track(trial)
        for align in ALIGN:
            for s in STRIDES:
                keep = (20, 35, 49) if trial is show else ()
                votes, blob_t, refs = run_trial(trial, P, s, align, keep)
                for m, V in votes.items():
                    rows.append(dict(trial=trial['id'], align=align, stride=s, m=m, JF=jf(trial, V)))
                if s == 1 and align == 'tracker':
                    rows.append(dict(trial=trial['id'], align='-', stride=0, m=0, JF=jf(trial, blob_t)))
                if refs:
                    ref_tiles[(align, s)] = refs
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    base = df[df.stride == 0].set_index('trial').JF                 # T4-3
    print(f'T4-3 (frame-t blob): mean J&F {base.mean():.3f}')
    print('align   stride  span  best m  T4-2 J&F  minus T4-3  trials better')
    for align in ALIGN:
        for s in STRIDES:
            sub = df[(df['align'] == align) & (df.stride == s)]
            m = sub.groupby('m').JF.mean().idxmax()
            v = sub[sub.m == m].set_index('trial').JF
            d = v - base
            print(f'{align:8s} {s:4d}  {s * (N - 1) + 1:4d}  {m:5d}  {v.mean():8.3f}  {d.mean():+9.3f}  {(d > 0).sum():4d}/{len(d)}')

    rows_img = []
    for (align, s), refs in sorted(ref_tiles.items()):
        tiles = []
        for t, ref in sorted(refs.items()):
            im = cv2.cvtColor(cv2.resize(ref, (180, 180)), cv2.COLOR_GRAY2BGR)
            cv2.putText(im, f'{align} s={s} +{t}', (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            tiles.append(im)
        rows_img.append(np.hstack(tiles))
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    cv2.imwrite(OUT_IMG, np.vstack(rows_img))
    print(f'median references for {show["id"]} -> {OUT_IMG}')


if __name__ == '__main__':
    main()
