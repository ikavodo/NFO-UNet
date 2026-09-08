"""Does prompted SAM2 (video-mode memory propagation) drift under fragmented occlusion?

    python -m tracking.eval.sam2_drift_check --seq seq1_gt --run-index 4

Written to check ONE specific claim before it went on a poster: box-prompted SAM2 loses the person
behind foliage after sustained occlusion. Prompts SAM2's VIDEO predictor with a box at the FIRST
ground-truth frame of a run (the standard "track this object" usage), propagates forward through
the WHOLE traversal with no further input, and measures mask-centroid error against ground truth
EVERY frame - a real measurement, not a GT-free proxy, because every frame in an NFO GT run has a
labelled box (unlike master_thesis's real-webcam-footage version of this same check, which had no
ground truth to measure against at all).

Reported honestly whichever way it comes out - this script does not know in advance whether SAM2
drifts here, and the poster text is written from whatever this prints, not the other way round.
"""
import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from tracking.eval.lookbehind_discrimination import gt_runs, load_sequence

FRAMES_DIR = '/tmp/claude-1001/-home-akovi-PycharmProjects-NFO-UNet/d139c459-f3be-4f3a-8bd5-bf7cc9c9b767/scratchpad/sam2_drift_frames'


def gt_box_px(gt, f, w, h):
    x, y, bw, bh = gt[f]
    return (x * w, y * h, (x + bw) * w, (y + bh) * h)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', default='seq1_gt')
    p.add_argument('--data-dir', default='data/nfo_processed')
    p.add_argument('--run-index', type=int, default=4)
    p.add_argument('--out-dir', default='images/stream')
    a = p.parse_args()

    frames, gt = load_sequence(os.path.join(a.data_dir, a.seq))
    T_all, H, W = frames.shape
    run = gt_runs(gt)[a.run_index]
    abs_idx = list(range(run[0], run[1] + 1))
    T = len(abs_idx)
    print(f'{a.seq} run {a.run_index} f{run[0]}-{run[1]} (n={T})')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    video_predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2.1-hiera-small', device=device)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    for t, f in enumerate(abs_idx):
        cv2.imwrite(f'{FRAMES_DIR}/{t:05d}.jpg', frames[f])
    box0 = np.array(gt_box_px(gt, abs_idx[0], W, H), dtype=np.float32)

    pred = np.zeros((T, H, W), dtype=bool)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=FRAMES_DIR)
        video_predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box0)
        for fidx, obj_ids, mask_logits in video_predictor.propagate_in_video(state, reverse=False):
            pred[fidx] = (mask_logits[0, 0] > 0).cpu().numpy()
    for fp in glob.glob(f'{FRAMES_DIR}/*.jpg'):
        os.remove(fp)

    err_px, area_px, lost = [], [], []
    for t, f in enumerate(abs_idx):
        gx1, gy1, gx2, gy2 = gt_box_px(gt, f, W, H)
        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        m = pred[t]
        area_px.append(int(m.sum()))
        if m.sum() == 0:
            err_px.append(np.nan)
            lost.append(True)
        else:
            ys, xs = np.where(m)
            err_px.append(float(np.hypot(xs.mean() - gcx, ys.mean() - gcy)))
            lost.append(False)
    err_px, area_px, lost = np.array(err_px), np.array(area_px), np.array(lost)

    valid = ~np.isnan(err_px)
    print(f'  frames with an EMPTY mask (total loss): {lost.sum()}/{T} '
          f'({100 * lost.mean():.0f}%)')
    if valid.any():
        print(f'  centroid error vs GT (frames with a mask): mean={err_px[valid].mean():.1f}px '
              f'median={np.median(err_px[valid]):.1f}px max={err_px[valid].max():.1f}px')
        print(f'  first frame where error exceeds 20px (a body width at this scale): '
              f'{"never" if not (err_px[valid] > 20).any() else abs_idx[np.flatnonzero(err_px > 20)[0]]}')

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(abs_idx, err_px, '.-', color='crimson')
    axes[0].axhline(20, color='gray', linestyle='--', linewidth=1, label='~1 body width')
    for t, f in enumerate(abs_idx):
        if lost[t]:
            axes[0].axvspan(f - 0.5, f + 0.5, color='crimson', alpha=0.15)
    axes[0].set_ylabel('mask-centroid error\nvs GT (px)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'{a.seq} run {a.run_index}: box-prompted SAM2 video propagation, '
                      f'no re-prompting, frame {run[0]} onward')
    axes[1].plot(abs_idx, area_px, '.-', color='steelblue')
    axes[1].set_ylabel('mask area (px)')
    axes[1].set_xlabel('frame index')
    for ax in axes:
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(f'{a.out_dir}/{a.seq}_sam2_video_drift.png', dpi=130)
    print(f'  wrote {a.out_dir}/{a.seq}_sam2_video_drift.png')

    sample_t = np.linspace(0, T - 1, 6).astype(int)
    tiles = []
    for t in sample_t:
        f = abs_idx[t]
        ov = cv2.cvtColor(frames[f], cv2.COLOR_GRAY2BGR)
        ov[pred[t]] = (0.4 * ov[pred[t]] + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        gx1, gy1, gx2, gy2 = (int(v) for v in gt_box_px(gt, f, W, H))
        cv2.rectangle(ov, (gx1, gy1), (gx2, gy2), (0, 0, 255), 1)
        cv2.putText(ov, f'f{f}  err={err_px[t]:.0f}px' if not lost[t] else f'f{f}  LOST',
                   (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(cv2.resize(ov, (280, 280)))
    montage_path = f'{a.out_dir}/{a.seq}_sam2_video_drift_frames.png'
    cv2.imwrite(montage_path, np.hstack(tiles))
    print(f'  wrote {montage_path}  (green=SAM2 mask, red=GT box)')


if __name__ == '__main__':
    main()
