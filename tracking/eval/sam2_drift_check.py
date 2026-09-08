"""Does prompted SAM2 (video-mode memory propagation) drift under fragmented occlusion?

    python -m tracking.eval.sam2_drift_check --seq seq1_gt --run-index 4
    python -m tracking.eval.sam2_drift_check --seq seq1_gt --run-index 4 --prompts box
    python -m tracking.eval.sam2_drift_check --seq seq1_gt --run-index 4 --prompts point

Written to check ONE specific claim before it went on a poster: box-prompted SAM2 loses the person
behind foliage after sustained occlusion (confirmed - see the first version of this script and
images/stream/seq1_gt_sam2_video_drift*.png). This version asks the follow-up: is that specific to
a BOX prompt (which gives SAM2 the object's extent, not just its location), and separately, does
the failure look like drift from frame 0, or a correctly-tracked stretch that then JUMPS onto a
distractor partway through ("gets stuck at some point in the middle")?

Both prompt kinds (box, and a single point at the GT box's center) run by default, both seeded with
GROUND TRUTH at frame 0 - if the object were prompt-perfectly localised and still ends up 180px away
by the end of a clip, the failure is in the propagation, not in an imprecise starting point. A
mid-sequence jump is distinguished from drift-from-the-start by checking for an error spike after a
genuinely low-error settled stretch, not just any threshold crossing.

Every frame in an NFO GT run has a labelled box, so error against ground truth is a real
measurement here, not a GT-free proxy the way master_thesis's real-webcam version of this same
check had to settle for. Reported honestly whichever way it comes out.
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


def run_propagation(video_predictor, frames_dir, T, H, W, gt, f0, prompt_kind):
    """Seed the video predictor at frame 0 with a GT-derived prompt (box or a single point at the
    GT box's center) and propagate forward with NO further correction. Returns the [T,H,W] mask
    stack. Both prompt kinds are tried because a box gives SAM2 the object's EXTENT as well as its
    location on frame 0, while a point gives only location - if drift happens either way, it is not
    an artefact of prompt geometry specifically."""
    box0 = np.array(gt_box_px(gt, f0, W, H), dtype=np.float32)
    kwargs = dict(box=box0) if prompt_kind == 'box' else dict(
        points=np.array([[(box0[0] + box0[2]) / 2, (box0[1] + box0[3]) / 2]], dtype=np.float32),
        labels=np.array([1]))
    pred = np.zeros((T, H, W), dtype=bool)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=frames_dir)
        video_predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, **kwargs)
        for fidx, obj_ids, mask_logits in video_predictor.propagate_in_video(state, reverse=False):
            pred[fidx] = (mask_logits[0, 0] > 0).cpu().numpy()
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', default='seq1_gt')
    p.add_argument('--data-dir', default='data/nfo_processed')
    p.add_argument('--run-index', type=int, default=4)
    p.add_argument('--prompts', nargs='+', choices=('box', 'point'), default=['box', 'point'],
                   help="which GT-derived prompt(s) to seed frame 0 with, propagating forward "
                        "with NO further correction. Runs both by default, to check whether "
                        "drift is specific to one prompt geometry or happens either way.")
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

    results = {}
    for kind in a.prompts:
        print(f'  --- prompt: {kind} ---')
        pred = run_propagation(video_predictor, FRAMES_DIR, T, H, W, gt, abs_idx[0], kind)

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
        results[kind] = (pred, err_px, area_px, lost)

        valid = ~np.isnan(err_px)
        print(f'    frames with an EMPTY mask (total loss): {lost.sum()}/{T} ({100 * lost.mean():.0f}%)')
        if valid.any():
            print(f'    centroid error vs GT: mean={err_px[valid].mean():.1f}px '
                  f'median={np.median(err_px[valid]):.1f}px max={err_px[valid].max():.1f}px')
            over = np.flatnonzero(err_px > 20)
            first_over = 'never' if len(over) == 0 else abs_idx[over[0]]
            print(f'    first frame exceeding one body-width (20px) of error: {first_over}')
            # "gets stuck at some point in the MIDDLE": distinct from drifting from frame 0 - a
            # correct prompt that later JUMPS onto a distractor partway through the clip, rather
            # than being wrong from the start. Detected as a big error INCREASE after a genuinely
            # low-error stretch, not just any threshold crossing (which the box-only run already
            # showed can happen from a near-zero baseline).
            settled = err_px[valid][:5].mean() if valid.sum() >= 5 else np.nan
            jump_idx = np.flatnonzero((err_px > 3 * max(settled, 5)) & valid)
            if len(jump_idx) and jump_idx[0] > 5:
                print(f'    jumps from a settled track (~{settled:.0f}px) to >{3 * max(settled, 5):.0f}px '
                      f'at frame {abs_idx[jump_idx[0]]} - a MID-SEQUENCE loss, not drift from the start')

    for fp in glob.glob(f'{FRAMES_DIR}/*.jpg'):
        os.remove(fp)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    colors = {'box': 'crimson', 'point': 'darkorange'}
    for kind, (pred, err_px, area_px, lost) in results.items():
        axes[0].plot(abs_idx, err_px, '.-', color=colors[kind], label=f'{kind} prompt', markersize=4)
        axes[1].plot(abs_idx, area_px, '.-', color=colors[kind], label=f'{kind} prompt', markersize=4)
    axes[0].axhline(20, color='gray', linestyle='--', linewidth=1, label='~1 body width')
    axes[0].set_ylabel('mask-centroid error\nvs GT (px)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'{a.seq} run {a.run_index}: SAM2 video propagation from a GT prompt at '
                      f'frame {run[0]}, no re-prompting after')
    axes[1].set_ylabel('mask area (px)')
    axes[1].set_xlabel('frame index')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.3)
    plt.tight_layout()
    plot_path = f'{a.out_dir}/{a.seq}_sam2_video_drift.png'
    plt.savefig(plot_path, dpi=130)
    print(f'  wrote {plot_path}')

    for kind, (pred, err_px, area_px, lost) in results.items():
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
        montage_path = f'{a.out_dir}/{a.seq}_sam2_video_drift_frames_{kind}.png'
        cv2.imwrite(montage_path, np.hstack(tiles))
        print(f'  wrote {montage_path}  (green=SAM2 mask, red=GT box, prompt={kind})')


if __name__ == '__main__':
    main()
