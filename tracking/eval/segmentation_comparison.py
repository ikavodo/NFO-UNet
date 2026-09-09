"""SAM2 vs our method: apples-to-apples, same clip, same 6 frames, same prompt TYPE.

    python -m tracking.eval.segmentation_comparison

Three fairness fixes over the first version of this script:

1. POINT prompts throughout, not box. Our method's own SAM2 call (gt_sam_gate.py's
   segment_reference) only ever gives SAM2 a point - the tracker supplies an estimated centroid,
   never a measured box. Comparing against a box-prompted SAM2 baseline would hand the baseline
   strictly more information than our method ever gets, which is not a fair fight. Costs nothing
   empirically: box vs point were already tested for this exact drift finding and converged to
   nearly identical failure curves and jump frames (see sam2_drift_check.py's --prompts run) - so
   switching to point-only changes no qualitative conclusion, it only closes a fairness gap.

2. Mask-only for the GOOD-INIT comparison, no ground-truth box on either row - the first version
   drew a GT box on SAM2's row only (inherited from sam2_drift_check.py's own rendering) and none
   on ours, an asymmetry a viewer could reasonably read as one method being more "checked" than the
   other. Both rows now show only the segmentation mask.

3. A new BAD-INIT comparison, WITH the GT box drawn on both rows this time - here the box earns
   its place, because the whole point of this comparison is letting a viewer see exactly how far a
   badly-initialized SAM2 mask has drifted from the truth, at a glance, without needing the
   error-vs-frame plot alongside it.

Both SAM2 conditions (bad-init: prompted at the run's own first frame f1154; good-init: prompted
at the confirmed-clear checkpoint f1216) are propagated in THIS script - not cropped from an
older render - specifically so a real, fresh wall-clock time can be measured for each and reported
alongside the images, next to our method's own (much cheaper) per-window SAM2 image-calls.
"""
import os
import time

import cv2
import numpy as np
import torch

from tracking.eval.gt_integrated_image import build_gt_winner
from tracking.eval.gt_sam_gate import render_row
from tracking.eval.lookbehind_discrimination import gt_runs, load_sequence
from tracking.eval.sam2_drift_check import FRAMES_DIR, gt_box_px

SEQ = 'seq1_gt'
RUN_INDEX = 4
FRAMES = (1216, 1234, 1252, 1271, 1289, 1308)          # shared across every row in both figures
WINDOWS = ((1216, 1246), (1247, 1277), (1278, 1308))   # 93 = 31x3: tiles the SAME span SAM2 covers
TILE = 280


def run_propagation_point(video_predictor, frames_dir, T, H, W, gt, f0, local_idx):
    """Point-prompted (not box) video propagation - see module docstring point 1."""
    x1, y1, x2, y2 = gt_box_px(gt, f0, W, H)
    point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
    pred = np.zeros((T, H, W), dtype=bool)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = video_predictor.init_state(video_path=frames_dir)
        video_predictor.add_new_points_or_box(state, frame_idx=local_idx, obj_id=1,
                                              points=point, labels=np.array([1]))
        for fidx, obj_ids, mask_logits in video_predictor.propagate_in_video(
                state, start_frame_idx=local_idx, reverse=False):
            pred[fidx] = (mask_logits[0, 0] > 0).cpu().numpy()
    return pred


def draw_gt_box(vis, gt, f, W, H):
    x1, y1, x2, y2 = (int(v) for v in gt_box_px(gt, f, W, H))
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return vis


def label_bar(width, text):
    bar = np.zeros((28, width, 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return bar


def sam2_row(pred, frames, abs_idx, gt, W, H, with_box):
    tiles = []
    for f in FRAMES:
        t = f - abs_idx[0]
        ov = cv2.cvtColor(frames[f], cv2.COLOR_GRAY2BGR)
        m = pred[t]
        ov[m] = (0.4 * ov[m] + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        if with_box:
            draw_gt_box(ov, gt, f, W, H)
        cv2.putText(ov, f'f{f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(cv2.resize(ov, (TILE, TILE)))
    return np.hstack(tiles)


def our_row(frames, gt, H, W, with_box):
    """Reuses gt_sam_gate.py's render_row unchanged; only adds the GT box on top when asked,
    since render_row itself never draws one.

    The box is drawn AFTER render_row's tiles have already been resized twice (224x224 native ->
    260x260 inside render_row -> TILExTILE here), so it must be computed in TILE-sized pixel
    space, not native W/H - gt boxes are stored normalised (fraction of frame), so passing
    (TILE, TILE) to gt_box_px gives coordinates already correct for THIS canvas directly, no
    separate rescale needed. Passing native (W, H) here (the first version's bug) drew the box at
    224-pixel scale on a 280-pixel canvas - too small and shifted toward the corner."""
    tiles = []
    for window in WINDOWS:
        abs_idx = list(range(window[0], window[1] + 1))
        winner, _ = build_gt_winner(gt, window, W, H)
        row = render_row(winner, frames[abs_idx], abs_idx, frames, H, W, crop_mult=4.5,
                         samples=0, label='', abs_frames=[f for f in FRAMES if window[0] <= f <= window[1]])
        strip = row[:, 260:]                          # drop render_row's own reference-image tile
        n = sum(window[0] <= f <= window[1] for f in FRAMES)
        strip = cv2.resize(strip, (TILE * n, TILE))
        if with_box:
            frames_in_window = [f for f in FRAMES if window[0] <= f <= window[1]]
            for i, f in enumerate(frames_in_window):
                tile = strip[:, i * TILE:(i + 1) * TILE]
                draw_gt_box(tile, gt, f, TILE, TILE)
        tiles.append(strip)
    return np.hstack(tiles)


CACHE_DIR = 'tracking/sam2_pseudo_mask_tmp/segmentation_comparison_cache'


def cached_propagation(label, video_predictor, frames_dir, T, H, W, gt, f0, local_idx):
    """Propagation output never depends on the box-drawing code downstream of it, so once a
    real wall-clock timing has been measured it is cached (mask array + elapsed seconds) rather
    than re-paying ~40-65s of GPU compute every time a rendering-only bug gets fixed. The cached
    number IS still a real measurement, just from the run that produced it, not necessarily the
    most recent one."""
    import pickle
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = f'{CACHE_DIR}/{label}.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            pred, elapsed = pickle.load(f)
        print(f'  [cache hit] {label}: {elapsed:.1f}s (measured previously, not re-timed)')
        return pred, elapsed
    t0 = time.perf_counter()
    pred = run_propagation_point(video_predictor, frames_dir, T, H, W, gt, f0, local_idx)
    elapsed = time.perf_counter() - t0
    with open(path, 'wb') as f:
        pickle.dump((pred, elapsed), f)
    return pred, elapsed


def main():
    frames, gt = load_sequence(f'data/nfo_processed/{SEQ}')
    T_all, H, W = frames.shape
    run = gt_runs(gt)[RUN_INDEX]
    assert run == (1154, 1308), f'seq1_gt run {RUN_INDEX} changed to {run} - update WINDOWS/FRAMES'
    T = run[1] - run[0] + 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    video_predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2.1-hiera-small', device=device)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    import glob
    for t, f in enumerate(range(run[0], run[1] + 1)):
        cv2.imwrite(f'{FRAMES_DIR}/{t:05d}.jpg', frames[f])

    pred_bad, t_bad = cached_propagation('bad_init', video_predictor, FRAMES_DIR, T, H, W, gt, run[0], 0)

    checkpoint_local = 1216 - run[0]
    pred_good, t_good = cached_propagation('good_init', video_predictor, FRAMES_DIR, T, H, W, gt,
                                           1216, checkpoint_local)

    for fp in glob.glob(f'{FRAMES_DIR}/*.jpg'):
        os.remove(fp)

    n_good_frames = T - checkpoint_local
    print(f'SAM2 bad-init:  {t_bad:.1f}s total for {T} frames propagated  '
          f'({1000 * t_bad / T:.0f} ms/frame)')
    print(f'SAM2 good-init: {t_good:.1f}s total for {n_good_frames} frames propagated  '
          f'({1000 * t_good / n_good_frames:.0f} ms/frame)')

    t0 = time.perf_counter()
    ours_row_no_box = our_row(frames, gt, H, W, with_box=False)
    t_ours = time.perf_counter() - t0
    n_our_calls = len(WINDOWS)
    print(f'Our method:     {t_ours:.1f}s total for {n_our_calls} SAM2 calls covering '
          f'{n_good_frames} frames  ({1000 * t_ours / n_our_calls:.0f} ms/call)')
    speedup = t_good / t_ours
    print(f'speedup over the SAME span vs SAM2 good-init: {speedup:.1f}x')

    # --- figure 1: GOOD-INIT, mask-only both rows, apples-to-apples ---
    sam2_good_tiles = sam2_row(pred_good, frames, list(range(run[0], run[1] + 1)), gt, W, H, with_box=False)
    fig1 = np.vstack([
        label_bar(sam2_good_tiles.shape[1],
                 f'SAM2, confirmed-clear init (point prompt), one propagation - {t_good:.0f}s / {n_good_frames}f'),
        sam2_good_tiles,
        label_bar(ours_row_no_box.shape[1],
                 f'Our method, {n_our_calls} independent windows (point prompt) - {t_ours:.1f}s / {n_our_calls} calls'),
        ours_row_no_box,
    ])
    cv2.imwrite('images/stream/seq1_gt_segmentation_comparison_goodinit.png', fig1)

    # --- figure 2: BAD-INIT, GT box on both rows ---
    sam2_bad_tiles = sam2_row(pred_bad, frames, list(range(run[0], run[1] + 1)), gt, W, H, with_box=True)
    ours_row_box = our_row(frames, gt, H, W, with_box=True)
    fig2 = np.vstack([
        label_bar(sam2_bad_tiles.shape[1],
                 f'SAM2, arbitrary init at f{run[0]} (point prompt), one propagation - {t_bad:.0f}s / {T}f. Red: ground truth'),
        sam2_bad_tiles,
        label_bar(ours_row_box.shape[1], 'Our method, same frames, same prompt type. Red: ground truth'),
        ours_row_box,
    ])
    cv2.imwrite('images/stream/seq1_gt_segmentation_comparison_badinit.png', fig2)

    print('wrote images/stream/seq1_gt_segmentation_comparison_goodinit.png')
    print('wrote images/stream/seq1_gt_segmentation_comparison_badinit.png')


if __name__ == '__main__':
    main()
