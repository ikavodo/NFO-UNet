"""Gate a per-frame visibility mask using ONE SAM2 mask computed on the integrated image.

    python -m tracking.eval.gt_sam_gate --seq seq1_gt --window 31 --limit 2

Adapted from master_thesis/experiments/prototypes/em_tum_sam2_prompt_iteration.py's core idea,
cut down to just the mechanism the user asked to see - none of that script's comparison metrics
(SAM2-auto video propagation, Jaccard, entropy-purity, boundary fidelity, soft posterior) are
reproduced here. This is the quick version, not the audited one.

THE MECHANISM, four steps:
  1. Build the aligned stack and its integrated (fused) reference image for one GT segment -
     exactly gt_integrated_image.py's own align_frames/fuse pipeline, GT-driven velocity.
  2. Run SAM2 ONCE on that integrated reference, point-prompted at the aligned anchor position -
     which, by construction of align_frames' crop_at, is always the crop's own center pixel (the
     whole point of alignment is that the person sits at a fixed location in every aligned frame).
     This gives ONE segmentation mask S, in the aligned/integrated coordinate frame.
  3. Per frame t, threshold the residual |aligned[t] - reference| RESTRICTED TO S (a per-frame Otsu
     cut, calibrated only on residual values inside S, same as the reference script's V_t) to get
     which pixels within the object's own mask currently match the reference appearance - i.e.
     which pixels of the person are actually visible (not occluded) in frame t.
  4. Project (V_t & S) back onto frame t's OWN coordinates - undoing step 1's alignment - so the
     result is a mask on the ORIGINAL, un-aligned footage, not just in the aligned/cropped frame.

Step 4 is a plain inverse translation, not a general un-warp, because align_frames' crop_size here
equals the whole 224x224 frame (crop_mult*mean_height already clips to the frame in this dataset -
see gt_integrated_image.py), so crop_at degenerates to a pure per-frame x/y shift with zero-fill,
and undoing it is the same shift in the other direction.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from tracking.core.blob_tracker import _Track, detect_blobs, score_and_fit, track_blobs
from tracking.core.integrate_image import align_frames, anchor_for_frame, fuse
from tracking.core.preprocess import estimate_person_height, filter_by_shape, foreground_mask, refine_mask
from tracking.core.track_sequence import scale_relative_params
from tracking.eval.gt_integrated_image import build_gt_winner
from tracking.eval.lookbehind_discrimination import gt_runs, load_sequence

_predictor = None


def sam2_predictor():
    global _predictor
    if _predictor is None:
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', None, device=device)
        ckpt = hf_hub_download('facebook/sam2.1-hiera-small', 'sam2.1_hiera_small.pt')
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        _predictor = SAM2ImagePredictor(model)
    return _predictor


def segment_reference(reference_img, point_xy):
    pred = sam2_predictor()
    pred.set_image(np.stack([reference_img] * 3, axis=-1))
    masks, scores, _ = pred.predict(point_coords=np.array([point_xy]),
                                    point_labels=np.array([1]), multimask_output=True)
    return masks[np.argmax(scores)] > 0.5, float(scores.max())


def visibility_within_mask(aligned, reference_img, S):
    """Per-frame Otsu on the residual, calibrated ONLY on values inside S (see module docstring
    step 3), not the whole frame - the object's own residual distribution, not swamped by
    irrelevant background."""
    T = aligned.shape[0]
    residual = np.abs(aligned.astype(np.int16) - reference_img.astype(np.int16)[None])
    V = np.zeros((T,) + reference_img.shape, dtype=bool)
    if S.sum() == 0:
        return V
    for t in range(T):
        r8 = np.clip(residual[t], 0, 255).astype(np.uint8)
        thresh, _ = cv2.threshold(r8[S].reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        V[t] = r8 <= thresh
    return V & S[None]


def project_to_frame(mask, cx_t, cy, crop_size, frame_shape):
    """Undo align_frames' crop_at for one frame: crop_at reads frame[y0:y0+size, x0:x0+size] into
    the aligned output (x0=int(cx-size/2), y0=int(cy-size/2) - SAME truncation, so this must match
    exactly or the projection drifts by a pixel). Placing the aligned-space mask back at (x0,y0) is
    the exact inverse, and reduces to a plain translate+zero-fill because crop_size==frame_shape
    here (no true sub-crop, see module docstring)."""
    x0, y0 = int(cx_t - crop_size / 2), int(cy - crop_size / 2)
    M = np.float32([[1, 0, x0], [0, 1, y0]])
    return cv2.warpAffine(mask.astype(np.uint8), M, (frame_shape[1], frame_shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0) > 0


def build_tracker_winner(seg_frames, bg_frames=30, var_threshold=16.0, max_age=6, min_track_length=3):
    """The TRACKER's own equivalent of build_gt_winner: no ground truth anywhere in this
    function. person_height is measured from the footage itself (estimate_person_height), which
    scale_relative_params turns into every other scale-dependent parameter (association gate,
    merge radius, expected-height scoring term, morphology kernels, Kalman covariances) - the
    same scale-free pipeline track_window()/track_windows_in_sequence use elsewhere in this
    project, reproduced here at the statement level (not called through those wrappers) because
    both of them REDUCE their result to a single readout position; align_frames needs the full
    score_and_fit dict ('frames'/'history'/'vx'), which is exactly what this returns instead.

    Returns (winner_or_None, measured_person_height). winner is None if no track reached
    min_track_length within this segment - a real possible outcome, not an error, and the caller
    must handle it rather than assume a track always exists the way GT guarantees one.
    """
    person_height = estimate_person_height(seg_frames, bg_frames=bg_frames, var_threshold=var_threshold)
    kw, (p_var, q_var, r_var) = scale_relative_params(person_height)
    masks = foreground_mask(seg_frames, bg_frames=bg_frames, var_threshold=var_threshold)
    masks = refine_mask(masks, kw['close_kernel_size'], kw['open_kernel_size'])
    masks = filter_by_shape(masks, min_area=kw['min_area'], min_solidity=0.1)
    dets = detect_blobs(masks, min_area=kw['min_area'])
    saved = (_Track.P_VAR, _Track.Q_VAR, _Track.R_VAR)
    _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = p_var, q_var, r_var    # restored in finally: this
    try:                                                              # is shared class state
        tracks = track_blobs(dets, max_dist=kw['max_dist'], max_age=max_age)
        winner = score_and_fit(tracks, min_track_length=min_track_length,
                               expected_height=kw['expected_height'], height_tolerance=0.5)
    finally:
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = saved
    return winner, person_height


def render_row(winner, seg, abs_idx, frames, H, W, crop_mult, samples, label):
    """One row of the comparison: integrated reference + SAM2 mask, then that mask projected
    onto `samples` raw frames. Shared by the GT and tracker paths so the two rows are built by
    IDENTICAL code - the only difference between rows is which winner produced the alignment."""
    T = len(abs_idx)
    center_t = T // 2
    mean_h = float(np.mean([winner['history'][t][2] for t in winner['history']
                            if winner['history'][t][2] is not None]) or 0.0)
    ay = float(np.mean([winner['history'][t][1] for t in winner['frames']]))
    crop = int(np.clip(round(crop_mult * max(mean_h, 1.0)), 60, min(H, W)))

    aligned = align_frames(seg, winner, crop_size=crop)
    reference = fuse(aligned, method='median')
    ax, _ = anchor_for_frame(winner, center_t)
    point_xy = (crop / 2, crop / 2)
    S, score = segment_reference(reference, point_xy)
    V = visibility_within_mask(aligned, reference, S)
    frac_visible = V[:, S].mean() if S.sum() else float('nan')
    print(f'    [{label}] mean_height={mean_h:.1f}px crop={crop} SAM2 mask area={S.sum()}px '
          f'score={score:.3f} visible-within-mask={frac_visible:.3f}')

    contours, _ = cv2.findContours(S.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_vis = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    for c in contours:
        cv2.drawContours(ref_vis, [c], -1, (0, 255, 0), 2)
    cv2.circle(ref_vis, (int(point_xy[0]), int(point_xy[1])), 4, (0, 0, 255), -1)
    tiles = [cv2.resize(ref_vis, (260, 260))]
    sample_t = np.linspace(0, T - 1, min(samples, T)).astype(int)
    for t in sample_t:
        cx_t = ax + winner['vx'] * (t - center_t)
        proj = project_to_frame(V[t], cx_t, ay, crop, (H, W))
        raw = frames[abs_idx[t]]
        ov = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        ov[proj] = (0.4 * ov[proj] + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        cv2.putText(ov, f'f{abs_idx[t]}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(cv2.resize(ov, (260, 260)))
    row = np.hstack(tiles)
    cv2.putText(row, label, (4, row.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', default='seq1_gt')
    p.add_argument('--data-dir', default='data/nfo_processed')
    p.add_argument('--window', type=int, default=31)
    p.add_argument('--limit', type=int, default=2)
    p.add_argument('--crop-mult', type=float, default=4.5)
    p.add_argument('--samples', type=int, default=4, help='frames to show per segment')
    p.add_argument('--compare', action='store_true',
                   help="also build the SAME segment's alignment from the TRACKER's own estimate "
                        "(no ground truth: person_height measured from the footage, scale_relative_params "
                        "for everything else) and stack it as a second row beneath the GT-driven "
                        "row, same sampled frames, for a direct visual comparison")
    p.add_argument('--out-dir', default='images/stream')
    a = p.parse_args()

    seq_dir = os.path.join(a.data_dir, a.seq)
    frames, gt = load_sequence(seq_dir)
    T_all, H, W = frames.shape
    runs = gt_runs(gt)[:a.limit]
    print(f'{a.seq}: {len(runs)} segments (of {len(gt_runs(gt))} total)')

    for si, run in enumerate(runs):
        if a.window and (run[1] - run[0] + 1) > a.window:
            mid = (run[0] + run[1]) // 2
            half = a.window // 2
            run = (max(run[0], mid - half), min(run[1], mid - half + a.window - 1))
        gt_winner, abs_idx = build_gt_winner(gt, run, W, H)
        seg = frames[abs_idx]
        print(f'  segment {si} f{run[0]}-{run[1]} (n={len(abs_idx)}):')
        rows = [render_row(gt_winner, seg, abs_idx, frames, H, W, a.crop_mult, a.samples,
                           f'GT alignment  vx={gt_winner["vx"]:+.2f}')]

        if a.compare:
            tr_winner, tr_height = build_tracker_winner(seg)
            if tr_winner is None:
                print(f'    [tracker] no track reached min_track_length in this segment - '
                      f'no comparison row (this is a real outcome, not an error)')
            else:
                rows.append(render_row(tr_winner, seg, abs_idx, frames, H, W, a.crop_mult, a.samples,
                                       f'tracker alignment  vx={tr_winner["vx"]:+.2f}  '
                                       f'(measured height {tr_height:.0f}px)'))

        w = max(r.shape[1] for r in rows)
        rows = [r if r.shape[1] == w else cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
                                                              cv2.BORDER_CONSTANT) for r in rows]
        montage = np.vstack(rows)
        out = f'{a.out_dir}/{a.seq}_sam_gate_seg{si}.png'
        cv2.imwrite(out, montage)
        print(f'    wrote {out}')


if __name__ == '__main__':
    main()
