"""T4 (ours, one-shot causal) and its 2x2 controls T4-1, T4-2(m), T4-3 (spec §4).

    ../master_thesis/.venv/bin/python -m benchmark.nfo_vos.run_t4

One tracker run per trial is shared by all variants, so they differ only in (integration on/off) x
(SAM2 on/off). Everything is strictly causal: frame t's output uses frames t0..t only (tested in
tests/test_run_t4.py). 224 space throughout; masks are cached at 224.

Reused unchanged: foreground_mask/refine_mask/filter_by_shape (tracking/core/preprocess.py),
detect_blobs/track_blobs/merged_center (tracking/core/blob_tracker.py), scale_relative_params
(tracking/core/track_sequence.py), crop_at/fuse/restrict_to_nearby (tracking/core/integrate_image.py),
visibility_within_mask/project_to_frame and the image predictor of gt_sam_gate.py.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import trials as TR
from tracking.core.blob_tracker import _Track, detect_blobs, merged_center, track_blobs
from tracking.core.integrate_image import crop_at, fuse, restrict_to_nearby
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask
from tracking.core.track_sequence import scale_relative_params
from tracking.eval import gt_sam_gate
from tracking.eval.gt_sam_gate import project_to_frame, visibility_within_mask

BUFFER = 7            # last 7 consecutive frames (NTH_FRAME = 1), spec §4
MAX_AGE = 6           # track_blobs' default; a track unseen for longer is dead -> re-acquire
CROP_MULT = 4.5       # gt_sam_gate.py's default crop_mult
CKPT = os.path.abspath('../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt')
CACHE = 'results/benchmark/pilot/masks'
VOTE_M = range(1, BUFFER + 1)


def methods():
    return ['t4', 't4-1', *[f't4-2_m{m}' for m in VOTE_M], 't4-3']


def use_base_plus():
    """Spec §4: gt_sam_gate's image predictor is hardcoded to sam2.1-hiera-small; switch the same
    module-level predictor to base_plus."""
    if gt_sam_gate._predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        gt_sam_gate._predictor = SAM2ImagePredictor(
            build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml', CKPT, device='cuda'))


def blob_vote(aligned_blobs, m):
    """A pixel joins S_blob if at least min(m, N) of the N buffered aligned blobs cover it."""
    return aligned_blobs.sum(0) >= min(m, len(aligned_blobs))


def prompt_in_crop(blob, x0, y0, gt=None):
    """(point, box) for the SAM2 image call, in crop coordinates (frame coords minus the crop
    origin x0, y0 that crop_at uses). At t0 (gt = the trial) it is the shared prompt every method
    gets: p* and the GT box (spec §3). After t0 the same rule is applied to the method's own
    evidence, no GT: the deepest point (distance-transform maximum) of the tracker's person blobs
    at frame t, and their merged bounding box. Returns (None, None) when there is no blob."""
    off = np.array([x0, y0, x0, y0], float)
    if gt is not None:
        return (gt['point_224'][0] - x0, gt['point_224'][1] - y0), np.array(gt['box_224']) - off
    if not blob.any():
        return None, None
    D = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(D), D.shape)
    ys, xs = np.nonzero(blob)
    return (px - x0, py - y0), np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float) - off


def segment_prompted(img, point, box, fallback):
    """gt_sam_gate.segment_reference with an optional box (same predictor, same multimask argmax)."""
    pred = gt_sam_gate._predictor
    pred.set_image(np.stack([img] * 3, axis=-1))
    pt = np.array([point if point is not None else fallback])
    masks, scores, _ = pred.predict(point_coords=pt, point_labels=np.array([1]),
                                    box=box, multimask_output=True)
    return masks[np.argmax(scores)] > 0.5


def load(idx_range):
    return np.stack([cv2.imread(f'{TR.SEQ_DIR}/{i:05d}_or.jpg', 0) for i in idx_range])


def track(trial, max_frames=None):
    """Causal tracker. Track at t0: the detection whose centre lies in the GT box (largest if
    several). Followed by identity; once unseen for > MAX_AGE frames, re-acquired as the detection
    nearest the last predicted position. Returns per-frame foreground masks, detections and the
    followed chain {local t: (x, y)}."""
    frames = load(trial['frames'][:max_frames] if max_frames else trial['frames'])
    w0, w1 = trial['warmup']
    h0 = trial['box_224'][3] - trial['box_224'][1]
    kw, (p_var, q_var, r_var) = scale_relative_params(h0)
    masks = foreground_mask(frames, warmup_frames=load(range(w0, w1 + 1)))
    masks = refine_mask(masks, kw['close_kernel_size'], kw['open_kernel_size'])
    masks = filter_by_shape(masks, min_area=kw['min_area'], min_solidity=0.1)
    dets = detect_blobs(masks, min_area=kw['min_area'])
    saved = (_Track.P_VAR, _Track.Q_VAR, _Track.R_VAR)
    _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = p_var, q_var, r_var
    try:
        tracks = track_blobs(dets, max_dist=kw['max_dist'], max_age=MAX_AGE)
    finally:
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = saved

    def at(k):
        return [tr for tr in tracks if k in tr.history]

    x0, y0, x1, y1 = trial['box_224']
    area = lambda tr, k: tr.history[k][2] * tr.history[k][3]          # blob height x width
    inside = [tr for tr in at(0) if x0 <= tr.history[0][0] <= x1 and y0 <= tr.history[0][1] <= y1]
    cur = max(inside, key=lambda tr: area(tr, 0)) if inside else None
    chain, last_seen = {}, 0
    if cur is not None:
        chain[0] = cur.history[0][:2]
    else:                                   # no blob in the box at t0: start from the box centre
        chain[0] = ((x0 + x1) / 2, (y0 + y1) / 2)
    for k in range(1, len(frames)):
        if cur is not None and k in cur.history:
            chain[k], last_seen = cur.history[k][:2], k
        elif k - last_seen > MAX_AGE or cur is None:
            px, py = readout_line(chain, k)[:2]
            cands = at(k)
            if cands:
                cur = min(cands, key=lambda tr: np.hypot(tr.history[k][0] - px, tr.history[k][1] - py))
                chain[k], last_seen = cur.history[k][:2], k
    return dict(frames=frames, masks=masks, dets=dets, chain=chain, kw=kw, h0=h0)


def readout_line(chain, t):
    """Position/velocity at t from the chain's own detections inside the causal buffer: OLS on x
    (the tracker's motion model is horizontal-only), mean y. Coasts on the last known position
    when the buffer holds fewer than two detections."""
    pts = sorted((k, chain[k]) for k in range(max(0, t - BUFFER + 1), t + 1) if k in chain)
    if len(pts) >= 2:
        ks = np.array([k for k, _ in pts], float)
        xs = np.array([p[0] for _, p in pts])
        vx, b = np.polyfit(ks, xs, 1)
        x = chain[t][0] if t in chain else vx * t + b
        return x, float(np.mean([p[1] for _, p in pts])), float(vx)
    k_last = max(k for k in chain if k <= t)
    return chain[k_last][0], chain[k_last][1], 0.0


def run_trial(trial, max_frames=None):
    """fp16 autocast for every SAM2 call, the same precision B0/T1 run at (run_sam2_video.py)."""
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        return _run_trial(trial, max_frames)


def _run_trial(trial, max_frames=None):
    use_base_plus()
    t_start = time.perf_counter()
    P = track(trial, max_frames)
    t_track = time.perf_counter() - t_start
    frames, masks, dets, kw = P['frames'], P['masks'], P['dets'], P['kw']
    T, H, W = frames.shape
    crop = int(np.clip(round(CROP_MULT * P['h0']), 60, min(H, W)))
    centre = (crop / 2, crop / 2)
    out = {m: np.zeros((T, H, W), bool) for m in methods()}
    timing = {m: t_track for m in methods()}            # every variant pays the shared tracker
    for t in range(T):
        ax, ay, vx = readout_line(P['chain'], t)
        cx, cy = merged_center(dets[t], ax, ay, kw['merge_radius'])      # whole-person centre
        B = list(range(max(0, t - BUFFER + 1), t + 1))
        xs = [cx + vx * (k - t) for k in B]               # constant-velocity path, pinned at t
        aligned = np.stack([crop_at(frames[k], x, cy, crop) for k, x in zip(B, xs)])
        back = lambda m: project_to_frame(m, cx, cy, crop, (H, W))

        s = time.perf_counter()
        blobs = [restrict_to_nearby((masks[k] > 0).astype(np.uint8), masks[k], dets[k], x, cy,
                                    kw['merge_radius']) > 0 for k, x in zip(B, xs)]
        t_blob = time.perf_counter() - s
        out['t4-3'][t] = blobs[-1]
        timing['t4-3'] += t_blob
        x0, y0 = int(cx - crop / 2), int(cy - crop / 2)               # crop_at's own origin
        point, box = prompt_in_crop(blobs[-1], x0, y0, gt=trial if t == 0 else None)

        s = time.perf_counter()
        ref = fuse(aligned, method='median')
        S = segment_prompted(ref, point, box, centre)
        out['t4'][t] = back(visibility_within_mask(aligned, ref, S)[-1])
        timing['t4'] += time.perf_counter() - s + t_blob

        s = time.perf_counter()               # T4-1: reference = current frame; visibility = identity
        out['t4-1'][t] = back(segment_prompted(aligned[-1], point, box, centre))
        timing['t4-1'] += time.perf_counter() - s + t_blob

        s = time.perf_counter()
        ab = np.stack([crop_at(b.astype(np.uint8), x, cy, crop) > 0 for b, x in zip(blobs, xs)])
        for m in VOTE_M:
            out[f't4-2_m{m}'][t] = back(visibility_within_mask(aligned, ref, blob_vote(ab, m))[-1])
        t_vote = (time.perf_counter() - s) / len(VOTE_M)
        for m in VOTE_M:
            timing[f't4-2_m{m}'] += t_blob + t_vote
    fr = np.array(trial['frames'][:T])
    return {m: dict(masks=out[m], frames=fr, sec_per_frame=timing[m] / T) for m in methods()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    trials = [t for t in json.load(open(TR.OUT)) if t['admissible']][:a.limit]
    for m in methods():
        os.makedirs(os.path.join(CACHE, m), exist_ok=True)
    for t in trials:
        paths = {m: os.path.join(CACHE, m, f"{t['id']}.npz") for m in methods()}
        if all(os.path.exists(q) for q in paths.values()) and not a.force:
            continue
        torch.cuda.reset_peak_memory_stats()
        res = run_trial(t)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        for m, r in res.items():
            np.savez_compressed(paths[m], peak_mem_gb=peak, **r)
        print(f"{t['id']}: t4 {res['t4']['sec_per_frame'] * 1000:.0f} ms/frame, "
              f"t4-3 {res['t4-3']['sec_per_frame'] * 1000:.0f} ms/frame, peak {peak:.2f} GB")


if __name__ == '__main__':
    main()
