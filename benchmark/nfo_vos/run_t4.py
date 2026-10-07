"""T4-1n: per-frame SAM2 (image predictor, no memory) prompted by the causal blob tracker, on the
native frame; plus that tracker, which A/4 (a4_build.py, velocity.py) reuses.

    python -m benchmark.nfo_vos.run_t4 [--trial-index N]       # -> masks/t4-1n/<trial>.npz

At t0 the prompt is the shared GT box + p* (spec §3). After t0 it is GT-free: the deepest point
(distance-transform maximum) of the tracker's person blobs at t and their merged box, mapped from
224 to native through the padded square. Strictly causal: frame t uses frames t0..t only.
The 224-input T4 family, its blob controls and the box variants were removed (2026-10-07,
dead ends; see docs/nfo_failure_log.md and git history).

Reused unchanged: foreground_mask/refine_mask/filter_by_shape (tracking/core/preprocess.py),
detect_blobs/track_blobs/merged_center (tracking/core/blob_tracker.py), scale_relative_params
(tracking/core/track_sequence.py), restrict_to_nearby (tracking/core/integrate_image.py).
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
from tracking.core.integrate_image import restrict_to_nearby
from tracking.core.preprocess import filter_by_shape, foreground_mask, refine_mask
from tracking.core.track_sequence import scale_relative_params

BUFFER = 7            # causal velocity window (frames)
MAX_AGE = 6           # track_blobs' default; a track unseen for longer is dead -> re-acquire
CKPT = os.path.abspath(os.environ.get('NFO_SAM2_CKPT', '../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt'))
CACHE = f'{TR.RES}/masks'
METHOD = 't4-1n'
_predictor = None


def image_predictor():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        _predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_b+.yaml', CKPT, device='cuda'))
    return _predictor


def prompt_from_blob(blob):
    """(point, box) in 224 frame coordinates from the tracker's person blobs: distance-transform
    maximum and merged bounding box. (None, None) when there is no blob."""
    if not blob.any():
        return None, None
    D = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(D), D.shape)
    ys, xs = np.nonzero(blob)
    return (float(px), float(py)), np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def box_to_native(box):
    """224 box edges -> native 800x600 edges through the padded square, the inverse of the
    GT's own scale_and_pad (gen_nfo_pseudo_masks.gt_to_native, in pixel units)."""
    s = max(TR.NATIVE_W, TR.NATIVE_H)
    px, py = (s - TR.NATIVE_W) / 2, (s - TR.NATIVE_H) / 2
    return np.array([box[0] / TR.SIZE * s - px, box[1] / TR.SIZE * s - py,
                     box[2] / TR.SIZE * s - px, box[3] / TR.SIZE * s - py], float)


def segment_native(trial, idx, point, box):
    """SAM2 image predictor on the FULL native frame, single-mask output: the setup that matched
    B0 at t0 with the identical prompt (old pilot: J 0.700 vs 0.703)."""
    pred = image_predictor()
    pred.set_image(cv2.imread(f"data/nfo_final/nfo_final/{trial['seq']}/{idx:05d}.jpg")[:, :, ::-1].copy())
    masks, _, _ = pred.predict(point_coords=np.array([point]), point_labels=np.array([1]),
                               box=None if box is None else np.asarray(box), multimask_output=False)
    return masks[0] > 0.5


def load(idx_range, seq=TR.SEQ):
    return np.stack([cv2.imread(f'{TR.seq_dir(seq)}/{i:05d}_or.jpg', 0) for i in idx_range])


def detect_and_track(frame_idx, warmup, h0, seq=TR.SEQ):
    """MOG2 (warmed on the person-free warm-up range) -> morphology -> blobs -> Kalman/Hungarian
    tracks over CONTIGUOUS 224 frames. All scale-dependent parameters from the person height h0."""
    frames = load(frame_idx, seq)
    w0, w1 = warmup
    kw, (p_var, q_var, r_var) = scale_relative_params(h0)
    masks = foreground_mask(frames, warmup_frames=load(range(w0, w1 + 1), seq))
    masks = refine_mask(masks, kw['close_kernel_size'], kw['open_kernel_size'])
    masks = filter_by_shape(masks, min_area=kw['min_area'], min_solidity=0.1)
    dets = detect_blobs(masks, min_area=kw['min_area'])
    saved = (_Track.P_VAR, _Track.Q_VAR, _Track.R_VAR)
    _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = p_var, q_var, r_var
    try:
        tracks = track_blobs(dets, max_dist=kw['max_dist'], max_age=MAX_AGE)
    finally:
        _Track.P_VAR, _Track.Q_VAR, _Track.R_VAR = saved
    return dict(frames=frames, masks=masks, dets=dets, tracks=tracks, kw=kw)


def track(trial, max_frames=None):
    """Causal tracker. Track at t0: the detection whose centre lies in the GT box (largest if
    several). Followed by identity; once unseen for > MAX_AGE frames, re-acquired as the detection
    nearest the last predicted position. Returns per-frame foreground masks, detections and the
    followed chain {local t: (x, y)}."""
    h0 = trial['box_224'][3] - trial['box_224'][1]
    D = detect_and_track(trial['frames'][:max_frames] if max_frames else trial['frames'], trial['warmup'], h0,
                         trial.get('seq', TR.SEQ))
    frames, masks, dets, tracks, kw = D['frames'], D['masks'], D['dets'], D['tracks'], D['kw']

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


def readout_line(chain, t, span=BUFFER):
    """Position/velocity at t from the chain's own detections in the causal span [t-span+1, t]:
    OLS on x (the tracker's motion model is horizontal-only), mean y. Coasts on the last known
    position when the span holds fewer than two detections."""
    pts = sorted((k, chain[k]) for k in range(max(0, t - span + 1), t + 1) if k in chain)
    if len(pts) >= 2:
        ks = np.array([k for k, _ in pts], float)
        xs = np.array([p[0] for _, p in pts])
        vx, b = np.polyfit(ks, xs, 1)
        x = chain[t][0] if t in chain else vx * t + b
        return x, float(np.mean([p[1] for _, p in pts])), float(vx)
    k_last = max(k for k in chain if k <= t)
    return chain[k_last][0], chain[k_last][1], 0.0


def run_trial(trial, max_frames=None):
    """T4-1n over one window; fp16 autocast like B0/T1."""
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        s = time.perf_counter()
        P = track(trial, max_frames)
        masks, dets, kw = P['masks'], P['dets'], P['kw']
        T = len(P['frames'])
        out = np.zeros((T, TR.NATIVE_H, TR.NATIVE_W), bool)
        for t in range(T):
            if t == 0:
                pn, bn = trial['point_native'], trial['box_native']
            else:
                ax, ay, _ = readout_line(P['chain'], t)
                cx, cy = merged_center(dets[t], ax, ay, kw['merge_radius'])      # whole-person centre
                blob = restrict_to_nearby((masks[t] > 0).astype(np.uint8), masks[t], dets[t], cx, cy,
                                          kw['merge_radius']) > 0
                p224, b224 = prompt_from_blob(blob)
                pn = TR.to_native(*(p224 if p224 is not None else (cx, cy)))
                bn = box_to_native(b224) if b224 is not None else None
            out[t] = segment_native(trial, trial['frames'][t], pn, bn)
        return dict(masks=out, frames=np.array(trial['frames'][:T]), sec_per_frame=(time.perf_counter() - s) / T)


def select_trials(trials, limit=None, index=None):
    """Admissible trials; index picks exactly one (a SLURM array task), limit the first few."""
    ok = [t for t in trials if t['admissible']]
    return ok[index:index + 1] if index is not None else ok[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--trial-index', type=int, default=None, help='run one admissible trial (array task)')
    a = p.parse_args()
    os.makedirs(os.path.join(CACHE, METHOD), exist_ok=True)
    for t in select_trials(json.load(open(TR.OUT)), a.limit, a.trial_index):
        path = os.path.join(CACHE, METHOD, f"{t['id']}.npz")
        if os.path.exists(path) and not a.force:
            continue
        torch.cuda.reset_peak_memory_stats()
        r = run_trial(t)
        np.savez_compressed(path, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, **r)
        print(f"{t['id']}: {METHOD} {r['sec_per_frame'] * 1000:.0f} ms/frame", flush=True)


if __name__ == '__main__':
    main()
