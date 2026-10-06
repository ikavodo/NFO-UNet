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
CKPT = os.path.abspath(os.environ.get('NFO_SAM2_CKPT', '../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt'))
CACHE = 'results/benchmark/pilot/masks'
VOTE_M = range(1, BUFFER + 1)


def methods(variant='point'):
    """'point': SAM2 prompted with box + interior point. 'box': the same box, no point (t4b, t4b-1)
    - SAM2 decides what the dominant object in the tracker's box is. Only the SAM2 variants exist
    for 'box'; the blob controls do not use a prompt."""
    if variant == 'box':
        return ['t4b', 't4b-1']
    if variant == 'ibox':
        return ['t4c-1']
    if variant == 'native':
        return ['t4-1n']
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


def box_to_native(box):
    """224 box edges -> native 800x600 edges through the padded square, the inverse of the
    GT's own scale_and_pad (gen_nfo_pseudo_masks.gt_to_native, in pixel units)."""
    s = max(TR.NATIVE_W, TR.NATIVE_H)
    px, py = (s - TR.NATIVE_W) / 2, (s - TR.NATIVE_H) / 2
    return np.array([box[0] / TR.SIZE * s - px, box[1] / TR.SIZE * s - py,
                     box[2] / TR.SIZE * s - px, box[3] / TR.SIZE * s - py], float)


def segment_native(trial, idx, point, box):
    """SAM2 image predictor on the FULL native frame, single-mask output: the setup that matches
    B0 at t0 with the identical prompt (diag_init_resolution.csv: J 0.700 vs 0.703)."""
    pred = gt_sam_gate._predictor
    img = cv2.imread(f"data/nfo_final/nfo_final/{trial['seq']}/{idx:05d}.jpg")[:, :, ::-1].copy()
    pred.set_image(img)
    masks, _, _ = pred.predict(point_coords=np.array([point]), point_labels=np.array([1]),
                               box=None if box is None else np.asarray(box), multimask_output=False)
    return masks[0] > 0.5


def integrated_box(aligned_blobs):
    """(x0, y0, x1, y1) of the union of the aligned buffer blobs, in crop coordinates: the
    integrated amodal extent (pilot: amodal box IoU 0.652 vs 0.546 for the frame-t blob box).
    None if no blob anywhere in the buffer."""
    u = aligned_blobs.any(0)
    if not u.any():
        return None
    ys, xs = np.nonzero(u)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def segment_prompted(img, point, box, fallback):
    """gt_sam_gate.segment_reference with an optional box (same predictor, same multimask argmax)."""
    pred = gt_sam_gate._predictor
    pred.set_image(np.stack([img] * 3, axis=-1))
    if point is None and box is None:
        point = fallback
    pt, lab = (np.array([point]), np.array([1])) if point is not None else (None, None)
    masks, scores, _ = pred.predict(point_coords=pt, point_labels=lab, box=box, multimask_output=True)
    return masks[np.argmax(scores)] > 0.5


def load(idx_range):
    return np.stack([cv2.imread(f'{TR.SEQ_DIR}/{i:05d}_or.jpg', 0) for i in idx_range])


def detect_and_track(frame_idx, warmup, h0):
    """MOG2 (warmed on the person-free warm-up range) -> morphology -> blobs -> Kalman/Hungarian
    tracks over CONTIGUOUS 224 frames. Shared by T4 (forward from t0) and composite.py (history
    up to t0). All scale-dependent parameters from the person height h0 (224 px)."""
    frames = load(frame_idx)
    w0, w1 = warmup
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
    return dict(frames=frames, masks=masks, dets=dets, tracks=tracks, kw=kw)


def track(trial, max_frames=None):
    """Causal tracker. Track at t0: the detection whose centre lies in the GT box (largest if
    several). Followed by identity; once unseen for > MAX_AGE frames, re-acquired as the detection
    nearest the last predicted position. Returns per-frame foreground masks, detections and the
    followed chain {local t: (x, y)}."""
    h0 = trial['box_224'][3] - trial['box_224'][1]
    D = detect_and_track(trial['frames'][:max_frames] if max_frames else trial['frames'], trial['warmup'], h0)
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
    position when the span holds fewer than two detections. span > BUFFER for a strided buffer:
    every contiguous detection in the span is used, not only the sampled frames."""
    pts = sorted((k, chain[k]) for k in range(max(0, t - span + 1), t + 1) if k in chain)
    if len(pts) >= 2:
        ks = np.array([k for k, _ in pts], float)
        xs = np.array([p[0] for _, p in pts])
        vx, b = np.polyfit(ks, xs, 1)
        x = chain[t][0] if t in chain else vx * t + b
        return x, float(np.mean([p[1] for _, p in pts])), float(vx)
    k_last = max(k for k in chain if k <= t)
    return chain[k_last][0], chain[k_last][1], 0.0


def run_trial(trial, max_frames=None, variant='point'):
    """fp16 autocast for every SAM2 call, the same precision B0/T1 run at (run_sam2_video.py)."""
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        return _run_trial(trial, max_frames, variant)


def _run_trial(trial, max_frames=None, variant='point'):
    use_base_plus()
    t_start = time.perf_counter()
    P = track(trial, max_frames)
    t_track = time.perf_counter() - t_start
    frames, masks, dets, kw = P['frames'], P['masks'], P['dets'], P['kw']
    T, H, W = frames.shape
    crop = int(np.clip(round(CROP_MULT * P['h0']), 60, min(H, W)))
    centre = (crop / 2, crop / 2)
    names = methods(variant)
    box_only = variant == 'box'
    seg_names = ('t4b', 't4b-1') if box_only else ('t4', 't4-1')
    blob_controls = variant == 'point'
    shape = (TR.NATIVE_H, TR.NATIVE_W) if variant == 'native' else (H, W)
    out = {m: np.zeros((T,) + shape, bool) for m in names}
    timing = {m: t_track for m in names}            # every variant pays the shared tracker
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
        if blob_controls:
            out['t4-3'][t] = blobs[-1]
            timing['t4-3'] += t_blob
        x0, y0 = int(cx - crop / 2), int(cy - crop / 2)               # crop_at's own origin
        point, box = prompt_in_crop(blobs[-1], x0, y0, gt=trial if t == 0 else None)
        if box_only and box is not None:
            point = None                      # no blob -> no box -> falls back to the centre point
        if variant == 'native':
            # T4-1 with ONLY the SAM2 input changed: native full frame instead of the 224 crop.
            # The tracker stays at 224; its prompt maps to native through the padded square.
            s = time.perf_counter()
            if t == 0:
                pn, bn = trial['point_native'], trial['box_native']
            else:
                p224, b224 = prompt_in_crop(blobs[-1], 0, 0)
                pn = TR.to_native(*(p224 if p224 is not None else (cx, cy)))
                bn = box_to_native(b224) if b224 is not None else None
            out['t4-1n'][t] = segment_native(trial, trial['frames'][t], pn, bn)
            timing['t4-1n'] += time.perf_counter() - s + t_blob
            continue
        if variant == 'ibox':
            # T4-1 with ONLY the box source changed: the integrated extent, not the frame-t blob
            s = time.perf_counter()
            ab = np.stack([crop_at(b.astype(np.uint8), x, cy, crop) > 0 for b, x in zip(blobs, xs)])
            if t > 0:                         # t0 keeps the shared GT prompt
                ibox = integrated_box(ab)
                box = ibox if ibox is not None else box
            out['t4c-1'][t] = back(segment_prompted(aligned[-1], point, box, centre))
            timing['t4c-1'] += time.perf_counter() - s + t_blob
            continue

        s = time.perf_counter()
        ref = fuse(aligned, method='median')
        S = segment_prompted(ref, point, box, centre)
        out[seg_names[0]][t] = back(visibility_within_mask(aligned, ref, S)[-1])
        timing[seg_names[0]] += time.perf_counter() - s + t_blob

        s = time.perf_counter()               # T4-1: reference = current frame; visibility = identity
        out[seg_names[1]][t] = back(segment_prompted(aligned[-1], point, box, centre))
        timing[seg_names[1]] += time.perf_counter() - s + t_blob
        if box_only:
            continue

        s = time.perf_counter()
        ab = np.stack([crop_at(b.astype(np.uint8), x, cy, crop) > 0 for b, x in zip(blobs, xs)])
        for m in VOTE_M:
            out[f't4-2_m{m}'][t] = back(visibility_within_mask(aligned, ref, blob_vote(ab, m))[-1])
        t_vote = (time.perf_counter() - s) / len(VOTE_M)
        for m in VOTE_M:
            timing[f't4-2_m{m}'] += t_blob + t_vote
    fr = np.array(trial['frames'][:T])
    return {m: dict(masks=out[m], frames=fr, sec_per_frame=timing[m] / T) for m in names}


def select_trials(trials, limit=None, index=None):
    """Admissible trials; index picks exactly one (a SLURM array task), limit the first few."""
    ok = [t for t in trials if t['admissible']]
    return ok[index:index + 1] if index is not None else ok[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--variant', choices=('point', 'box', 'ibox', 'native'), default='point')
    p.add_argument('--trial-index', type=int, default=None, help='run one admissible trial (array task)')
    a = p.parse_args()
    trials = select_trials(json.load(open(TR.OUT)), a.limit, a.trial_index)
    for m in methods(a.variant):
        os.makedirs(os.path.join(CACHE, m), exist_ok=True)
    for t in trials:
        paths = {m: os.path.join(CACHE, m, f"{t['id']}.npz") for m in methods(a.variant)}
        if all(os.path.exists(q) for q in paths.values()) and not a.force:
            continue
        torch.cuda.reset_peak_memory_stats()
        res = run_trial(t, variant=a.variant)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        for m, r in res.items():
            np.savez_compressed(paths[m], peak_mem_gb=peak, **r)
        print(f"{t['id']}: " + ', '.join(f"{m} {r['sec_per_frame'] * 1000:.0f} ms/frame"
                                        for m, r in res.items() if m in ('t4', 't4b')) + f", peak {peak:.2f} GB", flush=True)


if __name__ == '__main__':
    main()
