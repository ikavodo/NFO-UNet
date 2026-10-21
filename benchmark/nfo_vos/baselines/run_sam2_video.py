"""Baselines B0 (vanilla SAM2.1 b+ video predictor) and T1 (SAMURAI) on the NFO windows (trials.json).

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.baselines.run_sam2_video --method t1 [--trial-index N]
    (B0 needs the official sam2 package, T1 the SAMURAI repo's sam2 fork; see run_sam2_video.sbatch)

Deliberately imports nothing from this repo beyond the stdlib/numpy/torch/cv2, so the same file runs in both
environments. Both methods: native 800x600 frames, the same GT box + point p* at t0 (add_new_points_or_box), strictly
forward propagation, fp16 autocast and CPU offload exactly as SAMURAI's scripts/main_inference.py:76-78 (@76ba195), so
the predictor/config is the only difference between B0 and T1. Output: masks/<method><NFO_TAG>/<trial>.npz
(the reported b0-c2 / t1-c2 are NFO_TAG=-c2, the cluster rerun).
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import torch

CKPT = os.path.abspath(os.environ.get('NFO_SAM2_CKPT', '../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt'))
CONFIGS = {'b0': 'configs/sam2.1/sam2.1_hiera_b+.yaml',          # sam2 1.1.0 package
           't1': 'configs/samurai/sam2.1_hiera_b+.yaml'}         # samurai repo's sam2 fork
NATIVE_DIR = 'data/nfo_final/nfo_final'
RUN = os.environ.get('NFO_RUN', 'rr')            # same convention as trials.RUN (kept import-free)
TRIALS = f'results/benchmark/{RUN}/trials.json'
CACHE = f'results/benchmark/{RUN}/masks'


def build_predictor(method):
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(CONFIGS[method], CKPT, device='cuda')


def run_trial(predictor, trial, max_frames=None):
    """One window: masks [T, 600, 800] bool, frames, seconds per frame, peak GPU memory."""
    frames = trial['frames'][:max_frames] if max_frames else trial['frames']
    stage = tempfile.mkdtemp(prefix='nfo_vos_')
    try:
        for i, k in enumerate(frames):              # SAM2's loader sorts by int(filename)
            os.symlink(os.path.abspath(f"{NATIVE_DIR}/{trial['seq']}/{k:05d}.jpg"), os.path.join(stage, f'{i}.jpg'))
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            state = predictor.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0,
                                            points=np.array([trial['point_native']], np.float32),
                                            labels=np.array([1], np.int32), box=np.array(trial['box_native'], np.float32))
            masks = np.zeros((len(frames), state['video_height'], state['video_width']), bool)
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for fi, _, logits in predictor.propagate_in_video(state):
                masks[fi] = (logits[0, 0] > 0).cpu().numpy()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t_start) / len(frames)
        return dict(masks=masks, frames=np.array(frames), sec_per_frame=dt,
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    finally:
        shutil.rmtree(stage)


def select_trials(trials, limit=None, index=None):
    """Admissible trials; index picks exactly one (a SLURM array task), limit the first few."""
    ok = [t for t in trials if t['admissible']]
    return ok[index:index + 1] if index is not None else ok[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=sorted(CONFIGS), required=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--trial-index', type=int, default=None, help='run one admissible trial (array task)')
    a = p.parse_args()
    name = a.method + os.environ.get('NFO_TAG', '')            # e.g. '-c2': the cluster rerun
    out_dir = os.path.join(CACHE, name)
    os.makedirs(out_dir, exist_ok=True)
    predictor = build_predictor(a.method)
    for t in select_trials(json.load(open(TRIALS)), a.limit, a.trial_index):
        path = os.path.join(out_dir, f"{t['id']}.npz")
        if os.path.exists(path) and not a.force:
            continue
        r = run_trial(predictor, t)
        np.savez_compressed(path, **r)
        print(f"{name} {t['id']}: {r['sec_per_frame'] * 1000:.0f} ms/frame, "
              f"peak {r['peak_mem_gb']:.2f} GB, last-frame area {r['masks'][-1].sum()}")


if __name__ == '__main__':
    main()
