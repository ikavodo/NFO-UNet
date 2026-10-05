"""B0 (vanilla SAM2.1 video predictor) and T1 (SAMURAI) on the pilot trials (spec §4).

    ../master_thesis/.venv/bin/python -m benchmark.nfo_vos.run_sam2_video --method b0
    ../samurai/.venv/bin/python      -m benchmark.nfo_vos.run_sam2_video --method t1

Deliberately imports nothing from NFO-UNet beyond the stdlib/numpy/torch, so the same file runs
in both venvs. Both methods: native 800x600 frames, the same box + p* prompt at t0
(add_new_points_or_box), strictly forward propagation, fp16 autocast and CPU offload exactly as
SAMURAI's own scripts/main_inference.py:76-78 (@76ba195), so the predictor/config is the only
difference between B0 and T1. Masks are saved at native resolution; score.py maps them to 224.
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import torch

CKPT = os.path.abspath('../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt')
CONFIGS = {'b0': 'configs/sam2.1/sam2.1_hiera_b+.yaml',          # sam2 1.1.0 package
           't1': 'configs/samurai/sam2.1_hiera_b+.yaml'}         # samurai repo's sam2 fork
NATIVE_DIR = 'data/nfo_final/nfo_final'
TRIALS = 'results/benchmark/pilot/trials.json'
CACHE = 'results/benchmark/pilot/masks'


def build_predictor(method):
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor(CONFIGS[method], CKPT, device='cuda')


def run_trial(predictor, trial, max_frames=None, prepend=None):
    """prepend: path of a composite frame (composite.py) staged as frame 0 and prompted there
    instead of raw t0, so it becomes the conditioning memory; its own output is dropped."""
    frames = trial['frames'][:max_frames] if max_frames else trial['frames']
    off = 1 if prepend else 0
    stage = tempfile.mkdtemp(prefix='nfo_vos_')
    try:
        if prepend:
            os.symlink(os.path.abspath(prepend), os.path.join(stage, '0.jpg'))
        for local, raw in enumerate(frames):        # SAM2's loader sorts by int(filename)
            os.symlink(os.path.abspath(f"{NATIVE_DIR}/{trial['seq']}/{raw:05d}.jpg"),
                       os.path.join(stage, f'{local + off}.jpg'))
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            state = predictor.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=0,
                                            points=np.array([trial['point_native']], np.float32),
                                            labels=np.array([1], np.int32),
                                            box=np.array(trial['box_native'], np.float32))
            masks = np.zeros((len(frames), state['video_height'], state['video_width']), bool)
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for fi, _, logits in predictor.propagate_in_video(state):
                if fi >= off:
                    masks[fi - off] = (logits[0, 0] > 0).cpu().numpy()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t_start) / len(frames)
        return dict(masks=masks, frames=np.array(frames), sec_per_frame=dt,
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    finally:
        shutil.rmtree(stage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=sorted(CONFIGS), required=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--prepend-dir', default=None, help='composites dir -> method <name>-cp')
    a = p.parse_args()
    trials = [t for t in json.load(open(TRIALS)) if t['admissible']][:a.limit]
    name = a.method + ('-cp' if a.prepend_dir else '')
    out_dir = os.path.join(CACHE, name)
    os.makedirs(out_dir, exist_ok=True)
    predictor = build_predictor(a.method)
    for t in trials:
        path = os.path.join(out_dir, f"{t['id']}.npz")
        if os.path.exists(path) and not a.force:
            continue
        comp = os.path.join(a.prepend_dir, f"{t['id']}.jpg") if a.prepend_dir else None
        if comp and not os.path.exists(comp):
            print(f"{name} {t['id']}: no composite, skipped"); continue
        r = run_trial(predictor, t, prepend=comp)
        np.savez_compressed(path, **r)
        print(f"{name} {t['id']}: {r['sec_per_frame'] * 1000:.0f} ms/frame, "
              f"peak {r['peak_mem_gb']:.2f} GB, frame-50 area {r['masks'][-1].sum()}")


if __name__ == '__main__':
    main()
