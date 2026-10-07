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


def run_trial(predictor, trial, max_frames=None, schedule=None, comp_dir=None):
    """One window. schedule (A/4, a4.py): times t at which composite C_t (comp_dir/<t>.jpg) is
    inserted just before raw frame R_t into the same memory bank; None or [] = plain run."""
    frames = trial['frames'][:max_frames] if max_frames else trial['frames']
    # staged sequence: (kind, t) per SAM2 frame; 'c' = composite C_t, 'r' = raw frame t
    seq = []
    for t in range(len(frames)):
        if schedule is not None and t in schedule:
            seq.append(('c', t))
        seq.append(('r', t))
    stage = tempfile.mkdtemp(prefix='nfo_vos_')
    try:
        for i, (kind, t) in enumerate(seq):         # SAM2's loader sorts by int(filename)
            src = (f'{comp_dir}/{t:02d}.jpg' if kind == 'c'
                   else f"{NATIVE_DIR}/{trial['seq']}/{frames[t]:05d}.jpg")
            os.symlink(os.path.abspath(src), os.path.join(stage, f'{i}.jpg'))
        torch.cuda.reset_peak_memory_stats()
        # SAM2's own per-frame confidence is computed in _forward_sam_heads and then discarded by
        # both builds; capture it with an instance-level wrapper (no change to either SAM2 build).
        # Logged: best candidate's predicted IoU and the object-score logit, per SAM-head call.
        calls = []
        orig = predictor._forward_sam_heads

        def capture(*args, **kwargs):
            out = orig(*args, **kwargs)
            calls.append((float(out[2].max()), float(out[6].max()) if out[6] is not None else float('nan')))
            return out
        predictor._forward_sam_heads = capture
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            state = predictor.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            # prompt the first staged frame, and raw t0 too when C_t0 precedes it
            # (A/4: the same prompt on C_t0 and R_t0)
            prompt_idx = [0] + ([1] if schedule is not None and seq[0] == ('c', 0) else [])
            prompt_calls = {}
            for pi in prompt_idx:
                predictor.add_new_points_or_box(state, frame_idx=pi, obj_id=0,
                                                points=np.array([trial['point_native']], np.float32),
                                                labels=np.array([1], np.int32),
                                                box=np.array(trial['box_native'], np.float32))
                prompt_calls[pi] = calls[-1] if calls else (np.nan, np.nan)
            masks = np.zeros((len(frames), state['video_height'], state['video_width']), bool)
            extent = np.zeros_like(masks)                # composite-frame masks (A/4 only)
            inserted = np.zeros(len(frames), bool)
            pred_iou = np.full(len(frames), np.nan, np.float32)
            obj_score = np.full(len(frames), np.nan, np.float32)
            seen = len(calls)
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for fi, _, logits in predictor.propagate_in_video(state):
                call = calls[-1] if len(calls) > seen else prompt_calls.get(fi, (np.nan, np.nan))
                seen = len(calls)
                kind, t = seq[fi]
                m = (logits[0, 0] > 0).cpu().numpy()
                if kind == 'r':
                    masks[t] = m
                    pred_iou[t], obj_score[t] = call
                elif kind == 'c':
                    extent[t], inserted[t] = m, True
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t_start) / len(frames)
        out = dict(masks=masks, frames=np.array(frames), sec_per_frame=dt, pred_iou=pred_iou,
                   obj_score=obj_score, peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        if schedule is not None:
            out.update(extent=extent, inserted=inserted)
        return out
    finally:
        predictor._forward_sam_heads = orig
        shutil.rmtree(stage)


def schedule_for(trial, mode, tau, baseline_npz):
    """A/4 insertion times for one trial: 'fixed' (every 4th t) or 'gated' (open-loop, from the
    baseline run's logged predicted IoU, density-capped). See a4.py."""
    from benchmark.nfo_vos import a4
    if mode == 'fixed':
        return a4.schedule_fixed(len(trial['frames']))
    return a4.schedule_gated(np.load(baseline_npz)['pred_iou'], tau)


def select_trials(trials, limit=None, index=None):
    """Admissible trials; index picks exactly one (a SLURM array task), limit the first few.
    Same contract as run_t4.select_trials, duplicated so this file stays importable in the
    SAMURAI venv without NFO-UNet's other dependencies."""
    ok = [t for t in trials if t['admissible']]
    return ok[index:index + 1] if index is not None else ok[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=sorted(CONFIGS), required=True)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--force', action='store_true')
    p.add_argument('--trial-index', type=int, default=None, help='run one admissible trial (array task)')
    p.add_argument('--a4', choices=('fixed', 'gated'), default=None, help='A/4 composite insertion')
    p.add_argument('--tau', type=float, default=0.6, help='gate: insert when baseline predicted IoU < tau')
    a = p.parse_args()
    trials = select_trials(json.load(open(TRIALS)), a.limit, a.trial_index)
    name = a.method if not a.a4 else a.method + ('-a4' if a.a4 == 'fixed' else f'-g{a.tau:g}')
    out_dir = os.path.join(CACHE, name)
    os.makedirs(out_dir, exist_ok=True)
    predictor = build_predictor(a.method)
    for t in trials:
        path = os.path.join(out_dir, f"{t['id']}.npz")
        if os.path.exists(path) and not a.force:
            continue
        if a.a4:
            sched = schedule_for(t, a.a4, a.tau, os.path.join(CACHE, a.method, f"{t['id']}.npz"))
            r = run_trial(predictor, t, schedule=sched, comp_dir=f"results/benchmark/{RUN}/composites_a4/{t['id']}")
        else:
            r = run_trial(predictor, t)
        np.savez_compressed(path, **r)
        print(f"{name} {t['id']}: {r['sec_per_frame'] * 1000:.0f} ms/frame, "
              f"peak {r['peak_mem_gb']:.2f} GB, last-frame area {r['masks'][-1].sum()}")


if __name__ == '__main__':
    main()
