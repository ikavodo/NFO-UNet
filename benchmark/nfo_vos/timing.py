"""Per-frame runtime: per-frame adapter pipeline vs SAM2 video (B0) / SAMURAI (T1), one 40-frame seq2 window, same GPU.

Components (median over repeats, CUDA-synchronised): one SAM2.1-b+ image-encoder pass at 1024²; the pixel composite +
evidence mass (CPU, 6 frames); the adapter forward; the frozen prompt encoder + mask decoder. End to end: the code paths
used for the benchmark masks (video_adapter.per_frame_inputs + dump_arms.per_frame_masks; video_adapter.run_window),
not optimised: per-frame inputs re-encode the empty scene every frame, run_window reads JPEGs from disk.

    TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.timing
"""
import json
import time

import numpy as np
import torch


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def med(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = tick(); fn(); ts.append(tick() - t0)
    return 1000 * float(np.median(ts))


def main():
    from benchmark.nfo_vos import run_t4 as R, velocity as V
    from benchmark.nfo_vos.downstream.dump_arms import CKPT, per_frame_masks
    from benchmark.nfo_vos.layers import fusion_data as FD
    from benchmark.nfo_vos.layers.fusion_adapter import Fusion, decode
    from benchmark.nfo_vos.layers.prompt_oracle import composite_and_mass
    from benchmark.nfo_vos.layers.video_adapter import per_frame_inputs, run_window
    from benchmark.nfo_vos.run_sam2_video import build_predictor
    print('GPU', torch.cuda.get_device_name(0), 'torch', torch.__version__)
    pred = R.image_predictor()
    net = Fusion(token=True).cuda(); net.init_fallback(pred.model); net.load_state_dict(torch.load(CKPT)); net.eval()
    t = next(x for x in json.load(open('results/benchmark/rr/trials.json')) if x['seq'] == 'seq2')
    T = len(t['frames'])
    img = np.stack([V.native(t['frames'][10], t['seq'])] * 3, -1)
    res = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        res['encoder pass (1 image)'] = med(lambda: pred.set_image(img))
        frames = [V.native(t['frames'][i], t['seq']).astype(np.float32) for i in range(6)]
        bg = frames[0]
        res['pixel composite + mass (CPU)'] = med(lambda: composite_and_mass(frames, [np.float32([[1, 0, 2], [0, 1, 0]])] * 6, bg, z0='adaptive'))
        d = per_frame_inputs(pred, net, t)[10]
        f = FD.encode(pred, [d['it']])
        res['encode X + C + B (3 passes)'] = med(lambda: FD.encode(pred, [d['it']]))
        res['adapter forward'] = med(lambda: net(f, sam=pred.model))
        o = net(f, sam=pred.model)
        res['prompt enc + mask decoder'] = med(lambda: decode(pred.model, o, token=o['token']))
    t0 = tick()
    inputs = per_frame_inputs(pred, net, t)
    per_frame_masks(pred, net, t, inputs, 'adapter-pf')
    res['END TO END ours (per frame, as benchmarked)'] = 1000 * (tick() - t0) / T
    for arm, mode in (('b0', 'plain'), ('t1', 'plain')):
        try:
            vp = build_predictor(arm)
        except Exception as e:
            print(arm, 'skipped:', e); continue
        run_window(vp, t, inputs, mode)                          # warm-up
        t0 = tick(); run_window(vp, t, inputs, mode)
        res[f'END TO END {arm} video (per frame, incl. JPEG load)'] = 1000 * (tick() - t0) / T
        del vp; torch.cuda.empty_cache()
    for k, v in res.items():
        print(f'{k:48s} {v:8.1f} ms')


if __name__ == '__main__':
    main()
