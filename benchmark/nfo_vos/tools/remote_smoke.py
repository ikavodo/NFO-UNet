"""Cluster smoke test for adapter training (fusion_adapter.sbatch), ≈ 3 min: run it before any long job.

1. every path the training reads exists (GPJATK, occluders, foliage template, CanaTree, NFO frames + labels, ckpt);
2. GPU, bf16 support, sam2 import, checkpoint load;
3. ONE synthetic training clip end to end (clip_raw → build → Fusion → loss with the LOSS env → backward → step),
   timed;
4. one NFO selection item (seq1 brush frame, kernel-f16 tracker) — the real-frame validation path.
Prints PASS/FAIL per check and exits non-zero on any failure.

    PYTHONPATH=. python -m benchmark.nfo_vos.tools.remote_smoke
"""
import glob
import os
import sys
import time
import traceback

import numpy as np

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check('paths')
def _paths():
    from benchmark.nfo_vos.adapter.fusion_adapter import SAM2_CKPT
    from benchmark.nfo_vos.synth.gpjatk_foliage import GP
    need = {
        GP: lambda p: len(glob.glob(f'{p}/*')) > 0,
        'images/veronica111886-yakushima-2336735.jpg': os.path.isfile,
        'data/CanaTree100/images': lambda p: len(glob.glob(f'{p}/*.png')) >= 50,
        SAM2_CKPT: os.path.isfile,
        'results/benchmark/rr/trials.json': os.path.isfile,
    }
    for s in ('seq1', 'seq2', 'seq3', 'seq4'):
        need[f'data/nfo_final/nfo_final/{s}'] = lambda p: len(glob.glob(f'{p}/*.jpg')) > 1000
        need[f'data/nfo_processed/{s}_gt/groundtruth.txt'] = os.path.isfile
    need['data/nfo_processed/seq1_gt/01154_handmask_raw.png'] = os.path.isfile
    need['data/nfo_processed/seq2_gt/00052_handAv.png'] = os.path.isfile
    n_gp = sum(1 for _ in glob.iglob(f'{GP}/**/*', recursive=True))
    bad = [p for p, ok in need.items() if not (os.path.exists(p) and ok(p))]
    print(f'  GPJATK files: {n_gp} (local reference: 167,918 incl. dirs ≈ files)')
    if n_gp < 160000:
        bad.append(f'GPJATK incomplete ({n_gp} files)')
    assert not bad, 'missing / incomplete: ' + '; '.join(bad)


@check('gpu + sam2')
def _gpu():
    import torch
    assert torch.cuda.is_available(), 'no CUDA'
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)   # default True on Pascal (emulated): misleading
    print(f'  GPU {name}, sm {cc[0]}.{cc[1]}, native bf16 {bf16}, torch {torch.__version__}, '
          f'mem {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB')
    assert cc >= (7, 5), f'sm {cc[0]}.{cc[1]}: Pascal or older (no usable fp16/bf16 tensor cores)'
    assert bf16 or os.environ.get('AMP') == 'fp16', f'sm {cc[0]}.{cc[1]} has no native bf16: run with AMP=fp16'
    assert torch.cuda.get_device_properties(0).total_memory >= 15 * 2**30, 'training needs ~16 GB'
    torch.zeros(1, device='cuda').sum().item()                 # "no kernel image" fails here
    image_predictor()


@check('one training step (synthetic)')
def _train():
    import torch
    from benchmark.nfo_vos.adapter import fusion_adapter as FA, fusion_data as FD
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    from benchmark.nfo_vos.synth import gpjatk_foliage as GF
    pred = image_predictor()
    GF.set_bg_pool(FD.BG_SPLIT['train'])
    t0 = time.time()
    raw = FD.clip_raw(FD.specs('train')[0], np.random.default_rng(0))
    t1 = time.time()
    f, meta = FD.build(pred, raw, max_targets=8, rng=np.random.default_rng(0))
    net = FA.Fusion(token=True).cuda(); net.init_fallback(pred.model)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    with torch.autocast('cuda', dtype=FD.AMP):
        o = net(f, sam=pred.model)
        loss = FA.loss_fn(FA.decode(pred.model, o, token=o.get('token')).float(), f['V'], f.get('W'), **FA.LOSS_CFG)
    loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f'  LOSS={FA.LOSS} loss {loss.item():.4f}; clip synthesis {t1 - t0:.1f} s, build+step {time.time() - t1:.1f} s '
          f'({len(meta["ts"])} targets), peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB')
    assert np.isfinite(loss.item())


@check('NFO selection item (seq1)')
def _nfo():
    from benchmark.nfo_vos.adapter.fusion_nfo import items
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    its = items('select', image_predictor())
    print(f'  seq1 selection items: {len(its)}')
    assert len(its) > 0


def main():
    fails = 0
    for name, fn in CHECKS:
        t = time.time()
        try:
            fn()
            print(f'PASS {name} ({time.time() - t:.0f} s)', flush=True)
        except Exception as e:
            fails += 1
            print(f'FAIL {name}: {e}', flush=True)
            traceback.print_exc(limit=3)
    print('SMOKE', 'OK' if not fails else f'{fails} FAILED')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
