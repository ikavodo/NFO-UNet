"""Refactor guard: snapshot the outputs of the live code paths, then compare two snapshots bit for bit.

    TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tests.parity_snapshot save before.pkl
    ... refactor ...
    TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tests.parity_snapshot save after.pkl
    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.tests.parity_snapshot diff before.pkl after.pkl

Covers: adapter + tracker masks via fast_masks (3 windows), the hand-label test split (fusion_nfo.items), one synthetic
training clip (fusion_data.clip_raw + build, fixed seed), and the report.py tables (cached masks, CPU).
Needs a GPU and the data in data/ (see CLAUDE.md).
"""
import contextlib
import io
import json
import pickle
import sys

import numpy as np
import torch


def to_np(x):
    if torch.is_tensor(x):
        return x.float().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_np(v) for v in x]
    return x


def save(path):
    from benchmark.nfo_vos.eval import report
    from benchmark.nfo_vos.eval.dump_arms import CKPT
    from benchmark.nfo_vos.adapter import fusion_data as FD, fusion_nfo as FN
    from benchmark.nfo_vos.adapter.fusion_adapter import Fusion, image_predictor
    from benchmark.nfo_vos.adapter.inference import fast_masks
    snap = {}
    pred = image_predictor()
    net = Fusion().cuda(); net.init_fallback(pred.model); net.load_state_dict(torch.load(CKPT)); net.eval()
    trials = json.load(open('results/benchmark/rr/trials.json'))
    for i in (0, 25, 50):
        snap[f'adapter/{i}'] = fast_masks(pred, net, trials[i])[0]
    snap['trk/0'] = fast_masks(pred, None, trials[0])[0]
    for split in ('test', 'select'):
        snap[f'items/{split}'] = to_np(FN.items(split, pred))
    rng = np.random.default_rng(0)
    raw = FD.clip_raw(FD.specs('val')[0], rng)
    d = FD.build(pred, raw, max_targets=4, rng=np.random.default_rng(1))
    snap['build'] = to_np(d)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report.main()
    snap['report'] = buf.getvalue()
    pickle.dump(snap, open(path, 'wb'))
    print('saved', path, sorted(snap))


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')
    return a == b


def diff(p, q):
    a, b = pickle.load(open(p, 'rb')), pickle.load(open(q, 'rb'))
    bad = [k for k in sorted(a.keys() | b.keys()) if k not in a or k not in b or not same(a[k], b[k])]
    print('IDENTICAL' if not bad else f'DIFFERENT: {bad}')
    return not bad


if __name__ == '__main__':
    {'save': lambda: save(sys.argv[2]), 'diff': lambda: sys.exit(0 if diff(*sys.argv[2:4]) else 1)}[sys.argv[1]]()
