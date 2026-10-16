"""SC1 + SC2 of docs/layer_sam2/training/plan_memory_layers.md (zero-shot, seq1 = selection sequence).
SC1 plumbing (review: exact): arm 'b0' (hooks installed, inactive) must give pixel-identical masks to 'plain' (no hooks).
C1 oracle (review): 'oracle-mem' writes past frames' pseudo-GT into the bank; if it cannot beat the memory-free
numbers, memory cannot deliver (caveat: the label is extent-leaning; report P and R).
SC2 value vs the frozen silhouette pseudo-GT (all labelled frames, 15 windows): video arms b0 / prompts / adapter, beside
the per-frame memory-free results (tracker-prompted SAM2 0.507, adapter v3.4 0.546 on every 4th frame).
SC3 recall over time: mean R per local frame index, first vs second half of the window.
    TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.video_check [ckpt] [n_windows]
"""
import json
import sys

import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR
from benchmark.nfo_vos.layers import fusion_data as FD
from benchmark.nfo_vos.layers.fusion_adapter import Fusion
from benchmark.nfo_vos.layers.video_adapter import per_frame_inputs, run_window
from benchmark.nfo_vos.run_sam2_video import build_predictor

ARMS = ('plain', 'b0', 'prompts', 'adapter', 'adapter-pre', 'oracle-mem')


def st(m, h):
    tp = (m & h).sum(); p, r = tp / max(m.sum(), 1), tp / max(h.sum(), 1)
    return np.array([tp / max((m | h).sum(), 1), p, r, 5 * p * r / max(4 * p + r, 1e-9)])


def main(ckpt='results/benchmark/rr/layers_fusion_adapter/token_seed0/fusion_it200.pt', n=15):
    pred = R.image_predictor()
    _o = FD._enc; FD._enc = lambda p, imgs, chunk=2: _o(p, imgs, 2)
    net = Fusion(token=True).cuda(); net.init_fallback(pred.model); net.load_state_dict(torch.load(ckpt)); net.eval()
    vp = build_predictor('b0')
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == 'seq1'][:int(n)]
    win, sc1, rt = {a: [] for a in ARMS}, [], {a: [] for a in ARMS}
    for t in T:
        L = np.load(f"{TR.RES}/labels_sil_frozen/{t['id']}.npz"); js = sorted(int(j) for j in L.files)
        inputs = per_frame_inputs(pred, net, t)
        labels = {j: L[str(j)] for j in js}
        M = {a: run_window(vp, t, inputs, a, labels=labels) for a in ARMS}
        sc1.append(int((M['b0'] != M['plain']).sum()))                     # exact: hooks inactive must change nothing
        for a in ARMS:
            rows = [st(M[a][j], L[str(j)]) for j in js]
            win[a].append(np.mean(rows, 0))
            h1, h2 = [r for j, r in zip(js, rows) if j < 20], [r for j, r in zip(js, rows) if j >= 20]
            rt[a].append((np.mean([r[2] for r in h1]), np.mean([r[2] for r in h2]), np.mean([r[1] for r in h1]), np.mean([r[1] for r in h2])))
        print(t['id'], 'SC1 differing px %d' % sc1[-1], ' '.join(f"{a} J {win[a][-1][0]:.3f}" for a in ARMS), flush=True)
    print(f'\nSC1 plumbing: pixels differing between hooks-inactive b0 and plain SAM2 video = {sum(sc1)} (pass: 0)')
    print(f'SC2 seq1 {len(T)} windows vs pseudo-GT (all labelled frames): J / P / R / F2')
    for a in ARMS:
        print(f'  video {a:8s} ' + ' / '.join(f'{v:.3f}' for v in np.mean(win[a], 0)))
    rng = np.random.default_rng(0)
    for a, b in (('prompts', 'b0'), ('adapter', 'prompts'), ('adapter-pre', 'prompts'), ('oracle-mem', 'prompts')):
        for i, m in ((0, 'J'), (2, 'R')):
            d = np.array(win[a])[:, i] - np.array(win[b])[:, i]; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
            print(f'    Δ{m} {a} − {b} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
    print('SC3 first half (frames < 20) → second half, R and P: ' + ' | '.join(
        f"{a} R {np.mean([r[0] for r in rt[a]]):.2f}→{np.mean([r[1] for r in rt[a]]):.2f} P {np.mean([r[2] for r in rt[a]]):.2f}→{np.mean([r[3] for r in rt[a]]):.2f}" for a in ARMS))
    print('Reference (memory-free, every 4th frame): tracker-prompted SAM2 0.507, adapter v3.4 it200 0.546')


if __name__ == '__main__':
    main(*sys.argv[1:])
