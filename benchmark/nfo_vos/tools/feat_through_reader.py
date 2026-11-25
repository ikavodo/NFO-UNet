"""Distil the one-pass feature integral THROUGH the frozen pixel reader (findings §10.x item 23; advisor route 1, after
Yang et al., "Knowledge Distillation via Softmax Regression Representation Learning", ICLR 2021: the student is trained
through the teacher's frozen head).

Why: every feature-route Fusion retrained on synthetic ignored Ĉ (items 9, 16, 17, 21), and synthetic data does not reward
either integral, so a retrained reader's sensitivity J_C to C is arbitrary. The pixel adapter's Fusion f_pf (token_seed0
it200) is the one reader whose use of C transfers to NFO. Here f_pf stays frozen and the aggregator (BigS16: α-mixed
C8/C4 + mass a + the big s16 student) is trained so that f_pf's OUTPUT on the student's (C, a) matches its output on the
pixel route's (enc(C), a_pix): global cosine to enc(C) weighs all feature directions equally, f_pf reads only some.

Data: NFO seq1 crossings s4–s6 (every 2nd history frame), log on s7 (every 4th); s0–s3 untouched (the s0 screen judges,
seq2–4 s0 primary). Unlabelled use only. Loss (fp32 on bf16-autocast forwards, as inference): BCE(decoder logits_s,
σ(logits_t)) + Σ_ℓ ‖o_s[Xℓ] − o_t[Xℓ]‖² / N_ℓ + ‖tok_s − tok_t‖² / N_tok + 0.2·anchor, N_* = dataset means of the
teacher's residual / token-correction energy; anchor = mean_ℓ (1 − cos(C_s^ℓ, enc(C)^ℓ)) (s16 centred by mu16) + MSE(a_s,
a_pix), against adversarial inputs to the frozen reader (design review, 2026-11-25). AdamW, lr 1e-4 (α head, mass) /
3e-5 (big s16 net), cosine, 1000 steps (fixed in advance), 2 frames per step, seed 0.

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_through_reader
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.adapter import inference as INF
from benchmark.nfo_vos.adapter.feat_integral import ccos
from benchmark.nfo_vos.adapter.fusion_adapter import decode, image_predictor
from benchmark.nfo_vos.eval.dump_arms import CKPT, load_net
from benchmark.nfo_vos.tools.feat_distil import frames

A = 'results/benchmark/rr/layers_fusion_adapter/feat_bigpf'
S = ('16', '8', '4')
cpu = lambda x: x.detach().half().cpu()
PIXEL_A = bool(os.environ.get('PIXEL_A'))                  # v2: the reader gets the pixel gate's mass a (no student a)
TRAIN = [int(i) for i in os.environ.get('TRAIN_IDX', '4,5,6').split(',')]
EVERY = int(os.environ.get('EVERY', '2'))
TAG = os.environ.get('TAG', 'trained_it1000')


@torch.no_grad()
def collect(pred, feat, pix, t, every):
    """Per kept frame j: the aggregator's inputs (shifted keys K, validity, shifts), frame / empty-scene features, tracker
    prompts (feature run), and the pixel route's enc(C) and mass a (pixel run, same frame)."""
    Xn, bg, g = frames(t); d = {}

    def hf(j, f):
        if j % every == 0 and '_agg' in f:
            K, v, dx = f['_agg']
            d[j] = dict(K={l: cpu(K[l]) for l in S}, v=v.detach().cpu(), dx=dx.detach().cpu(),
                        X={l: cpu(f[f'X{l}']) for l in S}, B={l: cpu(f[f'B{l}']) for l in S},
                        trk=f['trk'].cpu(), box=f['box_trk'].cpu(), Crun=f['C16'].float().cpu())
    INF.masks_from_arrays(pred, feat, Xn, bg, g, probe=hf)

    def hp(j, f):
        if j in d:
            assert torch.allclose(f['trk'].cpu(), d[j]['trk']) and torch.allclose(f['X16'].half().cpu().float(), d[j]['X']['16'].float())
            d[j]['Cp'] = {l: cpu(f[f'C{l}']) for l in S}; d[j]['ap'] = cpu(f['a'])
    INF.masks_from_arrays(pred, pix, Xn, bg, g, probe=hp)
    return [x for x in d.values() if 'Cp' in x]


def base(x):
    f = {f'{k}{l}': x[k][l].cuda().float() for k in ('X', 'B') for l in S}
    f['trk'], f['box_trk'] = x['trk'].cuda(), x['box'].cuda()
    f['pt'], f['box'] = f['trk'], f['box_trk']
    return f


def student(agg, x):
    K = {l: x['K'][l].cuda().float() for l in S}; dx = x['dx'].cuda()
    return agg(K, x['v'].cuda(), {l: x['X'][l].cuda().float() for l in S}, {l: x['B'][l].cuda().float() for l in S},
               torch.arange(6, 0, -1, device='cuda')[None], phase=dx / 16 - torch.round(dx / 16))


def main():
    torch.manual_seed(0)
    pred = image_predictor(); sam = pred.model
    feat = load_net(pred, f'{A}/plug.pt'); fus = load_net(pred, CKPT)            # same f_pf weights; fus has no agg
    agg = feat.agg
    for p in fus.parameters():
        p.requires_grad_(False)
    seq1 = [t for t in json.load(open(TR.OUT)) if t['seq'] == 'seq1']
    assert 0 not in TRAIN and 7 not in TRAIN                                       # s0 = screen, s7 = log
    tr = sum([collect(pred, feat, fus, seq1[i], EVERY) for i in TRAIN], []); va = collect(pred, feat, fus, seq1[7], 4)
    del feat.agg
    print(f'frames: train {len(tr)} (seq1 {TRAIN}, every {EVERY}), log {len(va)} (seq1 s7); pixel a {PIXEL_A}', flush=True)
    agg.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):                # parity: stored inputs → in-run Ĉ16
        for x in (tr[0], va[0]):
            c = student(agg, x)['C16'].float().cpu()
            print(f'parity C16 rel err {((c - x["Crun"]).norm() / x["Crun"].norm()).item():.2e}', flush=True)

    def outs(x, C=None, a=None):
        f = base(x)
        if C is None:
            f.update({f'C{l}': x['Cp'][l].cuda().float() for l in S}, a=x['ap'].cuda().float())
        else:
            f.update({f'C{l}': C[f'C{l}'].float() for l in S}, a=a.float())
        o = fus(f, sam=sam)
        return o, decode(sam, o, token=o['token']), f

    N = {l: [] for l in S}; N['tok'] = []                                           # dataset-mean normalisers
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for x in tr[::4]:
            o, _, f = outs(x)
            f0 = dict(f, a=torch.zeros_like(f['a']), **{f'C{l}': f[f'B{l}'] for l in S})
            tok0 = fus(f0, sam=sam)['token']                                          # a = 0 → w = 0 → token = sp
            for l in S:
                N[l].append((o[f'X{l}'].float() - f[f'X{l}']).pow(2).mean().item())
            N['tok'].append((o['token'].float() - tok0.float()).pow(2).mean().item())
    N = {k: max(np.mean(v), 1e-12) for k, v in N.items()}
    print('normalisers ' + ' '.join(f'{k} {v:.3e}' for k, v in N.items()), flush=True)
    mu = agg.mu16.float()

    def terms(x):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ot, lt, _ = outs(x)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            oa = student(agg, x)
            os_, ls, _ = outs(x, oa, x['ap'].cuda().float() if PIXEL_A else oa['a'])
        T = {'logit': F.binary_cross_entropy_with_logits(ls.float(), torch.sigmoid(lt.float()))
                      - F.binary_cross_entropy_with_logits(lt.float(), torch.sigmoid(lt.float()))}
        for l in S:
            T[f'r{l}'] = (os_[f'X{l}'].float() - ot[f'X{l}'].float()).pow(2).mean() / N[l]
        T['tok'] = (os_['token'].float() - ot['token'].float()).pow(2).mean() / N['tok']
        T['anchor'] = torch.stack([(1 - ccos(oa[f'C{l}'].float(), x['Cp'][l].cuda().float(), mu if l == '16' else None)).mean()
                                   for l in S]).mean() + (0 if PIXEL_A else (oa['a'].float() - x['ap'].cuda().float()).pow(2).mean())
        T['loss'] = T['logit'] + T['r16'] + T['r8'] + T['r4'] + T['tok'] + 0.2 * T['anchor']
        return T

    def log(it):
        agg.eval()
        with torch.no_grad():
            m = {}
            for x in va:
                for k, v in terms(x).items():
                    m.setdefault(k, []).append(v.item())
        print(f'it {it} [seq1 s7] ' + ' '.join(f'{k} {np.mean(v):.4f}' for k, v in m.items()), flush=True)
        agg.train()

    big = [p for n, p in agg.named_parameters() if n.startswith('net.')]
    rest = [p for n, p in agg.named_parameters() if not n.startswith('net.')]
    opt = torch.optim.AdamW([{'params': rest, 'lr': 1e-4}, {'params': big, 'lr': 3e-5}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1000)
    rng = np.random.default_rng(0)
    log(0); agg.train()
    for it in range(1, 1001):
        opt.zero_grad()
        for i in rng.choice(len(tr), 2, replace=False):
            (terms(tr[i])['loss'] / 2).backward()
        opt.step(); sched.step()
        if it % 100 == 0:
            log(it)
    torch.save({'fusion': fus.state_dict(), 'agg': agg.state_dict(), 'pixel_a': PIXEL_A}, f'{A}/{TAG}.pt')
    print('saved', f'{A}/{TAG}.pt', flush=True)


if __name__ == '__main__':
    main()
