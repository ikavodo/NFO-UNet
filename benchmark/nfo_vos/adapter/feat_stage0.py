"""Stage 0 + G1 (plan docs/superpowers/plans/2026-11-19-layer-sam-feature-integral.md, Task 5).

Stage 0: train the feature-integral aggregator (feat_integral.FeatIntegral) ALONE on the clean-feature loss
L_C = mean_s mean_{Ω} [1 − cos(Ĉ^s, T^s)] (T = features of frame t without the occluder, same noise and sway; Ω = hidden
person cells visible in some aligned history frame). No Fusion, no decoder.

G1 (pre-registered): on 30 held-out synthetic val clips, mean cos(·, T) on Ω per stride for
  learned  Ĉ after stage 0 (best val L_C)        fixed  Ĉ with recency τ = 3.5 × feature-change gate (no learning)
  pixel    enc(pixel integral image)  (today)    F_t    the frame's own features (do nothing)
  floor    cos(F_t, T) on VISIBLE person cells (what a perfect reconstruction can reach given noise / sway)
95% clip-bootstrap CIs of learned − fixed and learned − F_t at stride 16. Verdict: KILL if learned ≤ F_t at s16;
LEARNED>FIXED if the CI of learned − fixed is above 0; else FIXED>=LEARNED.

    PYTHONPATH=. ../samurai/.venv/bin/python -u -m benchmark.nfo_vos.adapter.feat_stage0 [steps] [seed]
"""
import json
import os
import sys
import time

import numpy as np
import torch

from benchmark.nfo_vos.adapter.feat_integral import FeatIntegral, STRIDES, ccos, feat_loss
from benchmark.nfo_vos.adapter.fusion_adapter import _ident, image_predictor
from benchmark.nfo_vos.adapter.fusion_data import AMP, Clips, build

OUT = 'results/benchmark/rr/feat_integral'
IMG = 'images/benchmark/feat_integral'
ARMS = ('learned', 'fixed', 'pixel', 'F_t')


def _agg_in(f):
    return ({s: f[f'K{s}'] for s in STRIDES}, f['Kvalid16'], {s: f[f'X{s}'] for s in STRIDES},
            {s: f[f'B{s}'] for s in STRIDES}, f['age'])


@torch.no_grad()
def evaluate(pred, agg, val, max_targets=16):
    """Per val clip: mean cos(·, T) on Ω per stride and arm (+ floor), and the learned val L_C."""
    rows, lc = [], []
    for k, raw in enumerate(val):
        f, _ = build(pred, raw, max_targets=max_targets, rng=np.random.default_rng(k), feat=True, with_pixel=True)
        mu = f['mu16']
        with torch.autocast('cuda', dtype=AMP):
            ol = agg(*_agg_in(f))
            of = agg(*_agg_in(f), fixed=True)
        T = {s: f[f'T{s}'] for s in STRIDES}
        Om = {s: f[f'Om{s}'] for s in STRIDES}
        lc.append(float(feat_loss(ol, T, Om, mu)))
        src = {'learned': {s: ol[f'C{s}'] for s in STRIDES}, 'fixed': {s: of[f'C{s}'] for s in STRIDES},
               'pixel': {s: f[f'P{s}'] for s in STRIDES}, 'F_t': {s: f[f'X{s}'] for s in STRIDES}}
        r = {'clip': k, 'n_om16': int(Om['16'].sum())}
        for s in STRIDES:
            m = Om[s][:, 0]
            for a in ARMS:
                r[f'{a}{s}'] = float(ccos(src[a][s].float(), T[s], mu if s == '16' else None)[m].mean()) if m.any() else np.nan
        v = f['Vis16'][:, 0]
        r['floor16'] = float(ccos(f['X16'], T['16'], mu)[v].mean()) if v.any() else np.nan
        rows.append(r)
        if k == 0:
            _trace(raw, f, ol, of, T, mu)
    return rows, float(np.mean(lc))


def _trace(raw, f, ol, of, T, mu):
    """Visual trace: one target — frame, occluder-free target, Ω, and cos(·, T) maps at s16 (F_t / fixed / learned)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    i = int(torch.argmax(f['Om16'].flatten(1).sum(1)))
    cm = lambda x: ccos(x[i:i + 1].float(), T['16'][i:i + 1], mu)[0].cpu().numpy()
    fig, ax = plt.subplots(1, 6, figsize=(12, 2.4))
    ims = [(f['Om16'][i, 0].float().cpu().numpy(), 'Ω (hidden, recoverable) s16'), (cm(f['X16']), 'cos(F_t, T)'),
           (cm(of['C16']), 'cos(Ĉ fixed, T)'), (cm(ol['C16']), 'cos(Ĉ learned, T)'),
           (ol['a'][i].float().cpu().numpy(), 'a = 1 − α∅ (learned)'), (of['a'][i].float().cpu().numpy(), 'a (fixed)')]
    for a, (im, ttl) in zip(ax, ims):
        h = a.imshow(im, vmin=-0.2 if 'cos' in ttl else 0, vmax=1); a.set_title(ttl, fontsize=8); a.axis('off')
    fig.colorbar(h, ax=ax, fraction=0.02)
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(f'{IMG}/g1_trace.png', dpi=80, bbox_inches='tight'); plt.close(fig)


def _boot(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main(steps=400, seed=0, eval_every=100, n_val=30, max_targets=32):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(OUT, exist_ok=True)
    pred = image_predictor()
    for p in pred.model.parameters():
        p.requires_grad_(False)
    agg = FeatIntegral(pred.model.no_mem_embed).cuda()
    opt = torch.optim.AdamW(agg.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP == torch.float16)
    dl = iter(torch.utils.data.DataLoader(Clips('train', seed), batch_size=None, num_workers=6, prefetch_factor=2,
                                          collate_fn=_ident, persistent_workers=True))
    val = list(torch.utils.data.DataLoader(Clips('val', n_val=n_val), batch_size=None, num_workers=6, collate_fn=_ident))
    rng, t0 = np.random.default_rng(seed), time.time()
    rows0, lc0 = evaluate(pred, agg, val)
    print(f'params {sum(p.numel() for p in agg.parameters())}; val clips {len(val)}; it 0 val L_C {lc0:.4f}', flush=True)
    best, log = (lc0, {k: v.clone() for k, v in agg.state_dict().items()}, 0), [dict(it=0, val_LC=lc0)]
    for it in range(1, steps + 1):
        f, _ = build(pred, next(dl), max_targets=max_targets, rng=rng, feat=True)
        with torch.autocast('cuda', dtype=AMP):
            o = agg(*_agg_in(f))
            loss = feat_loss(o, {s: f[f'T{s}'] for s in STRIDES}, {s: f[f'Om{s}'] for s in STRIDES}, f['mu16'])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 25 == 0:
            print(f'it {it} L_C {loss.item():.4f} {(time.time() - t0) / it:.2f} s/it', flush=True)
        if it % eval_every == 0 or it == steps:
            _, lc = evaluate(pred, agg, val)
            log.append(dict(it=it, val_LC=lc))
            print(f'VAL it {it}: L_C {lc:.4f} (step 0: {lc0:.4f})', flush=True)
            if lc < best[0]:
                best = (lc, {k: v.clone() for k, v in agg.state_dict().items()}, it)
    agg.load_state_dict(best[1])
    torch.save(agg.state_dict(), f'{OUT}/stage0_seed{seed}.pt')
    rows, lc = evaluate(pred, agg, val)
    print(f'\nG1 (best it {best[2]}, val L_C {lc:.4f}; {len(rows)} val clips) — mean cos(·, T) on Ω, per clip then averaged')
    print(f"{'s':>3} | " + ' '.join(f'{a:>8}' for a in ARMS) + '  | floor (visible)')
    res = {'best_it': best[2], 'val_LC': lc, 'log': log, 'rows': rows}
    for s in STRIDES:
        means = {a: float(np.nanmean([r[f'{a}{s}'] for r in rows])) for a in ARMS}
        res[f'mean{s}'] = means
        fl = f"  | {np.nanmean([r['floor16'] for r in rows]):.3f}" if s == '16' else ''
        print(f'{s:>3} | ' + ' '.join(f'{means[a]:8.3f}' for a in ARMS) + fl)
    d_lf = _boot([r['learned16'] - r['fixed16'] for r in rows])
    d_lt = _boot([r['learned16'] - r['F_t16'] for r in rows])
    d_lp = _boot([r['learned16'] - r['pixel16'] for r in rows])
    res.update(learned_minus_fixed16=d_lf, learned_minus_Ft16=d_lt, learned_minus_pixel16=d_lp)
    for k, d in (('learned − fixed', d_lf), ('learned − F_t', d_lt), ('learned − pixel', d_lp)):
        print(f's16 {k}: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]')
    verdict = 'KILL' if d_lt[0] <= 0 else ('LEARNED>FIXED' if d_lf[1] > 0 else 'FIXED>=LEARNED')
    res['verdict'] = verdict
    print(f'G1 {verdict}', flush=True)
    with open(f'{OUT}/g1_seed{seed}.json', 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main(*map(int, sys.argv[1:]))
