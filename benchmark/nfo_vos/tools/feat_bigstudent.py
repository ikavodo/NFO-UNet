"""Distil the big s16 student (feat_integral.big_net; tools/feat_capacity 'big', findings §10.x 18-19) to the pixel teacher
enc(C)16 on NFO (seq1 s4–s7, the capacity_crops cache) AND synthetic (feat_refine.crops_syn: 20 seed-11 train clips,
teacher P16), alternating batches, so its output means the same in both domains (the Fusion trains on synthetic).
3000 steps, AdamW 3e-4 cosine, seed 0. Wraps the distilled Refined aggregator (strides 8 / 4, evidence mass) as BigS16
and saves it: distil/bigs16_s4-7.pt. Logged: teacher cos on NFO train / held-out (s0–s3) and synthetic.

    PYTHONPATH=. NFO_RUN=fullpass python -m benchmark.nfo_vos.tools.feat_bigstudent
"""
import numpy as np
import torch

from benchmark.nfo_vos.adapter.feat_integral import BigS16, FeatIntegral, Refined, big_inputs, ccos
from benchmark.nfo_vos.adapter.fusion_adapter import AMP, image_predictor
from benchmark.nfo_vos.tools.feat_distil import OUT
from benchmark.nfo_vos.tools.feat_refine import crops_syn


def X(b):
    st = lambda k: torch.stack([x[k] for x in b]).cuda()
    ph = torch.stack([x['ph'] if 'ph' in x else x['dx'] / 16 - torch.round(x['dx'] / 16) for x in b]).cuda()
    return big_inputs(st('K'), st('v'), ph, st('Ft'), st('B'))


def main():
    torch.manual_seed(0)
    pred = image_predictor()
    tr, te = torch.load(f'{OUT}/capacity_crops.pt', weights_only=False)
    syn = crops_syn(pred)
    m = BigS16(Refined(FeatIntegral(pred.model.no_mem_embed).cuda())).cuda()
    m.base.load_state_dict(torch.load(f'{OUT}/refined_s4-7.pt', map_location='cuda'))
    mu, net = m.base.agg.mu16, m.net
    T = lambda b: torch.stack([x['T'] for x in b]).cuda().float()
    f = lambda b: net(X(b)).float()

    def fit(groups):
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
            return float(np.mean([ccos(f(g), T(g), mu).mean().item() for g in groups]))
    chunks = lambda d: [d[i:i + 16] for i in range(0, len(d), 16)]
    print(f'NFO train {len(tr)} / held-out {len(te)} frames; synthetic {sum(map(len, syn))} targets in {len(syn)} clips', flush=True)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    rng = np.random.default_rng(0)
    for it in range(1, 3001):
        b = [tr[i] for i in rng.choice(len(tr), 8, replace=False)] if it % 2 else syn[rng.integers(len(syn))]
        with torch.autocast('cuda', dtype=AMP):
            loss = 1 - ccos(f(b), T(b), mu).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            net.eval()
            print(f'it {it}: teacher cos NFO train {fit(chunks(tr)):.3f} held-out {fit(chunks(te)):.3f} synthetic {fit(syn):.3f}', flush=True)
            net.train()
    torch.save(m.state_dict(), f'{OUT}/bigs16_s4-7.pt')


if __name__ == '__main__':
    main()
