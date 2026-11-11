"""Ablation (2026-11-11): the integral image's evidence ā (weighted fraction of aligned past frames gated as changed,
`a` in fusion_data.encode) fed to frozen SAM2 as its dense mask prompt, logit = s·(ā − b), with the tracker box + point.
No adapter, no training. (s, b) are selected on the seq1 hand frames (split 'select') and reported on seq2–4 ('test').

    PYTHONPATH=. python -m benchmark.nfo_vos.tools.mask_input_ablation
"""
import itertools
import numpy as np
import torch
from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor, decode, AMP
from benchmark.nfo_vos.adapter.fusion_data import encode
from benchmark.nfo_vos.adapter import fusion_nfo as FN
from benchmark.nfo_vos.eval.dump_arms import load_net, CKPT

GRID = list(itertools.product((5.0, 10.0, 20.0), (0.2, 0.35, 0.5)))


def decode_mi(sam, f, logit):
    pe = sam.sam_prompt_encoder
    n = f['X16'].shape[0]
    coords = torch.cat([f['box'].view(n, 2, 2), f['pt'][:, None]], 1)
    labels = torch.tensor([[2, 3, 1]], device=coords.device).repeat(n, 1)
    sparse, dense = pe(points=(coords, labels), boxes=None, masks=logit[:, None])
    low, _, _, _ = sam.sam_mask_decoder(image_embeddings=f['X16'], image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=sparse,
                                        dense_prompt_embeddings=dense, multimask_output=False, repeat_image=False,
                                        high_res_features=[f['X4'], f['X8']])
    return low


def run(split, pred, net):
    res = {}
    for it in FN.items(split, pred):
        f = encode(pred, [it])
        ft = dict(f, box=torch.as_tensor(it['box_trk'], device='cuda').float(), pt=torch.as_tensor(it['trk'], device='cuda').float())
        a = f['a'].float()
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP):
            o = net(f, sam=pred.model)
            outs = {'trk': decode(pred.model, ft, point=True), 'adapter': decode(pred.model, o, token=o.get('token'))}
            for s, b in GRID:
                outs[f'mi s={s:g} b={b:g}'] = decode_mi(pred.model, ft, s * (a - b))
        H, keep = torch.from_numpy(it['H']).cuda(), torch.from_numpy(it['keep']).cuda()
        for k, low in outs.items():
            pr = torch.nn.functional.interpolate(low.float(), H.shape, mode='bilinear', align_corners=False)[0, 0] > 0
            tp = (pr & H & keep).sum().item()
            res.setdefault(k, []).append((tp / max(((pr | H) & keep).sum().item(), 1), tp / max((pr & keep).sum().item(), 1),
                                          tp / max((H & keep).sum().item(), 1)))
    return {k: np.array(v) for k, v in res.items()}


def main():
    pred = image_predictor()
    net = load_net(pred, CKPT)
    sel, test = run('select', pred, net), run('test', pred, net)
    best = max((k for k in sel if k.startswith('mi')), key=lambda k: sel[k][:, 0].mean())
    print(f'selected on seq1 (n={len(sel["trk"])}): {best}  (J {sel[best][:, 0].mean():.3f}; trk {sel["trk"][:, 0].mean():.3f})')
    print(f'seq2-4 hand frames, n={len(test["trk"])}:  J / P / R')
    for k in ('trk', 'adapter', best):
        print(f'  {k:22s} ' + ' / '.join(f'{x:.3f}' for x in test[k].mean(0)))
    rng = np.random.default_rng(0)
    for k in ('adapter', best):
        d = test[k][:, 0] - test['trk'][:, 0]
        bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)]
        print(f'  {k} - trk (J): {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
    print('all grid points, seq2-4 J / P / R (for reference, not for selection):')
    for k in sorted(k for k in test if k.startswith('mi')):
        print(f'  {k:22s} ' + ' / '.join(f'{x:.3f}' for x in test[k].mean(0)))


if __name__ == '__main__':
    main()
