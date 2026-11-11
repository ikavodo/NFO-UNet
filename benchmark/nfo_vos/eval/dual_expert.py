"""Dual-expert harness for the adapter gate (2026-11-11; plan in docs/panaf/2026-11-11-msc-project-direction.md, WP3).

Per frame, two experts on ONE image-encoder pass of that frame:
  E0  SAM2.1 video predictor with memory attention (B0; 't1' = SAMURAI's Kalman mode), prompted at t0 with the GT box +
      point exactly as baselines/run_sam2_video.py (fp16 autocast, forward propagation).
  E1  Layer-SAM (adapter, seed 0 checkpoint): inference.masks_from_arrays, fed the frame features E0's encoder pass
      produced (hook on SAM2VideoPredictor._get_image_feature). E1 still encodes the empty scene once and the integral
      image once per frame: that pass is intrinsic to E1.
E0 writes its own masks into memory, so E0 is plain B0 / T1 and E1 is a side computation: the experts stay independent and
the per-frame comparison is clean. Logged per frame: E0's object-score logit and IoU-head prediction, E1's same heads,
change-gate / evidence statistics inside E1's tracker box (inference.box_signals), mask agreement IoU(E0, E1), and the
outcome of each expert (J vs the exact visible mask on synthetic clips, box IoU vs the GT box everywhere, J on NFO hand
frames).

Deviations from the stand-alone arms: frames enter E0 through SAM2Transforms on the gray frame (the same tensor E1's
encoder sees) instead of SAM2's JPEG loader, and E1 sees fp16 frame features (stand-alone: bf16). Check against the
stand-alone arms with `check`.

    PYTHONPATH=. python -m benchmark.nfo_vos.eval.dual_expert synth [n_clips]   # GPJATK val clips × ρ sweep
    PYTHONPATH=. python -m benchmark.nfo_vos.eval.dual_expert nfo               # the 61 NFO benchmark windows
    PYTHONPATH=. python -m benchmark.nfo_vos.eval.dual_expert check             # E0 / E1 vs the stand-alone b0 / adapter-pf masks
    PYTHONPATH=. python -m benchmark.nfo_vos.eval.dual_expert analyse           # oracle bound, signal AUROCs, learned gate
"""
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

OUT = 'results/benchmark/gate'
RHOS = (0.0, 0.15, 0.3, 0.45, 0.6)
CFG = {'b0': 'configs/sam2.1/sam2.1_hiera_b+.yaml', 't1': 'configs/samurai/sam2.1_hiera_b+.yaml'}
SIGNALS = ('obj0', 'iou0', 'fill0', 'obj1', 'iou1', 'gate_box', 'n_comp', 'largest', 'mass_box', 'agree')


def build(method='b0'):
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from benchmark.nfo_vos.adapter.fusion_adapter import SAM2_CKPT
    from benchmark.nfo_vos.eval.dump_arms import load_net, CKPT
    vp = build_sam2_video_predictor(CFG[method], SAM2_CKPT, device='cuda')
    pred = SAM2ImagePredictor(vp)                       # same weights: E1's prompt encoder / decoder are vp's
    return vp, pred, load_net(pred, CKPT)


def run_clip(vp, pred, net, X, bg, box, pt):
    """X: list of [H, W] uint8 gray frames, bg: empty scene, box: t0 GT box xyxy, pt: t0 point (x, y). -> m0, m1, rows"""
    import sam2.sam2_video_predictor as V
    from benchmark.nfo_vos.adapter.inference import masks_from_arrays
    H, W = X[0].shape
    imgs = torch.stack([pred._transforms.transforms(torch.as_tensor(x).float().div(255)[None].expand(3, -1, -1)) for x in X])
    feats, orig_load, orig_gif = {}, V.load_video_frames, vp._get_image_feature

    def gif(state, frame_idx, batch_size):              # capture the frame features E0 computes (one encoder pass)
        out = orig_gif(state, frame_idx, batch_size)
        if frame_idx not in feats:
            _, bb = state['cached_features'][frame_idx]
            _, vf, _, _ = vp._prepare_backbone_features({k: list(bb[k]) for k in ('backbone_fpn', 'vision_pos_enc')})
            vf = list(vf)
            if vp.directly_add_no_mem_embed:
                vf[-1] = vf[-1] + vp.no_mem_embed
            f4, f8, f16 = [v.permute(1, 2, 0).reshape(1, -1, *s).float() for v, s in zip(vf, pred._bb_feat_sizes)]
            feats[frame_idx] = (f16, f8, f4)
        return out

    V.load_video_frames = lambda **k: (imgs, H, W)
    vp._get_image_feature = gif
    m0, sig0 = np.zeros((len(X), H, W), bool), {}
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            st = vp.init_state(None, offload_video_to_cpu=True, offload_state_to_cpu=True)
            vp.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=np.array([pt], np.float32),
                                     labels=np.array([1], np.int32), box=np.array(box, np.float32))
            for fi, _, logits in vp.propagate_in_video(st):
                m0[fi] = (logits[0, 0] > 0).cpu().numpy()
                od = st['output_dict']
                o = od['cond_frame_outputs'].get(fi) or od['non_cond_frame_outputs'][fi]
                sig0[fi] = (float(o['object_score_logits'].float().flatten()[0]), float(torch.as_tensor(o['best_iou_score']).float().flatten()[0]) if o.get('best_iou_score') is not None else np.nan)   # t0 (consolidated) has none
    finally:
        V.load_video_frames, vp._get_image_feature = orig_load, orig_gif
    aux = []
    with torch.inference_mode():
        m1, _ = masks_from_arrays(pred, net, X, bg.astype(np.float32), np.asarray(box, float), feats=[feats[i] for i in range(len(X))], aux=aux)
    rows = []
    for j, a in enumerate(aux):
        a.pop('j'); bx = a.pop('box'); bw, bh = max(bx[2] - bx[0], 1), max(bx[3] - bx[1], 1)
        rows.append(dict(j=j, obj0=sig0[j][0], iou0=sig0[j][1], fill0=float(m0[j].sum() / (bw * bh)),
                         obj1=a.pop('obj'), iou1=a.pop('pred_iou'), agree=iou(m0[j], m1[j]), **a))
    return m0, m1, rows


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def mask_box(m):
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def box_iou(a, b):
    if a is None:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return float(ix * iy / u) if u > 0 else 0.0


def write(rows, name):
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(f'{OUT}/{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('wrote', f'{OUT}/{name}.csv', len(rows), 'frames')


# ---------------------------------------------------------------- sources
def main_synth(n=24):
    from benchmark.nfo_vos.adapter import dataset as D, fusion_data as FD
    from benchmark.nfo_vos.synth import noise
    vp, pred, net = build()
    specs = D.clip_specs('val')[::max(1, len(D.clip_specs('val')) // n)][:n]     # held-out GPJATK subjects
    rows = []
    for ci, (seq, cam, _, seed) in enumerate(specs):
        for rho in RHOS:
            rng = np.random.default_rng(seed)
            d = FD.rescale_clip(D.make_clip(seq, cam, rho, seed, flutter=noise.FLUTTER_AMP, clump=FD.CLUMPS[rng.integers(3)]), 0.75)
            X, bg, Vm, S = list(FD.u8(d['X'])), FD.u8(d['bg']), np.asarray(d['V']) > 0, np.asarray(d['S']) > 0
            gb = mask_box(S[0])
            if gb is None or not Vm[0].any():
                continue
            dt = cv2.distanceTransform(Vm[0].astype(np.uint8), cv2.DIST_L2, 5)
            py, px = np.unravel_index(dt.argmax(), dt.shape)
            m0, m1, rr = run_clip(vp, pred, net, X, bg, gb, (px, py))
            for r in rr:
                j, gt = r['j'], mask_box(S[r['j']])
                r.update(src='synth', clip=f'{seq}_{cam}', rho=rho, J0=iou(m0[j], Vm[j]), J1=iou(m1[j], Vm[j]),
                         b0=box_iou(mask_box(m0[j]), gt) if gt else np.nan, b1=box_iou(mask_box(m1[j]), gt) if gt else np.nan)
            rows += rr
            print(f'{ci} {seq}_{cam} rho={rho}: J0 {np.mean([r["J0"] for r in rr]):.3f} J1 {np.mean([r["J1"] for r in rr]):.3f}', flush=True)
    write(rows, 'synth')


def main_nfo():
    from benchmark.nfo_vos import trials as TR
    from benchmark.nfo_vos.adapter.tracker import gt_box
    vp, pred, net = build()
    rows, masks = [], {}
    for t in [t for t in json.load(open('results/benchmark/rr/trials.json')) if t['admissible']]:
        seq, fr = t['seq'], t['frames']
        bg = np.median(np.stack([TR.native(i, seq) for i in range(t['warmup'][0], t['warmup'][1] + 1)]), 0)
        X = [TR.native(k, seq) for k in fr]
        m0, m1, rr = run_clip(vp, pred, net, X, bg, gt_box(seq, fr[0]), t['point_native'])
        hand = set(TR.hand_frames(seq))
        for r in rr:
            k = fr[r['j']]
            gt = TR.gt_to_native(TR.bbs(seq)[k][0], TR.NATIVE_W, TR.NATIVE_H)
            r.update(src='nfo', clip=t['id'], seq=seq, frame=k, b0=box_iou(mask_box(m0[r['j']]), gt), b1=box_iou(mask_box(m1[r['j']]), gt))
            if k in hand:
                Hm, keep = TR.hand_label(seq, k)
                keep = np.ones_like(Hm, bool) if keep is None else keep
                r.update(J0=iou(m0[r['j']] & keep, Hm & keep), J1=iou(m1[r['j']] & keep, Hm & keep))
        rows += rr
        masks[t['id']] = (m0, m1)
        print(t['id'], f"b0 {np.mean([r['b0'] for r in rr]):.3f} b1 {np.mean([r['b1'] for r in rr]):.3f}", flush=True)
    write(rows, 'nfo')
    np.savez_compressed(f'{OUT}/nfo_masks_first3.npz', **{f'{k}_{e}': masks[k][e] for k in list(masks)[:3] for e in (0, 1)})


def main_kth(n=60):
    """No-harm check on real unoccluded video: KTH walking/jogging crossings (d1/d3/d4, static camera), first 40 frames
    of each crossing, empty scene = per-pixel median of the whole video. Outcome: box IoU of each expert's mask box vs
    the KTH GT box (the same metric as on NFO; KTH has no GT masks)."""
    from benchmark.nfo_vos.eval import gait_masks as GM
    vp, pred, net = build()
    cr = sorted(GM.kth_crossings().items())
    pool = [(name, v, r) for name, v in cr for r in v['runs'] if r[1] - r[0] + 1 >= 30]
    pool = [pool[i] for i in np.random.default_rng(0).permutation(len(pool))[:n]]
    rows, cache = [], {}
    for name, v, (s0, e0) in pool:
        if name not in cache:
            fr = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in GM.read_avi(name)]
            cache = {name: (fr, np.median(np.stack(fr), 0))}
        fr, bg = cache[name]
        e0 = min(e0, s0 + 39, len(fr) - 1)
        g = v['gt'][s0:e0 + 1]
        gb = np.stack([g[:, 0] * 160, g[:, 1] * 120, (g[:, 0] + g[:, 2]) * 160, (g[:, 1] + g[:, 3]) * 120], 1)
        m0, m1, rr = run_clip(vp, pred, net, fr[s0:e0 + 1], bg, gb[0], ((gb[0, 0] + gb[0, 2]) / 2, (gb[0, 1] + gb[0, 3]) / 2))
        for r in rr:
            j = r['j']
            r.update(src='kth', clip=f'{name}_{s0}', label=v['label'], b0=box_iou(mask_box(m0[j]), gb[j]), b1=box_iou(mask_box(m1[j]), gb[j]))
        rows += rr
        print(name, s0, f"b0 {np.mean([r['b0'] for r in rr]):.3f} b1 {np.mean([r['b1'] for r in rr]):.3f}", flush=True)
    write(rows, 'kth')


def clip_boot(d, grp, n=2000, seed=0):
    """Mean of d and a 95% CI from resampling clips (frames within a clip are not independent)."""
    cl = sorted(set(grp)); per = np.array([d[grp == c].mean() for c in cl])
    rng = np.random.default_rng(seed)
    bs = [per[rng.integers(len(per), size=len(per))].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def analyse_real():
    """Gate fitted on NFO seq1 frames only, applied to seq2-4 (the held-out rule): does a real-data gate generalise
    across scenes? CIs: clip (window) bootstrap of the per-frame differences."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    R = list(csv.DictReader(open(f'{OUT}/nfo.csv')))
    f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, '', 'nan') else np.nan for r in R])
    a0, a1, grp = f('b0'), f('b1'), np.array([r['clip'] for r in R])
    s1 = np.array([r['seq'] == 'seq1' for r in R])
    X = np.nan_to_num(np.stack([f(s) for s in SIGNALS], 1))
    m = s1 & (np.abs(a1 - a0) > 0.01)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[m], (a1 > a0)[m].astype(int))
    te = ~s1
    gated = np.where(clf.predict_proba(X)[:, 1] > 0.5, a1, a0)
    print(f'NFO seq2-4 ({te.sum()} frames, {len(set(grp[te]))} windows), gate fitted on seq1 ({m.sum()} frames):')
    print(f'  E0 {a0[te].mean():.3f}  E1 {a1[te].mean():.3f}  gated {gated[te].mean():.3f}  oracle {np.maximum(a0, a1)[te].mean():.3f}'
          f'  (E1 chosen on {np.mean(gated[te] == a1[te]):.0%} of frames)')
    for nm, d in (('E1 - E0', a1 - a0), ('gated - E1', gated - a1), ('oracle - E1', np.maximum(a0, a1) - a1)):
        print('  %-12s %+.3f [%+.3f, %+.3f]' % ((nm,) + clip_boot(d[te], grp[te])))
    w = clf[-1].coef_[0]
    print('  LR weights (standardised):', ', '.join(f'{s} {c:+.2f}' for s, c in sorted(zip(SIGNALS, w), key=lambda x: -abs(x[1]))))
    if os.path.exists(f'{OUT}/kth.csv'):
        K = list(csv.DictReader(open(f'{OUT}/kth.csv')))
        k0, k1, kg = (np.array([float(r[c]) for r in K]) for c in ('b0', 'b1')), None, np.array([r['clip'] for r in K])
        k0, k1 = list(k0)
        print(f'KTH (no occlusion, {len(K)} frames, {len(set(kg))} crossings): E0 {k0.mean():.3f}  E1 {k1.mean():.3f}'
              f'  E1 worse by >0.05 on {np.mean(k1 < k0 - 0.05):.0%} of frames')
        print('  E1 - E0 %+.3f [%+.3f, %+.3f]' % clip_boot(k1 - k0, kg))
        for lab, nm in ((0, 'walking'), (1, 'jogging')):
            mm = np.array([int(r['label']) == lab for r in K])
            print(f'    {nm}: E0 {k0[mm].mean():.3f} E1 {k1[mm].mean():.3f}')


CHEAP = ('gate_box', 'n_comp', 'largest', 'mass_box')    # change-map statistics in the tracker box: pixel ops, either mode
F_SAM2, F_LAYER = ('obj0', 'iou0', 'fill0') + CHEAP, ('iou1', 'obj1') + CHEAP


def fit_switch():
    """Two causal switches, one per mode (the heads of the two modes live on different scales). Each predicts, from the
    ACTIVE model's signals at frame t, whether Layer-SAM beats SAM2 at t+1. Trained on NFO seq1 + odd KTH subjects from the
    logged dual runs (nfo.csv, kth.csv). -> results/benchmark/gate/switch.pkl"""
    import pickle
    from collections import defaultdict
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    v = lambda r, k: float(r[k]) if r.get(k) not in (None, '', 'nan') else 0.0
    data = {F_SAM2: ([], []), F_LAYER: ([], [])}
    for p, train in (('nfo', lambda c: c.startswith('seq1')), ('kth', lambda c: int(c[6:8]) % 2 == 1)):
        cl = defaultdict(list)
        for r in csv.DictReader(open(f'{OUT}/{p}.csv')):
            cl[r['clip']].append(r)
        for c, rs in cl.items():
            if not train(c):
                continue
            rs.sort(key=lambda r: int(r['j']))
            for r, nx in zip(rs[:-1], rs[1:]):
                if abs(v(nx, 'b1') - v(nx, 'b0')) > 0.01:
                    for F, (X, y) in data.items():
                        X.append([v(r, k) for k in F]); y.append(int(v(nx, 'b1') > v(nx, 'b0')))
    sw = {F: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')).fit(np.array(X), np.array(y))
          for F, (X, y) in data.items()}
    pickle.dump(sw, open(f'{OUT}/switch.pkl', 'wb'))
    return sw


def policy_clip(vp, pred, net, sw, X, bg, box, pt):
    """The deployable policy on one clip: ONE model decides each frame's mask. Start in SAM2 video mode; after frame t the
    active mode's switch picks the mode of t+1. Switch back to SAM2 = option (c): the tracker box + centre at that frame
    become a new conditioning prompt (one memory-encoder call). Layer-SAM's per-frame output depends only on frames, empty
    scene and tracker, never on the mode history, so it is computed once for the clip and read on the frames it is active
    (identical masks; the harness's extra compute is evaluation-only). -> masks [T, H, W], modes [T], n_switches"""
    import sam2.sam2_video_predictor as V
    from benchmark.nfo_vos.adapter.inference import masks_from_arrays
    H, W = X[0].shape
    aux = []
    with torch.inference_mode():
        m1, _ = masks_from_arrays(pred, net, X, bg.astype(np.float32), np.asarray(box, float), aux=aux)
    imgs = torch.stack([pred._transforms.transforms(torch.as_tensor(x).float().div(255)[None].expand(3, -1, -1)) for x in X])
    orig = V.load_video_frames
    V.load_video_frames = lambda **k: (imgs, H, W)
    out, modes, mode, nsw = np.zeros((len(X), H, W), bool), [], 0, 0
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            st = vp.init_state(None, offload_video_to_cpu=True, offload_state_to_cpu=True)
            vp.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=np.array([pt], np.float32),
                                     labels=np.array([1], np.int32), box=np.array(box, np.float32))
            prev = 0
            for t in range(len(X)):
                a = aux[t]
                if mode == 0:
                    if prev == 1 and t > 0:            # (c): re-prompt SAM2 from the tracker at the switch-back frame
                        b = np.array(a['box'], np.float32)
                        vp.add_new_points_or_box(st, frame_idx=t, obj_id=0, points=np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]], np.float32),
                                                 labels=np.array([1], np.int32), box=b)
                    for fi, _, lg in vp.propagate_in_video(st, start_frame_idx=t, max_frame_num_to_track=0):
                        out[t] = (lg[0, 0] > 0).cpu().numpy()
                    od = st['output_dict']; o = od['cond_frame_outputs'].get(t) or od['non_cond_frame_outputs'][t]
                    bx = a['box']; bw, bh = max(bx[2] - bx[0], 1), max(bx[3] - bx[1], 1)
                    sig = dict(obj0=float(o['object_score_logits'].float().flatten()[0]),
                               iou0=float(torch.as_tensor(o['best_iou_score']).float().flatten()[0]) if o.get('best_iou_score') is not None else 0.0,
                               fill0=float(out[t].sum() / (bw * bh)))
                    F = F_SAM2
                else:
                    out[t] = m1[t]; sig = dict(iou1=a['pred_iou'], obj1=a['obj']); F = F_LAYER
                sig.update({k: a[k] for k in CHEAP})
                modes.append(mode); prev = mode
                new = int(sw[F].predict_proba([[sig[k] for k in F]])[0, 1] > 0.5)
                nsw += new != mode; mode = new
    finally:
        V.load_video_frames = orig
    return out, np.array(modes), nsw


def main_policy():
    """Run the deployable switch on the held-out clips (NFO seq2-4 windows, KTH even subjects of the 60-crossing pool) and
    compare with the logged always-SAM2 / always-Layer-SAM box IoU on the same frames."""
    from benchmark.nfo_vos import trials as TR
    from benchmark.nfo_vos.adapter.tracker import gt_box
    from benchmark.nfo_vos.eval import gait_masks as GM
    sw = fit_switch()
    vp, pred, net = build()
    rows = []
    for t in [t for t in json.load(open('results/benchmark/rr/trials.json')) if t['admissible'] and t['seq'] != 'seq1']:
        seq, fr = t['seq'], t['frames']
        bg = np.median(np.stack([TR.native(i, seq) for i in range(t['warmup'][0], t['warmup'][1] + 1)]), 0)
        m, md, nsw = policy_clip(vp, pred, net, sw, [TR.native(k, seq) for k in fr], bg, gt_box(seq, fr[0]), t['point_native'])
        for j, k in enumerate(fr):
            rows.append(dict(src='nfo', clip=t['id'], j=j, mode=md[j], nsw=nsw,
                             bs=box_iou(mask_box(m[j]), TR.gt_to_native(TR.bbs(seq)[k][0], TR.NATIVE_W, TR.NATIVE_H))))
        print(t['id'], f'on {md.mean():.0%} switches {nsw}', flush=True)
    cr = sorted(GM.kth_crossings().items())
    pool = [(name, v, r) for name, v in cr for r in v['runs'] if r[1] - r[0] + 1 >= 30]
    pool = [pool[i] for i in np.random.default_rng(0).permutation(len(pool))[:60]]
    for name, v, (s0, e0) in pool:
        if v['subject'] % 2:
            continue
        fr = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in GM.read_avi(name)]
        e0 = min(e0, s0 + 39, len(fr) - 1)
        g = v['gt'][s0:e0 + 1]
        gb = np.stack([g[:, 0] * 160, g[:, 1] * 120, (g[:, 0] + g[:, 2]) * 160, (g[:, 1] + g[:, 3]) * 120], 1)
        m, md, nsw = policy_clip(vp, pred, net, sw, fr[s0:e0 + 1], np.median(np.stack(fr), 0), gb[0],
                                 ((gb[0, 0] + gb[0, 2]) / 2, (gb[0, 1] + gb[0, 3]) / 2))
        for j in range(len(m)):
            rows.append(dict(src='kth', clip=f'{name}_{s0}', j=j, mode=md[j], nsw=nsw, bs=box_iou(mask_box(m[j]), gb[j])))
        print(name, s0, f'on {md.mean():.0%} switches {nsw}', flush=True)
    write(rows, 'policy')
    report_policy()


def report_policy():
    P = list(csv.DictReader(open(f'{OUT}/policy.csv')))
    for src in ('nfo', 'kth'):
        L = {(r['clip'], int(r['j'])): r for r in csv.DictReader(open(f'{OUT}/{src}.csv'))}
        R = [r for r in P if r['src'] == src]
        bs = np.array([float(r['bs']) for r in R]); grp = np.array([r['clip'] for r in R])
        b0 = np.array([float(L[(r['clip'], int(r['j']))]['b0']) for r in R]); b1 = np.array([float(L[(r['clip'], int(r['j']))]['b1']) for r in R])
        best = b1 if b1.mean() > b0.mean() else b0
        nsw = {r['clip']: int(r['nsw']) for r in R}
        print(f'{src} held-out ({len(set(grp))} clips): always-SAM2 {b0.mean():.3f}  always-Layer-SAM {b1.mean():.3f}  policy {bs.mean():.3f}'
              f'  (Layer-SAM on {np.mean([int(r["mode"]) for r in R]):.0%}, switches/clip {np.mean(list(nsw.values())):.1f})')
        print('  policy - better fixed %+.3f [%+.3f, %+.3f]' % clip_boot(bs - best, grp))


def check():
    """E0 / E1 of the harness vs the stand-alone arms on 3 windows (mean IoU per frame); expect ~1 up to the stated deviations."""
    z = np.load(f'{OUT}/nfo_masks_first3.npz')
    for k in sorted({n.rsplit('_', 1)[0] for n in z.files}):
        for e, arm in ((0, 'b0'), (1, 'adapter-pf')):
            ref = np.load(f'results/benchmark/rr/masks/{arm}/{k}.npz')['masks']
            print(k, f'E{e} vs {arm}: mean IoU {np.mean([iou(a, b) for a, b in zip(z[f"{k}_{e}"], ref)]):.3f}')


# ---------------------------------------------------------------- analysis
def auroc(x, y):
    from sklearn.metrics import roc_auc_score
    k = np.isfinite(x)                                  # t0 has no E0 IoU-head value
    return roc_auc_score(y[k], x[k]) if 0 < y[k].mean() < 1 else np.nan


def analyse():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    for name, o0, o1 in (('synth', 'J0', 'J1'), ('nfo', 'b0', 'b1')):
        p = f'{OUT}/{name}.csv'
        if not os.path.exists(p):
            continue
        R = list(csv.DictReader(open(p)))
        f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, '') else np.nan for r in R])
        a0, a1, grp = f(o0), f(o1), np.array([r['clip'] for r in R])
        ok = ~(np.isnan(a0) | np.isnan(a1))
        print(f'\n===== {name}: outcome {o0}/{o1}, {ok.sum()} frames, {len(set(grp))} clips')
        bins = [('all', ok)] + ([(f'rho={rh:g}', ok & (f('rho') == rh)) for rh in RHOS] if name == 'synth' else [])
        print('  bin         E0     E1  oracle  E1-E0  P(E1 worse by >0.05)')
        for b, m in bins:
            print(f'  {b:9s} {a0[m].mean():.3f}  {a1[m].mean():.3f}  {np.maximum(a0, a1)[m].mean():.3f}  {(a1 - a0)[m].mean():+.3f}  {np.mean(a1[m] < a0[m] - 0.05):.2f}')
        m = ok & (np.abs(a1 - a0) > 0.01)                  # frames where the choice matters
        y = (a1 > a0)[m].astype(int)
        print(f'  signal AUROC for "E1 better" on {m.sum()} frames where |E1-E0| > 0.01 (0.5 = useless; <0.5 = inverse):')
        X = np.stack([f(s) for s in SIGNALS], 1)
        for i, s in enumerate(SIGNALS):
            print(f'    {s:9s} {auroc(X[m, i], y):.3f}')
        # learned gate: logistic regression, clip-grouped CV, choose E1 iff p > 0.5
        Xm, g = np.nan_to_num(X[m]), grp[m]
        pick = np.zeros(len(y), bool)
        for tr, te in GroupKFold(5).split(Xm, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xm[tr], y[tr])
            pick[te] = clf.predict_proba(Xm[te])[:, 1] > 0.5
        gated = a0.copy(); idx = np.where(m)[0]; gated[idx[pick]] = a1[idx[pick]]
        sel = np.where(m)[0]
        print(f'  gate (LR, grouped 5-fold): mean {gated[ok].mean():.3f}  vs E0 {a0[ok].mean():.3f} / E1 {a1[ok].mean():.3f} / oracle {np.maximum(a0, a1)[ok].mean():.3f}')
        for b, mm in bins[1:]:
            print(f'    {b:9s} gated {gated[mm].mean():.3f}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'synth': lambda: main_synth(int(sys.argv[2]) if len(sys.argv) > 2 else 24), 'nfo': main_nfo, 'check': check,
     'analyse': analyse, 'kth': main_kth, 'real': analyse_real, 'policy': main_policy}[cmd]()
