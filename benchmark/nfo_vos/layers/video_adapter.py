"""Layer adapter inside SAM2's own video loop (docs/layer_sam2/training/plan_memory_layers.md, option A, zero-shot;
2026-10-16). No loop is rewritten; three instance-level hooks on the frozen SAM2 video model:
  _track_step          records the current frame index
  _forward_sam_heads   (called with the MEMORY-CONDITIONED stride-16 features) adds the adapter residuals R^16/8/4 of
                       that frame, replaces an empty sparse prompt by the tracker box + point, and forces single-mask
                       output (the adapter was trained single-mask)
  sam_prompt_encoder   adds the adapter's learned correction δ to the point token of that frame
Residuals and δ depend only on raw features (F_t, F_C, F_B, a), not on memory, so they are precomputed per frame with the
image-mode path (fusion_adapter.Fusion). The predicted mask then enters SAM2's memory encoder and bank unchanged, so
improved masks propagate through memory.
Arms (run_window `arm`):  b0       plain SAM2 video, GT box + point at t0 only (the off-the-shelf reference)
                          prompts  + per-frame tracker box + point (no adapter)
                          adapter  + adapter residuals and δ (zero-shot from the per-frame checkpoint)
"""
import shutil
import tempfile

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import trials as TR, velocity as V
from benchmark.nfo_vos.factorised_gtfree import theil_at
from benchmark.nfo_vos.layers import fusion_data as FD
from benchmark.nfo_vos.layers.prompt_oracle import composite_and_mass
from benchmark.nfo_vos.layers.track_check import kernel16_chain

S1 = np.array([1024 / 800, 1024 / 600] * 2, np.float32)


@torch.no_grad()
def per_frame_inputs(pred, net, t, noC=False):
    """Tracker box / point per frame (1024 frame) and, where the 6-frame history exists, the adapter residuals and δ.
    noC: composite removed (C := empty scene, a := 0), the −composite ablation. d['it'] keeps the encoder item."""
    seq, fr = t['seq'], t['frames']
    _, _, kc, wh = kernel16_chain(t, pred, return_feats=True)
    w0, w1 = t['warmup']
    bg = np.median(np.stack([V.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    out = []
    for j in range(len(fr)):
        box = np.array([kc[j, 0] - wh[0] / 2, kc[j, 1] - wh[1] / 2, kc[j, 0] + wh[0] / 2, kc[j, 1] + wh[1] / 2], np.float32) * S1
        d = dict(box=box, pt=(kc[j] * S1[:2]).astype(np.float32), R=None, delta=None)
        if j >= 6 and net is not None:
            cs = list(range(max(0, j - 12), j + 1))
            fx = theil_at(cs, [kc[l, 0] for l in cs], [j] + list(range(j - 6, j)))
            C, a = composite_and_mass([V.native(fr[l], seq).astype(np.float32) for l in range(j - 6, j)],
                                      [np.float32([[1, 0, fx[j] - fx[l]], [0, 1, 0]]) for l in range(j - 6, j)], bg, z0='adaptive')
            if noC:
                C, a = bg, np.zeros_like(a)
            it = dict(X=V.native(fr[j], seq)[None], Ct=FD.u8(C)[None], a=FD.a256(a)[None], V=np.zeros((1, 256, 256), np.float32),
                      box=box[None], box_trk=box[None], trk=d['pt'][None], mode=np.zeros(1, int), bg=FD.u8(bg))
            f = FD.encode(pred, [it])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(f, sam=pred.model)
            d['R'] = {l: (o[f'X{l}'] - f[f'X{l}']).half() for l in ('16', '8', '4')}
            sp_plain, _ = pred.model.sam_prompt_encoder(
                points=(torch.cat([f['box_trk'].view(1, 2, 2), f['trk'][:, None]], 1),
                        torch.tensor([[2, 3, 1]], device='cuda')), boxes=None, masks=None)
            d['delta'] = (o['token'].float() - sp_plain.float())[:, 2].half()          # correction on the point token
            d['it'] = it
        out.append(d)
    return out


def install_hooks(vp, state):
    """state: dict with 'inputs' (per_frame_inputs list), 'arm', 't' (filled by the hook), optional 'labels'
    {frame: bool mask} for the memory oracle. Returns an uninstall fn. Arms: see module docstring; plus
    adapter-pre   R^16 added to the query features BEFORE memory attention (review I1), R^8/R^4 in the decoder
    oracle-mem    = prompts, but the memory encoder is fed the pseudo-GT of the frame (causal: only past frames are read)
    Prompts enter only the decoder; frames stay non-conditioning (review C4)."""
    model = vp
    orig_track, orig_heads, orig_pe = model._track_step, model._forward_sam_heads, model.sam_prompt_encoder.forward
    orig_prep, orig_enc = model._prepare_memory_conditioned_features, model._encode_new_memory

    def prep(frame_idx, is_init_cond_frame, current_vision_feats, *a, **k):
        inp = state['inputs'][frame_idx] if frame_idx < len(state['inputs']) else None
        if state['arm'] == 'adapter-pre' and inp is not None and inp['R'] is not None and not is_init_cond_frame:
            r = inp['R']['16'].to(current_vision_feats[-1].dtype)                  # [1, C, H, W] → tokens [HW, 1, C]
            current_vision_feats = list(current_vision_feats[:-1]) + [current_vision_feats[-1] + r.flatten(2).permute(2, 0, 1)]
        return orig_prep(frame_idx, is_init_cond_frame, current_vision_feats, *a, **k)

    def enc(current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
        lab = state.get('labels', {}).get(state.get('t')) if state['arm'] == 'oracle-mem' else None
        if lab is not None:
            m = torch.as_tensor(cv2.resize(lab.astype(np.float32), pred_masks_high_res.shape[-1:-3:-1]),
                                device=pred_masks_high_res.device)[None, None]
            pred_masks_high_res = (20 * m - 10).to(pred_masks_high_res.dtype)
            object_score_logits = torch.full_like(object_score_logits, 10.0 if lab.any() else -10.0)  # empty → absent
        return orig_enc(current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts)

    def track_step(frame_idx, *a, **k):
        state['t'] = frame_idx
        return orig_track(frame_idx, *a, **k)

    def heads(backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        t = state.get('t', 0)
        inp = state['inputs'][t] if t < len(state['inputs']) else None
        state['inject'] = False
        cap = state.get('capture')
        if cap is not None and t in cap['ts']:          # memory-conditioned decoder inputs, before any residual
            cap['F'][t] = (backbone_features.float().clone(), [h.float().clone() for h in high_res_features])
        if state['arm'] != 'b0' and inp is not None and point_inputs is None and mask_inputs is None:
            dev = backbone_features.device
            pts = torch.as_tensor(np.concatenate([inp['box'].reshape(2, 2), inp['pt'][None]])[None], device=dev)
            point_inputs = {'point_coords': pts, 'point_labels': torch.tensor([[2, 3, 1]], device=dev, dtype=torch.int32)}
            multimask_output = False
            if state['arm'] in ('adapter', 'adapter-pre') and inp['R'] is not None:
                if state['arm'] == 'adapter':
                    backbone_features = backbone_features + inp['R']['16'].to(backbone_features.dtype)
                high_res_features = [high_res_features[0] + inp['R']['4'].to(high_res_features[0].dtype),
                                     high_res_features[1] + inp['R']['8'].to(high_res_features[1].dtype)]
                state['inject'] = inp['delta'] is not None
        return orig_heads(backbone_features, point_inputs=point_inputs, mask_inputs=mask_inputs,
                          high_res_features=high_res_features, multimask_output=multimask_output)

    def pe(points, boxes, masks):
        sp, de = orig_pe(points, boxes, masks)
        if state.get('inject'):
            inp = state['inputs'][state['t']]
            sp = sp.clone(); sp[:, 2] = sp[:, 2] + inp['delta'].to(sp.dtype)
        return sp, de

    model._track_step, model._forward_sam_heads, model.sam_prompt_encoder.forward = track_step, heads, pe
    model._prepare_memory_conditioned_features, model._encode_new_memory = prep, enc

    def uninstall():
        model._track_step, model._forward_sam_heads, model.sam_prompt_encoder.forward = orig_track, orig_heads, orig_pe
        model._prepare_memory_conditioned_features, model._encode_new_memory = orig_prep, orig_enc
    return uninstall


def run_window(vp, t, inputs, arm, labels=None, dtype=torch.bfloat16, capture=None):
    """SAM2 video on one window with GT box + point at t0 (as run_sam2_video); returns [T, H, W] bool masks.
    arm 'plain': no hooks at all (SC1 reference). t may carry 'imgs' (synthetic clip frames) instead of an NFO 'seq'."""
    frames = t['frames']
    stage = tempfile.mkdtemp(prefix='vidadapt_')
    state = dict(inputs=inputs, arm=arm, labels=labels or {}, capture=capture)   # capture: {'ts': set, 'F': {}}
    uninstall = install_hooks(vp, state) if arm != 'plain' else (lambda: None)
    try:
        import os
        if 'imgs' in t:                                   # synthetic clip: frames given in memory (uint8, H×W)
            for i, im in enumerate(t['imgs']):
                cv2.imwrite(f'{stage}/{i}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 100])
            shape = t['imgs'][0].shape[:2]
        else:
            for i, k in enumerate(frames):
                os.symlink(os.path.abspath(f"{V.NATIVE}/{t['seq']}/{k:05d}.jpg"), f'{stage}/{i}.jpg')
            shape = (600, 800)
        masks = np.zeros((len(frames),) + shape, bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            st = vp.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            vp.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=np.array([t['point_native']], np.float32),
                                     labels=np.array([1], np.int32), box=np.array(t['box_native'], np.float32))
            for fi, _, lg in vp.propagate_in_video(st):
                masks[fi] = (lg[0, 0] > 0).cpu().numpy()
    finally:
        uninstall()
        shutil.rmtree(stage)
    return masks


def _gate_gpu(X, B):
    """prompt_oracle.composite_and_mass's gate, z0 = 'adaptive', on the GPU: robust z of |X − B| (median, 1.4826·MAD),
    threshold max(3, q90(z)). Depends only on the frame and the empty scene, so it is computed once per frame."""
    d = (X - B).abs()
    med = d.median()
    z = (d - med) / (1.4826 * (d - med).abs().median()).clamp(min=1e-6)
    return torch.sigmoid(z - torch.quantile(z.flatten(), 0.9).clamp(min=3.0))


def _shift_x(x, dx):
    """[K, H, W] shifted right by dx[k] native px: bilinear, border replicate (= cv2.warpAffine [[1, 0, dx], [0, 1, 0]])."""
    K, H, W = x.shape
    xs = torch.arange(W, device=x.device, dtype=x.dtype)[None] - dx[:, None]               # source column per output column
    gx = (2 * xs / (W - 1) - 1)[:, None, :].expand(K, H, W)
    gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)[None, :, None].expand(K, H, W)
    return torch.nn.functional.grid_sample(x[:, None], torch.stack([gx, gy], -1), mode='bilinear', padding_mode='border',
                                           align_corners=True)[:, 0]


@torch.no_grad()
def fast_masks(pred, net, t, tau=3.5, noC=False):
    """The adapter-pf masks of one window with each frame encoded ONCE (shared by tracker, adapter and decoder), the
    empty scene once per window, each frame's gate once, and the composite on the GPU. Same computation as
    per_frame_inputs + dump_arms.per_frame_masks, which re-encode frame / composite / empty scene twice per frame and
    build the composite in numpy. net None: the 'trk' arm (per-frame SAM2, tracker box + point, every frame). noC: the
    composite removed (C := empty scene, a := 0), as per_frame_inputs(noC=True). Returns ([T, H, W] bool, s / frame)."""
    import time
    from benchmark.nfo_vos.layers.fusion_adapter import decode
    from benchmark.nfo_vos.layers.track_check import gt_box, kernel_track
    seq, fr = t['seq'], t['frames']
    w0, w1 = t['warmup']
    bg = np.median(np.stack([V.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    torch.cuda.synchronize(); t0 = time.perf_counter()   # ponytail: empty scene = once-per-sequence setup, not timed per frame
    sam = pred.model

    def enc(img):
        """pred.set_image on a uint8 gray image, with SAM2's transforms (to-tensor, 1024² resize, normalise) run on the
        GPU instead of the CPU (sam2_image_predictor.py:110-127)."""
        x = torch.as_tensor(img, device='cuda').to(torch.uint8).float().div(255)[None].expand(3, -1, -1)
        _, vf, _, _ = sam._prepare_backbone_features(sam.forward_image(pred._transforms.transforms(x)[None]))
        if sam.directly_add_no_mem_embed:
            vf[-1] = vf[-1] + sam.no_mem_embed
        f4, f8, f16 = [v.permute(1, 2, 0).view(1, -1, *s_) for v, s_ in zip(vf, pred._bb_feat_sizes)]
        return f16.float(), f8.float(), f4.float()

    with torch.autocast('cuda', dtype=FD.AMP):
        Bf = enc(bg)
        Bg = torch.as_tensor(bg, device='cuda')
        Xn = [V.native(k, seq) for k in fr]
        F, G, m16 = [], [], []
        for x in Xn:                                                     # one encoder pass per frame
            F.append(enc(x))
            m = 1 - torch.nn.functional.cosine_similarity(F[-1][0][0], Bf[0][0], dim=0)
            m16.append(torch.nn.functional.interpolate(m[None, None], size=x.shape, mode='bilinear', align_corners=False)[0, 0])
            G.append(_gate_gpu(torch.as_tensor(x, device='cuda', dtype=torch.float32), Bg))
    g = gt_box(seq, fr[0])
    kc = kernel_track(torch.stack(m16).cpu().numpy(), ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), g[2] - g[0], g[3] - g[1])
    wh = (g[2] - g[0], g[3] - g[1])
    Xg = torch.as_tensor(np.stack(Xn), device='cuda', dtype=torch.float32)
    Gg = torch.stack(G)
    wk = torch.as_tensor(np.exp(-np.arange(5, -1, -1) / tau), device='cuda', dtype=torch.float32)
    wk = (wk / wk.sum())[:, None, None]
    out = np.zeros((len(fr), TR.NATIVE_H, TR.NATIVE_W), bool)
    for j in range(len(fr)):
        box = np.array([kc[j, 0] - wh[0] / 2, kc[j, 1] - wh[1] / 2, kc[j, 0] + wh[0] / 2, kc[j, 1] + wh[1] / 2], np.float32) * S1
        pt = (kc[j] * S1[:2]).astype(np.float32)
        f = {f'X{l}': F[j][i] for i, l in enumerate(('16', '8', '4'))}
        f['box'] = f['box_trk'] = torch.as_tensor(box[None], device='cuda')
        f['pt'] = f['trk'] = torch.as_tensor(pt[None], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if j < 6 or net is None:                                     # no history yet / trk arm: tracker prompts only
                low = decode(pred.model, f, point=True)
            else:
                cs = list(range(max(0, j - 12), j + 1))
                fx = theil_at(cs, [kc[l, 0] for l in cs], [j] + list(range(j - 6, j)))
                dx = torch.tensor([fx[j] - fx[l] for l in range(j - 6, j)], device='cuda', dtype=torch.float32)
                A, Gw = _shift_x(Xg[j - 6:j], dx), _shift_x(Gg[j - 6:j], dx)
                mass = (wk * Gw).sum(0)
                C = torch.where(mass > 1e-6, (wk * Gw * A).sum(0) / mass.clamp(min=1e-6), (wk * A).sum(0))
                if noC:
                    Cf, mass = Bf, torch.zeros_like(mass)
                else:
                    with torch.autocast('cuda', dtype=FD.AMP):
                        Cf = enc(C.clamp(0, 255))   # .to(uint8) in enc = FD.u8's truncation, as the benchmark path
                f.update({f'C{l}': Cf[i] for i, l in enumerate(('16', '8', '4'))})
                f.update({f'B{l}': Bf[i] for i, l in enumerate(('16', '8', '4'))})
                f['a'] = torch.nn.functional.interpolate(mass.clamp(0, 1)[None, None], size=(256, 256), mode='area')[0]
                o = net(f, sam=pred.model)
                low = decode(pred.model, o, token=o['token'])
        out[j] = cv2.resize(low[0, 0].float().cpu().numpy(), (TR.NATIVE_W, TR.NATIVE_H), interpolation=cv2.INTER_LINEAR) > 0
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / len(fr)
