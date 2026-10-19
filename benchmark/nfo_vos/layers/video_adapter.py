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
            object_score_logits = torch.full_like(object_score_logits, 10.0)
        return orig_enc(current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts)

    def track_step(frame_idx, *a, **k):
        state['t'] = frame_idx
        return orig_track(frame_idx, *a, **k)

    def heads(backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
        t = state.get('t', 0)
        inp = state['inputs'][t] if t < len(state['inputs']) else None
        state['inject'] = False
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


def run_window(vp, t, inputs, arm, labels=None, dtype=torch.bfloat16):
    """SAM2 video on one window with GT box + point at t0 (as run_sam2_video); returns [T, H, W] bool masks.
    arm 'plain': no hooks at all (SC1 reference)."""
    frames = t['frames']
    stage = tempfile.mkdtemp(prefix='vidadapt_')
    state = dict(inputs=inputs, arm=arm, labels=labels or {})
    uninstall = install_hooks(vp, state) if arm != 'plain' else (lambda: None)
    try:
        import os
        for i, k in enumerate(frames):
            os.symlink(os.path.abspath(f"{V.NATIVE}/{t['seq']}/{k:05d}.jpg"), f'{stage}/{i}.jpg')
        masks = np.zeros((len(frames), 600, 800), bool)
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
