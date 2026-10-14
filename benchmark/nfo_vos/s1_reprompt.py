"""S1 sanity check (2026-10-14, hand-tuned, seq1 only): causal re-prompting of SAMURAI (T1) with the V̂ mask.

Per window: box + point prompt at t0 (as T1); propagate to re-prompt frame j; V̂_j = (M_C(j) ∧ pix_bg > 6.31) ∪ mask_j,
with mask_j = THIS run's own mask at j (causal), M_C = SAM2 image predictor on the GT-free composite (blob-tracker
alignment, blob-tracker point prompt: the chosen GT-free arm); add_new_mask(j, V̂_j) (SAM2 config
use_mask_input_as_output_without_sam: the mask becomes frame j's output and a non-conditioning memory entry with
IoU 1, admitted by SAMURAI's memory selection; SAMURAI cannot take mid-video conditioning frames);
propagate from j. Frames before j are never recomputed. Arms: re-prompt every 20 frames (j = 20) and every 10
(j = 10, 20, 30).
Scored: seq1 hand frames (polygon, brush) with local index > first re-prompt and not themselves re-prompt frames,
i.e. frames where only SAM2's propagation of the injected mask can act. References: T1 (stored run) and post-hoc
V̂ (masks/t1-vhat, same rule applied per frame without feeding back into SAM2).
Questions: does the injected extent propagate (S1 > T1 on non-prompted frames)? Does it beat post-hoc V̂ there?
"""
import json
import os
import shutil
import tempfile

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.composite_tests import GT, stats
from benchmark.nfo_vos.factorised_gtfree import composite, theil_at
from benchmark.nfo_vos.run_sam2_video import NATIVE_DIR, build_predictor
from tracking.core.blob_tracker import merged_center

TAU = 6.31
OUT = f'{TR.RES}/masks'


def vhat_add(t, j, Mj, Tk, bg, pred):
    seq, k = t['seq'], t['frames'][j]
    cs = [l for l in range(max(0, j - 12), j + 1) if l in Tk['chain']]
    if len(cs) < 2:
        return None
    fx = theil_at(cs, [TR.to_native(*Tk['chain'][l])[0] for l in cs], [j] + list(range(j - 6, j)))
    C = composite(seq, k, {t['frames'][l]: (fx[j] - fx[l], 0.0) for l in range(j - 6, j)}, bg)
    anchor = Tk['chain'].get(j) or tuple(R.readout_line(Tk['chain'], j)[:2])
    pt = TR.to_native(*merged_center(Tk['dets'][j], *anchor, Tk['kw']['merge_radius']))
    pred.set_image(np.stack([np.clip(C, 0, 255).astype(np.uint8)] * 3, -1))
    m, _, _ = pred.predict(point_coords=np.array([pt]), point_labels=np.array([1]), multimask_output=False)
    pb = np.log(cv2.boxFilter((V.native(k, seq).astype(np.float32) - bg) ** 2, -1, (5, 5)) + 1)
    return (m[0] > 0) & (pb > TAU) & ~Mj


def run(vp, ip, t, js):
    w0, w1 = t['warmup']
    bg = np.median(np.stack([V.native(i, t['seq']) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    Tk = R.track(t)
    n = len(t['frames'])
    stage = tempfile.mkdtemp(prefix='nfo_s1_')
    try:
        for i, f in enumerate(t['frames']):
            os.symlink(os.path.abspath(f"{NATIVE_DIR}/{t['seq']}/{f:05d}.jpg"), os.path.join(stage, f'{i}.jpg'))
        masks = np.zeros((n, 600, 800), bool)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            st = vp.init_state(stage, offload_video_to_cpu=True, offload_state_to_cpu=True)
            vp.add_new_points_or_box(st, frame_idx=0, obj_id=0, points=np.array([t['point_native']], np.float32),
                                     labels=np.array([1], np.int32), box=np.array(t['box_native'], np.float32))
            cur, used = 0, []
            for j in list(js) + [None]:
                todo = (j if j is not None else n - 1) - cur
                for fi, _, lg in vp.propagate_in_video(st, start_frame_idx=cur, max_frame_num_to_track=todo):
                    masks[fi] = (lg[0, 0] > 0).cpu().numpy()
                if j is None:
                    break
                with torch.autocast('cuda', enabled=False):
                    add = vhat_add(t, j, masks[j], Tk, bg, ip)
                if add is not None and add.any():
                    vp.add_new_mask(st, frame_idx=j, obj_id=0, mask=masks[j] | add)
                    masks[j] |= add
                    used.append(j)
                cur = j
        return masks, used
    finally:
        shutil.rmtree(stage)


def main():
    T = [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == 'seq1']
    hand = {}
    for suf in ('handpoly', 'handmask_raw'):
        for f in os.listdir(GT):
            if f.endswith(f'_{suf}.png'):
                hand.setdefault(int(f[:5]), []).append(suf)
    T = [t for t in T if any(k in t['frames'] for k in hand)]
    vp, ip = build_predictor('t1'), R.image_predictor()
    # SAMURAI's memory selection (sam2_base.py:676-678) assumes every past frame is non-conditioning and reads its
    # best_iou_score; a mid-video mask correction is stored non-conditioning (SAM2 default) but its consolidated output
    # lacks SAMURAI's fields. Give the injected mask full trust (IoU 1) so SAMURAI admits it to the memory bank.
    pre = vp.propagate_in_video_preflight
    def preflight(st):
        pre(st)
        for out in st['output_dict']['non_cond_frame_outputs'].values():
            if 'best_iou_score' not in out:
                out['best_iou_score'] = torch.ones(1, 1, device=out['object_score_logits'].device)
    vp.propagate_in_video_preflight = preflight
    arms = {'s1_every20': (20,), 's1_every10': (10, 20, 30)}
    rows = []
    for a, js in arms.items():
        os.makedirs(f'{OUT}/t1-{a}', exist_ok=True)
        for t in T:
            p = f"{OUT}/t1-{a}/{t['id']}.npz"
            if os.path.exists(p):
                z = np.load(p); masks, used = z['masks'], list(z['used'])
            else:
                masks, used = run(vp, ip, t, js)
                np.savez_compressed(p, masks=masks, used=np.array(used))
            print(f"{a} {t['id']}: re-prompted at {used}", flush=True)
            T1 = np.load(f"{OUT}/t1/{t['id']}.npz")['masks'].astype(bool)
            VH = np.load(f"{OUT}/t1-vhat/{t['id']}.npz")['masks'].astype(bool)
            for k in hand:
                if k not in t['frames']:
                    continue
                j = t['frames'].index(k)
                if j <= js[0] or j in js:
                    continue
                for suf in hand[k]:
                    h = cv2.imread(f'{GT}/{k:05d}_{suf}.png', 0) == 1
                    rows.append(dict(arm=a, ref=suf, k=k, T1=stats(T1[j], h), VH=stats(VH[j], h), S1=stats(masks[j], h)))
    rng = np.random.default_rng(0)
    for a in arms:
        for suf in ('handpoly', 'handmask_raw'):
            rr = [r for r in rows if r['arm'] == a and r['ref'] == suf]
            if len(rr) < 2:
                continue
            print(f'\n{a} vs {suf} (n={len(rr)} non-prompted frames after the first re-prompt): J / P / R')
            for n_ in ('T1', 'VH', 'S1'):
                print(f'  {n_:3s} ' + ' / '.join(f'{v:.3f}' for v in np.mean([r[n_] for r in rr], 0)))
            for x, y in (('S1', 'T1'), ('S1', 'VH')):
                d = np.array([r[x][0] - r[y][0] for r in rr])
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
                print(f'  ΔJ {x}−{y} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}], '
                      f'better on {(d > 0).sum()}/{len(d)}')


if __name__ == '__main__':
    main()
