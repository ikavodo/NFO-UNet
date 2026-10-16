"""V̂'s composite as SAM2/SAMURAI memory init ("cp-V̂"), scored against the frozen silhouette label (2026-10-16).

Pre-registered before any number was seen:
  prepend frame at t0: C = V̂'s GT-free weighted-temporal composite (factorised_gtfree.composite) of t0-6..t0-1 onto t0,
      aligned by a Theil–Sen fit (x only, dy = 0) of the blob tracker's chain over global [t0-12, t0]; the tracker is
      run over those past frames in reverse order (t0 first), started from the t0 box exactly as run_t4.track does.
  M_C: SAM2 image predictor (sam2.1 b+) on C, prompt = the tracker's merged centre at t0 (point only, as V̂).
  planted frame = bg·(1 − M_C) + C·M_C, bg = warm-up median.  It is staged before raw t0 and prompted with the same
      box + p* the plain runs use at t0 (raw t0 itself unprompted), so it is the conditioning memory.
  gate: prepend only if the chain covers ≥ 3 of the 6 history frames; otherwise the arm IS the plain run (no
      history before a segment start).
  score: the vhat_vs_silhouette.py frames and label (local 6..W−2, non-empty SAM2-ceiling), window-level bootstrap,
      Δ(cp − plain) for T1 and B0; also restricted to gated-in windows.

    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.cp_vbar build
    PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.run_sam2_video --method t1 --prepend-dir results/benchmark/rr/composites_cpv
    ... --method b0 (same) ;  then:  ... cp_vbar score
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

from benchmark.nfo_vos import run_t4 as R, trials as TR, velocity as V
from benchmark.nfo_vos.factorised_gtfree import composite, theil_at
from benchmark.nfo_vos.run_sam2_video import gt_mask_native
from tracking.core.blob_tracker import merged_center

SEQ = 'seq1'
CDIR = f'{TR.RES}/composites_cpv'
IMG = 'images/benchmark/rr_checks/cp_vbar'
MIN_HIST = 3


def trials():
    return [t for t in json.load(open(TR.OUT)) if t['admissible'] and t['seq'] == SEQ]


def build():
    os.makedirs(CDIR, exist_ok=True); os.makedirs(IMG, exist_ok=True)
    pred, info = R.image_predictor(), {}
    for t in trials():
        t0, w0, w1 = t['t0'], *t['warmup']
        bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
        past = list(range(t0, max(-1, t0 - 13), -1))                     # reverse: local l = t0 - l
        Tk = R.track(dict(t, frames=past))
        g = {t0 - l: TR.to_native(*p)[0] for l, p in Tk['chain'].items()}
        hist = [k for k in range(t0 - 6, t0) if k in g]
        if len(hist) < MIN_HIST or t0 - 6 < 0:
            info[t['id']] = dict(prepend=False, n_hist=len(hist)); print(t['id'], 'gated out', len(hist)); continue
        ks = sorted(g)
        fx = theil_at(ks, [g[k] for k in ks], [t0] + list(range(t0 - 6, t0)))
        C = composite(SEQ, t0, {k: (fx[t0] - fx[k], 0.0) for k in range(t0 - 6, t0)}, bg)
        cx, cy = merged_center(Tk['dets'][0], *Tk['chain'][0], Tk['kw']['merge_radius'])
        with torch.inference_mode():
            pred.set_image(np.stack([np.clip(C, 0, 255).astype(np.uint8)] * 3, -1))
            m, _, _ = pred.predict(point_coords=np.array([TR.to_native(cx, cy)]), point_labels=np.array([1]),
                                   multimask_output=False)
        M = m[0] > 0
        P = np.where(M, C, bg)
        cv2.imwrite(f"{CDIR}/{t['id']}.jpg", np.clip(P, 0, 255).astype(np.uint8))
        x0, y0, x1, y1 = (int(v) for v in t['box_native'])
        raw = V.native(t0, SEQ)
        vis = [cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for a in (raw, C, P)]
        for v in vis:
            cv2.rectangle(v, (x0, y0), (x1, y1), (0, 255, 0), 2)
        vis[1][M] = (0.6 * vis[1][M] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.imwrite(f"{IMG}/{t['id']}.jpg", cv2.resize(np.hstack(vis), None, fx=0.5, fy=0.5))
        info[t['id']] = dict(prepend=True, n_hist=len(hist), area_MC=int(M.sum()),
                             area_box=int((x1 - x0) * (y1 - y0)))
        print(t['id'], info[t['id']], flush=True)
    json.dump(info, open(f'{CDIR}/info.json', 'w'), indent=1)


def score():
    from benchmark.nfo_vos.vhat_vs_silhouette import LDIR, st
    from benchmark.nfo_vos.run_sam2_video import gt_mask_native
    info = json.load(open(f'{CDIR}/info.json'))
    W, ids = {a: [] for a in ('t1', 't1-cpv', 't1-m0mc', 't1-om', 'b0', 'b0-cpv')}, []   # all local-env runs (-loc)
    for t in trials():
        L = np.load(f"{LDIR}/{t['id']}.npz")
        js = [int(j) for j in L.files]
        for a in W:
            src = a if info[t['id']]['prepend'] or '-' not in a or a == 't1-om' else a.split('-')[0]
            M = np.load(f"{TR.RES}/masks/{src}-loc/{t['id']}.npz")['masks'].astype(bool)
            W[a].append(np.mean([st(M[j], L[str(j)]) for j in js], 0))
        ids.append(t['id'])
    rng = np.random.default_rng(0)
    gin = np.array([info[i]['prepend'] for i in ids])
    print(f'seq1, {len(ids)} windows ({gin.sum()} with prepend), vs frozen silhouette label: J / P / R / F2')
    for a in W:
        print(f'  {a:7s} ' + ' / '.join(f'{v:.3f}' for v in np.mean(W[a], 0)))
    for b, arm in (('t1', 't1-cpv'), ('t1', 't1-m0mc'), ('t1', 't1-om'), ('b0', 'b0-cpv')):
        for sub, sel in (('all', np.ones_like(gin)), ('prepended', gin)):
            for i, m in ((0, 'J'), (1, 'P'), (2, 'R'), (3, 'F2')):
                d = (np.array(W[arm])[:, i] - np.array(W[b])[:, i])[sel]
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
                print(f'  Δ{m:2s} {arm} − {b} [{sub:9s} n={len(d):2d}] {d.mean():+.3f} '
                      f'[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] better {(d > 0).sum()}/{len(d)}')
    print('per window ΔJ t1:', {i: round(float(a[0] - b[0]), 3) for i, a, b in zip(ids, W['t1-cpv'], W['t1'])})


def show():
    """Per prepended window: planted frame (prompted: box + p*) | raw t0 with the T1-cpv mask | raw t0 with plain T1."""
    rows = []
    for t in trials():
        pth = f"{CDIR}/{t['id']}.jpg"
        if not os.path.exists(pth):
            continue
        x0, y0, x1, y1 = (int(v) for v in t['box_native'])
        raw = cv2.cvtColor(V.native(t['t0'], SEQ), cv2.COLOR_GRAY2BGR)
        pl = cv2.cvtColor(cv2.imread(pth, 0), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(pl, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(pl, tuple(int(v) for v in t['point_native']), 6, (0, 255, 0), -1)
        panes = [pl]
        for arm, col in (('t1-cpv-loc', (255, 0, 255)), ('t1-loc', (255, 255, 0))):
            m = np.load(f"{TR.RES}/masks/{arm}/{t['id']}.npz")['masks'][0].astype(bool)
            v = raw.copy(); v[m] = (0.5 * v[m] + 0.5 * np.array(col)).astype(np.uint8)
            panes.append(v)
        for v, txt in zip(panes, ('planted prepend (prompted)', 'raw t0: T1-cpv mask', 'raw t0: plain T1 mask')):
            cv2.putText(v, f"{t['id']}  {txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(cv2.resize(np.hstack(panes), None, fx=0.4, fy=0.4))
        cv2.imwrite(f"{IMG}/next_{t['id']}.jpg", rows[-1])
    cv2.imwrite(f'{IMG}/next_all.jpg', np.vstack(rows))


def comp_t0(t):
    """As build(): bg, the reverse-tracked chain over [t0-12, t0], and the composite at t0 (None if gated out)."""
    t0, (w0, w1) = t['t0'], t['warmup']
    bg = np.median(np.stack([V.native(i, SEQ) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    Tk = R.track(dict(t, frames=list(range(t0, max(-1, t0 - 13), -1))))
    g = {t0 - l: TR.to_native(*p)[0] for l, p in Tk['chain'].items()}
    if sum(k in g for k in range(t0 - 6, t0)) < MIN_HIST or t0 - 6 < 0:
        return bg, None
    ks = sorted(g)
    fx = theil_at(ks, [g[k] for k in ks], [t0] + list(range(t0 - 6, t0)))
    return bg, composite(SEQ, t0, {k: (fx[t0] - fx[k], 0.0) for k in range(t0 - 6, t0)}, bg)


def init_masks():
    """Composite mask as the t0 MASK prompt (memory insertion, 2026-10-16, fixed before running).
    SAM2's add_new_mask with use_mask_input_as_output_without_sam = True (both configs, sam2_base.py:923) outputs
    the given mask at t0 and memory-encodes (raw t0 features, mask): no prepended frame, no decoder call at t0.
    Masks (prompt on C and on bg: the plain runs' t0 box + p*):
      mc   M_C = l_C > 0                              (uncleaned; isolates the cleaning)
      z16  M_C ∧ static_z(f_C, f_B) > 3                (stride-16 cells where C differs from the empty scene;
                                                       sigmoid(z − 3) gate of diag_composite_diff.py:238)
      bgs  σ(l_C) · (1 − σ(l_B)) > 0.5                (l_B: the same prompt on the empty scene = what SAM2 would
                                                       segment there without the person)
    Gated-out windows (no history) keep the plain run."""
    from benchmark.nfo_vos.diag_composite_diff import static_z
    pred = R.image_predictor()
    sig = lambda x: 1 / (1 + np.exp(-x))
    for t in trials():
        bg, C = comp_t0(t)
        if C is None:
            continue
        kw = dict(point_coords=np.array([t['point_native']]), point_labels=np.array([1]),
                  box=np.array(t['box_native']), multimask_output=False, return_logits=True)
        out = {}
        with torch.inference_mode():
            for name, img in (('C', C), ('B', bg)):
                pred.set_image(np.stack([np.clip(img, 0, 255).astype(np.uint8)] * 3, -1))
                out[name] = (pred.predict(**kw)[0][0], pred._features['image_embed'][0].clone())
            z = static_z(out['C'][1], out['B'][1])[None, None]
            z = torch.nn.functional.interpolate(z, size=(600, 800), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        lC, lB = out['C'][0], out['B'][0]
        M = dict(mc=lC > 0, z16=(lC > 0) & (z > 3), bgs=sig(lC) * (1 - sig(lB)) > 0.5)
        np.savez_compressed(f"{CDIR}/m0_{t['id']}.npz", **M)
        P = gt_mask_native(SEQ, t['t0'])
        print(t['id'], ' '.join(f"{k} {m.sum():5d}/J {(m & P).sum() / max((m | P).sum(), 1):.2f}" for k, m in M.items()),
              flush=True)
        raw = cv2.cvtColor(V.native(t['t0'], SEQ), cv2.COLOR_GRAY2BGR)
        panes = []
        for k, m in (('C', None), *M.items()):
            v = cv2.cvtColor(np.clip(C, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) if m is None else raw.copy()
            if m is not None:
                v[m] = (0.5 * v[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
            cv2.putText(v, f"{t['id']} {k}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            panes.append(v)
        cv2.imwrite(f"{IMG}/m0_{t['id']}.jpg", cv2.resize(np.hstack(panes), None, fx=0.35, fy=0.35))


if __name__ == '__main__':
    {'build': build, 'score': score, 'show': show, 'init_masks': init_masks}[sys.argv[1]]()
