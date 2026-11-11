"""Gait (walk / jog) from stabilised masks under fragmented occlusion: pilot (design + pre-registered criteria:
docs/vos_benchmark/2026-11-11-gait-from-masks-design.md).

    PYTHONPATH=. python -m benchmark.nfo_vos.eval.gait_masks kth     # SAM2 silhouettes of KTH walking/jogging -> canvases
    PYTHONPATH=. python -m benchmark.nfo_vos.eval.gait_masks nfo     # canvases of every NFO arm on the benchmark windows
    PYTHONPATH=. python -m benchmark.nfo_vos.eval.gait_masks fit     # one logistic regression per k, KTH CV, NFO table
Needs data/kth/{walking,jogging}.zip (www.csc.kth.se/cvap/actions/) and kth_labels.zip (repo root, KTH GT boxes).
"""
import glob
import json
import os
import re
import sys
import zipfile

import cv2
import numpy as np

KTH, LABELS = 'data/kth', 'kth_labels.zip'
OUT, IMG = 'results/benchmark/gait', 'images/benchmark/gait'
CH, CW = 24, 16                         # canvas: 1.2 x median GT box height, motion always rightward
KS = (1, 4, 8, 16, 24)
SCEN = ('d1', 'd3', 'd4')               # d2 zooms
NFO_ARMS = ('oracle', 'box', 'b0-c2', 't1-c2', 'trk', 'adapter-pf', 'adapter-s1', 'adapter-s2')
OURS = ('adapter-pf', 'adapter-s1', 'adapter-s2')


def trajectory(boxes):
    """boxes [L, 4] (x0, y0, x1, y1) px -> per-frame canvas centre (linear fit in x, median y), scale h, flip."""
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    t = np.arange(len(boxes))
    a, b = np.polyfit(t, cx, 1)
    return a * t + b, np.full(len(t), np.median(cy)), float(np.median(boxes[:, 3] - boxes[:, 1])), a < 0


def canvas(mask, cx, cy, h, flip):
    H = 1.2 * h; W = H * CW / CH
    x0, y0 = int(round(cx - W / 2)), int(round(cy - H / 2)); x1, y1 = x0 + int(round(W)), y0 + int(round(H))
    p = max(0, -x0, -y0, x1 - mask.shape[1], y1 - mask.shape[0])
    m = np.pad(mask.astype(np.float32), p)[y0 + p:y1 + p, x0 + p:x1 + p]
    c = cv2.resize(m, (CW, CH), interpolation=cv2.INTER_AREA)
    return c[:, ::-1] if flip else c


# ---------------------------------------------------------------- KTH
def kth_crossings():
    """{video: [(start, end), ...]} crossings of KTH walking/jogging d1/d3/d4 from the GT boxes: visible runs that move
    > 1 body height, trimmed to frames whose box does not touch the image border."""
    z, out = zipfile.ZipFile(LABELS), {}
    for n in z.namelist():
        m = re.match(r'kth/(person(\d+)_(walking|jogging)_(d\d)_uncomp)/groundtruth.txt', n)
        if not m or m[4] not in SCEN:
            continue
        a = np.array([[float(v) for v in l.split(',')] for l in z.read(n).decode().split()])
        ok = (a[:, 0] > 0.005) & (a[:, 0] + a[:, 2] < 0.995) & (a[:, 0] >= 0)
        runs, i = [], 0
        while i < len(a):
            if not ok[i]: i += 1; continue
            j = i
            while j < len(a) and ok[j]: j += 1
            xc = a[i:j, 0] + a[i:j, 2] / 2
            if j - i >= 8 and (xc.max() - xc.min()) * 160 > np.median(a[i:j, 3]) * 120:
                runs.append((i, j - 1))
            i = j
        if runs:
            out[m[1]] = dict(subject=int(m[2]), label=int(m[3] == 'jogging'), gt=a, runs=runs)
    return out


def read_avi(name):
    act = name.split('_')[1]
    p = f'{KTH}/videos/{name}.avi'
    if not os.path.exists(p):
        os.makedirs(f'{KTH}/videos', exist_ok=True)
        zipfile.ZipFile(f'{KTH}/{act}.zip').extract(f'{name}.avi', f'{KTH}/videos')
    cap, fr = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f[:, :, ::-1].copy())
    return fr


def occluder(seed, H=120, W=160, h=95):
    """Static procedural foliage in frame coordinates (synth/occluders.occ_perlin, as in adapter training): rho ~
    U[0.2, 0.6] (10% rho = 0), clump cell 0.06 h or 0.12 h (my call: GPJATK clump 12/24 px rescaled to KTH height)."""
    import torch
    from benchmark.nfo_vos.synth.occluders import occ_perlin
    rng = np.random.default_rng(seed)
    if rng.random() < 0.1:
        return np.zeros((H, W), bool)
    c = max(2, int(round(h * (0.06, 0.12)[rng.integers(2)])))
    _, o = occ_perlin(torch.zeros(1, 1, H, W), density=float(rng.uniform(0.2, 0.6)), noise_h=c, noise_w=c, movement_x=0,
                      movement_y=0, light_movement_x=0, light_movement_y=0, seed=int(seed), return_mask=True)
    return o[0].numpy().astype(bool)


def main_kth():
    import torch
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    P, rows = image_predictor(), []
    for vi, (name, v) in enumerate(sorted(kth_crossings().items())):
        fr = read_avi(name)
        for ri, (s, e) in enumerate(v['runs']):
            e = min(e, len(fr) - 1)
            g = v['gt'][s:e + 1]
            boxes = np.stack([g[:, 0] * 160, g[:, 1] * 120, (g[:, 0] + g[:, 2]) * 160, (g[:, 1] + g[:, 3]) * 120], 1)
            cx, cy, h, flip = trajectory(boxes)
            occ = occluder(vi * 100 + ri, h=h)
            cl, oc = [], []
            for k, f in enumerate(range(s, e + 1)):
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    P.set_image(fr[f])
                    m = P.predict(box=boxes[k], multimask_output=False)[0][0] > 0
                cl.append(canvas(m, cx[k], cy[k], h, flip)); oc.append(canvas(m & ~occ, cx[k], cy[k], h, flip))
            rows.append(dict(name=name, subject=v['subject'], label=v['label'], clean=np.array(cl, np.float16),
                             occ=np.array(oc, np.float16)))
        print(f'{vi + 1} {name}: {len(v["runs"])} crossings', flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(f'{OUT}/kth_canvases.npy', np.array(rows, dtype=object), allow_pickle=True)
    print('wrote', len(rows), 'KTH crossings')


# ---------------------------------------------------------------- NFO
def main_nfo():
    from benchmark.nfo_vos import trials as T
    trials = [t for t in json.load(open('results/benchmark/rr/trials.json')) if t['admissible']]
    rows = []
    for t in trials:
        seq, b = t['seq'], T.bbs(t['seq'])
        seg = range(t['seg_start'], t['seg_end'] + 1)
        nb = np.array([T.gt_to_native(b[f][0], T.NATIVE_W, T.NATIVE_H) for f in seg], float)
        cx, cy, h, flip = trajectory(nb)
        idx = [f - t['seg_start'] for f in t['frames']]
        C = {}
        for arm in NFO_ARMS:
            if arm == 'oracle':
                ms = [cv2.imread(f'data/nfo_processed/{seq}_gt/{f:05d}_sammask_native.png', 0) > 127 for f in t['frames']]
            elif arm == 'box':
                ms = []
                for i in idx:
                    m = np.zeros((T.NATIVE_H, T.NATIVE_W), bool); x0, y0, x1, y1 = np.round(nb[i]).astype(int)
                    m[max(y0, 0):y1, max(x0, 0):x1] = True; ms.append(m)
            else:
                z = np.load(f'results/benchmark/rr/masks/{arm}/{t["id"]}.npz')
                assert list(z['frames']) == list(t['frames']), (arm, t['id'])
                ms = z['masks']
            C[arm] = np.array([canvas(m, cx[i], cy[i], h, flip) for m, i in zip(ms, idx)], np.float16)
        rows.append(dict(id=t['id'], seq=seq, crossing=f"{seq}_s{t['seg_idx']}", label=int(t['gait'] == 'run'), C=C))
        print(t['id'], flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(f'{OUT}/nfo_canvases.npy', np.array(rows, dtype=object), allow_pickle=True)


# ---------------------------------------------------------------- fit + report
def windows(seq, k):
    seq = np.asarray(seq, np.float32)
    return np.stack([seq[t - k + 1:t + 1].ravel() for t in range(k - 1, len(seq))]) if len(seq) >= k else None


def kth_xy(rows, kind, k, idx=None):
    X, y, g = [], [], []
    for r in (rows if idx is None else [rows[i] for i in idx]):
        w = windows(r[kind], k)
        if w is not None:
            X.append(w); y += [r['label']] * len(w); g += [r['subject']] * len(w)
    return np.concatenate(X), np.array(y), np.array(g)


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main_fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    kth = list(np.load(f'{OUT}/kth_canvases.npy', allow_pickle=True))
    nfo = list(np.load(f'{OUT}/nfo_canvases.npy', allow_pickle=True))
    subj = np.array([r['subject'] for r in kth])
    print(f'KTH: {len(kth)} crossings ({sum(r["label"] for r in kth)} jogging), {len(set(subj))} subjects; '
          f'NFO: {len(nfo)} windows, {len(set(r["crossing"] for r in nfo))} crossings')
    res = []
    # nfo-seq1: trained on the seq1 oracle masks (real foliage), tested on seq2-4 only; groups = seq1 crossings
    s1 = [dict(subject=r['crossing'], label=r['label'], **{'nfo-seq1': r['C']['oracle']}) for r in nfo if r['seq'] == 'seq1']
    for kind in ('occ', 'clean', 'nfo-seq1'):
        train = s1 if kind == 'nfo-seq1' else kth
        subj = np.array([r['subject'] for r in train])
        for k in KS:
            # C by subject-grouped 5-fold CV on KTH (accuracy on held-out subjects, same masks kind)
            cv = {}
            for C in (1e-3, 1e-2, 1e-1):
                acc = []
                for tr, te in GroupKFold(4).split(np.zeros(len(train)), groups=subj):
                    X, y, _ = kth_xy(train, kind, k, tr); Xt, yt, _ = kth_xy(train, kind, k, te)
                    acc.append((LogisticRegression(C=C, max_iter=3000).fit(X, y).predict(Xt) == yt).mean())
                cv[C] = float(np.mean(acc))
            C = max(cv, key=cv.get)
            X, y, _ = kth_xy(train, kind, k)
            clf = LogisticRegression(C=C, max_iter=3000).fit(X, y)
            for arm in NFO_ARMS:
                per = {}
                for r in nfo:
                    w = windows(r['C'][arm], k)
                    per.setdefault(r['crossing'], []).extend((clf.predict(w) == r['label']).tolist())
                for cr, v in per.items():
                    res.append(dict(kind=kind, k=k, C=C, kth_cv=cv[C], arm=arm, crossing=cr, acc=float(np.mean(v))))
            print(f'{kind} k={k:2d} C={C:g} KTH-CV {cv[C]:.3f}', flush=True)
    import csv
    with open(f'{OUT}/gait_acc.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    report(res)


def report(res):
    for kind in ('occ', 'clean', 'nfo-seq1'):
        for split, keep in (('seq1-4', lambda c: True), ('seq2-4', lambda c: not c.startswith('seq1'))):
            print(f'\n== train {kind}, NFO {split}: per-frame accuracy (mean over crossings)')
            print('k  ' + ' '.join(f'{a:>10s}' for a in NFO_ARMS + ('ours',)) + '   KTH-CV')
            for k in KS:
                if kind == 'nfo-seq1' and split == 'seq1-4':
                    continue
                A = {a: {r['crossing']: r['acc'] for r in res if r['kind'] == kind and r['k'] == k and r['arm'] == a
                         and keep(r['crossing'])} for a in NFO_ARMS}
                cr = sorted(A['oracle'])
                A['ours'] = {c: np.mean([A[a][c] for a in OURS]) for c in cr}
                kc = next(r['kth_cv'] for r in res if r['kind'] == kind and r['k'] == k)
                print(f'{k:2d} ' + ' '.join(f'{np.mean([A[a][c] for c in cr]):10.3f}' for a in NFO_ARMS + ('ours',))
                      + f'   {kc:.3f}')
                if k == 16:
                    for x, y in (('oracle', 'box'), ('ours', 't1-c2'), ('ours', 'box'), ('t1-c2', 'box')):
                        m, lo, hi = boot(np.array([A[x][c] - A[y][c] for c in cr]))
                        print(f'     k=16 {x} - {y}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(cr)} crossings)')


def montage():
    """Canvases for a visual check: KTH clean/occ (one walk, one jog) and every NFO arm (one walk, one run window)."""
    kth = list(np.load(f'{OUT}/kth_canvases.npy', allow_pickle=True))
    nfo = list(np.load(f'{OUT}/nfo_canvases.npy', allow_pickle=True))
    up = lambda c: cv2.resize((np.asarray(c, np.float32) * 255).astype(np.uint8), (CW * 4, CH * 4), interpolation=cv2.INTER_NEAREST)
    strip = lambda seq, lab: np.vstack([lab_bar(lab, 16 * CW * 4 + 15 * 2), np.hstack(sum([[up(c), np.full((CH * 4, 2), 128, np.uint8)] for c in seq[:16]], [])[:-1])])
    rows = []
    for lab in (0, 1):
        r = next(r for r in kth if r['label'] == lab and len(r['clean']) >= 16)
        rows += [strip(r['clean'], f"KTH {r['name']} clean"), strip(r['occ'], f"KTH {r['name']} occ-aug")]
    for lab in (0, 1):
        r = next(r for r in nfo if r['label'] == lab and r['seq'] == 'seq2')
        rows += [strip(r['C'][a][12:], f"NFO {r['id']} {('walk', 'run')[lab]} {a}") for a in NFO_ARMS]
    os.makedirs(IMG, exist_ok=True)
    cv2.imwrite(f'{IMG}/canvases.png', np.vstack(rows))
    print('wrote', f'{IMG}/canvases.png')


def lab_bar(text, w):
    b = np.zeros((16, w), np.uint8)
    cv2.putText(b, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    return b


def montage_compact(n=6, step=3, U=3):
    """One row per class (walk / jog=run): blocks of n canvases (every `step` frames) per input source, side by side."""
    kth = list(np.load(f'{OUT}/kth_canvases.npy', allow_pickle=True))
    nfo = list(np.load(f'{OUT}/nfo_canvases.npy', allow_pickle=True))
    src = (('KTH clean', None, 'clean'), ('KTH +synthetic foliage', None, 'occ'), ('NFO oracle (SAM2, GT box)', 'oracle', 0),
           ('NFO GT box', 'box', 0), ('NFO B0 SAM2.1 video', 'b0-c2', 0), ('NFO T1 SAMURAI', 't1-c2', 0),
           ('NFO trk (SAM2+tracker)', 'trk', 0), ('NFO ours (Layer-SAM)', 'adapter-pf', 0))
    up = lambda c: cv2.resize((np.asarray(c, np.float32) * 255).astype(np.uint8), (CW * U, CH * U), interpolation=cv2.INTER_NEAREST)
    rows = []
    for lab in (0, 1):
        rk = next(r for r in kth if r['label'] == lab and len(r['clean']) >= n * step)
        rn = next(r for r in nfo if r['label'] == lab and r['seq'] == 'seq2')
        blocks = []
        for name, arm, kind in src:
            seq = rk[kind] if arm is None else rn['C'][arm][12:]
            b = np.hstack([np.pad(up(seq[i * step]), ((0, 0), (0, 1)), constant_values=90) for i in range(n)])
            blocks.append(np.vstack([lab_bar(name, b.shape[1]), b]))
        row = np.hstack(sum([[x, np.full((x.shape[0], 10), 255, np.uint8)] for x in blocks], [])[:-1])
        title = f"{('WALK', 'RUN (KTH: jogging)')[lab]}   KTH {rk['name']}   NFO {rn['id']}   every {step}th frame"
        rows += [lab_bar(title, row.shape[1]), row, np.full((8, row.shape[1]), 255, np.uint8)]
    os.makedirs(IMG, exist_ok=True)
    cv2.imwrite(f'{IMG}/canvases_compact.png', np.vstack(rows))
    print('wrote', f'{IMG}/canvases_compact.png', np.vstack(rows).shape)


if __name__ == '__main__':
    {'kth': main_kth, 'nfo': main_nfo, 'fit': main_fit, 'montage': montage, 'compact': montage_compact}[sys.argv[1]]()
