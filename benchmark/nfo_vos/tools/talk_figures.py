"""Figures of the talk (docs/poster/poster_latex/plots/talk/) that the poster does not have. Same data as poster_figures
(32 whole crossings, NFO_RUN=fullpass); annotator B = <seq>_gt/<k>_handAv_annot2.png (seq2, seq3; 2026-11-01).

    NFO_RUN=fullpass PYTHONPATH=. python -m benchmark.nfo_vos.tools.talk_figures [name ...]

names: scenes, prompt, qual2, masksize_sam, davis, recogstack, recogN_numbers, recogN4, changemap (default: scenes,
prompt, qual2, masksize_sam, davis). changemap needs SAM2.1 on a GPU (cluster):
    NFO_RUN=rr PYTHONPATH=. python -m benchmark.nfo_vos.tools.talk_figures changemap recogstack needs the native frames of a whole crossing (cluster); the talk uses a crop of
images/benchmark/downstream/stack_N7_vs_N21.jpg instead (recog_plain_vs_masked.jpg). recogN_numbers and recogN4 need
ultralytics (ap50).
"""
import csv
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from benchmark.nfo_vos.tools import poster_figures as PF
from benchmark.nfo_vos.tools.poster_figures import TR, BM, C_B0, C_OURS, C_SAMURAI, C_TRK, C_GREY, C_HAND, C_VOID, tint, crop_box, clip_of, arm_mask, jpr
from benchmark.nfo_vos.eval.recognition import masked_stack, ap50

OUT = 'docs/poster/poster_latex/plots/talk'
C_A = (40, 170, 60)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/{name}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote', f'{OUT}/{name}')


def label_B(seq, k):
    v = cv2.imread(f'data/nfo_processed/{seq}_gt/{k:05d}_handAv_annot2.png', 0)
    return None if v is None else (v == 1, v != 2)


def view(g, lab):
    person, keep = lab
    v = tint(g, person, C_HAND, 0.65).astype(np.float32)
    if keep is not None:
        v[~keep] = 0.25 * v[~keep] + 0.75 * np.array(C_VOID, np.float32)
    return v.astype(np.uint8)


def scenes(picks=(('seq2', 335), ('seq3', 367), ('seq4', 293))):     # seq1 native frames are not on the laptop
    """One frame per NFO scene (scenes 2-4), hand-drawn person box in green."""
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.2), gridspec_kw=dict(wspace=0.03))
    for ax, (seq, k) in zip(axes, picks):
        v = cv2.cvtColor(TR.native(k, seq), cv2.COLOR_GRAY2RGB)
        b = BM.gt_boxes(seq)[k]
        cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 4)
        ax.imshow(v); ax.axis('off')
    save(fig, 'scenes.png')


def prompt(seq='seq3', k=287):
    """A perfect box prompt on a heavily occluded frame: most of the box is foliage."""
    g = TR.native(k, seq)
    b = BM.gt_boxes(seq)[k]
    x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.35, pad_y=0.06)
    v = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 3)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4.6), gridspec_kw=dict(wspace=0.04))
    axes[0].imshow(v[y0:y1, x0:x1], aspect='auto'); axes[0].set_title('box prompt', fontsize=16)
    axes[1].imshow(view(g, TR.hand_label(seq, k))[y0:y1, x0:x1], aspect='auto'); axes[1].set_title('human label', fontsize=16)
    for ax in axes:
        ax.axis('off')
    save(fig, 'prompt.png')


def recogstack(seq='seq2', k=1374, N=5):
    """Recognition input at frame k: the single frame, the plain aligned mean of N frames, and the SAM2.1-masked mean.
    Detector confidences from the benchmark CSVs (YOLO11m, best person detection with IoU >= 0.5 vs the hand box)."""
    t = clip_of(seq, k)
    boxes = BM.gt_boxes(seq)
    fr = t['frames'][t['frames'].index(k) - N + 1:t['frames'].index(k) + 1]
    c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    sh = [(c(boxes[k])[0] - c(boxes[i])[0], c(boxes[k])[1] - c(boxes[i])[1]) for i in fr]
    imgs = [cv2.cvtColor(TR.native(i, seq), cv2.COLOR_GRAY2RGB) for i in fr]
    ones = [np.ones(imgs[0].shape[:2], bool)] * N
    plain = masked_stack(imgs, ones, sh, imgs[-1])
    masked = masked_stack(imgs, [arm_mask('b0-c2', t, i) for i in fr], sh, imgs[-1])
    conf = {}
    for f, key in ((f'{BM.OUT}/recog_N5_uniform_meanfill_fp.csv', 'N5'), (f'{BM.OUT}/recog_N1_uniform_meanfill_fpN_nomask.csv', 'N1')):
        for r in csv.DictReader(open(f)):
            if r['seq'] == seq and int(r['frame']) == k and r['id'] == t['id']:
                conf[key, r['arm']] = float(r['conf'])
    x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.45, pad_y=0.08)
    b = boxes[k]
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.6), gridspec_kw=dict(wspace=0.04))
    for ax, img, title, cf in zip(axes, (imgs[-1], plain, masked),
                                  ('single frame', f'plain average, N = {N}', f'SAM2.1-masked, N = {N}'),
                                  (conf.get(('N1', 'nomask')), conf.get(('N5', 'nomask')), conf.get(('N5', 'b0-c2')))):
        v = img.copy()
        cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 2)
        ax.imshow(v[y0:y1, x0:x1], aspect='auto'); ax.axis('off'); ax.set_title(title, fontsize=15)
        ax.text(0.03, 0.03, f'person conf {cf:.2f}', transform=ax.transAxes, fontsize=14, fontweight='bold',
                color='white', bbox=dict(facecolor='#0b0b0b', alpha=0.7, lw=0, pad=3))
    print(f'{seq} {k} ({t["id"]}): conf', conf)
    save(fig, 'recogstack.png')


QUAL2 = (('seq2', 52), ('seq3', 1110))      # largest recall gain vs both annotators (qual2_candidates); seq3 1498 is the reverse case


def qual2(picks=QUAL2):
    """Frame, annotator A, annotator B, SAM2.1, Layer-SAM; P / R of each mask against A and against B."""
    fig, axes = plt.subplots(len(picks), 5, figsize=(12, 4.6 * len(picks)), gridspec_kw=dict(wspace=0.04, hspace=0.12),
                             squeeze=False)
    for r, (seq, k) in enumerate(picks):
        t = clip_of(seq, k)
        g = TR.native(k, seq)
        A, B = TR.hand_label(seq, k), label_B(seq, k)
        x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.3, pad_y=0.05)
        panes = [(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), f'scene {seq[-1]}', None), (view(g, A), 'annotator A', None),
                 (view(g, B), 'annotator B', None)]
        for arm, name, col in (('b0-c2', 'SAM2.1', C_B0), ('adapter-pf', 'Layer-SAM', C_OURS)):
            m = arm_mask(arm, t, k)
            (_, pa, ra), (_, pb, rb) = jpr(m, *A), jpr(m, *B)
            rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
            panes.append((tint(g, m, rgb, 0.65), name, f'A: P {pa:.0%} R {ra:.0%}\nB: P {pb:.0%} R {rb:.0%}'))
        for c, (img, title, note) in enumerate(panes):
            ax = axes[r, c]; ax.imshow(img[y0:y1, x0:x1], aspect='auto'); ax.axis('off')
            ax.set_title(title, fontsize=17, fontweight='bold' if note else 'normal')
            if note:
                ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom',
                        color='white', bbox=dict(facecolor='#0b0b0b', alpha=0.7, lw=0, pad=3))
    save(fig, 'qual2.png')


def qual2_candidates():
    """Hand frames of seq2/3 ranked by Layer-SAM's recall gain over SAM2.1, against both annotators."""
    rows = []
    for seq in ('seq2', 'seq3'):
        for k in TR.hand_frames(seq):
            t = clip_of(seq, k)
            A, B = TR.hand_label(seq, k), label_B(seq, k)
            s, o = arm_mask('b0-c2', t, k), arm_mask('adapter-pf', t, k)
            rows.append((min(jpr(o, *A)[2] - jpr(s, *A)[2], jpr(o, *B)[2] - jpr(s, *B)[2]), seq, k,
                         [round(v, 2) for v in (*jpr(s, *A), *jpr(o, *A), *jpr(s, *B), *jpr(o, *B))]))
    for r in sorted(rows, reverse=True):
        print(r)


def recogN_numbers(ns=(1, 3, 5, 7, 11, 15, 21), arms=('adapter-pf', 'b0-c2', 't1-c2', 'nomask')):
    for N in ns:
        out = []
        for a in arms:
            f = f'{BM.OUT}/recog_N5_uniform_meanfill_fp.csv' if N == 5 else f'{BM.OUT}/recog_N{N}_uniform_meanfill_fpN_{a}.csv'
            if not os.path.exists(f):
                out.append(f'{a} -'); continue
            win = {}
            for r in csv.DictReader(open(f)):
                if r['arm'] == a:
                    win.setdefault(r['id'], []).append([tuple(x) for x in json.loads(r['dets'])])
            out.append(f'{a} {ap50([d for v in win.values() for d in v]):.3f}' if win else f'{a} -')
        print(f'N={N:2d}: ' + ', '.join(out))


def davis(src=f'{OUT}/DAVIS-VOS.webp', keep=383):
    """The user's DAVIS / YouTube-Objects strip without the 0-100 % row below it."""
    im = cv2.imread(src)
    cv2.imwrite(f'{OUT}/DAVIS-VOS.png', im[:keep])
    print('wrote', f'{OUT}/DAVIS-VOS.png', im[:keep].shape)


def masksize_sam():
    """Mask area / box area over a crossing, relative to frame 6, for the SAM2.1 variants only (no Layer-SAM): memory
    with one prompt vs a box every frame, with and without memory. Median over the 32 crossings, 95 % CI."""
    from benchmark.nfo_vos.eval import mask_fill as MF
    F = MF.fills(list(csv.DictReader(open(MF.CSV))))
    lens = [len(f) for f in F['b0-c2'].values()]
    kmax = max(k for k in range(MF.K0, max(lens)) if sum(L > k for L in lens) >= MF.MIN_N)
    k = np.arange(MF.K0, kmax + 1)
    S = MF.trends(F)
    pct = lambda a: f'{(np.exp(MF.boot_mean(S[a], list(S[a]))[0]) - 1) * 100:+.0f}% / 10 fr.'
    fig, ax = plt.subplots(figsize=(9.3, 5.0))
    for a, label, col, ls in (('b0-c2', 'SAM2.1: memory, one prompt', C_B0, '-'), ('t1-c2', 'SAMURAI: memory, one prompt', C_SAMURAI, '-'),
                              ('trk', 'box every frame, no memory', C_TRK, '--'), ('b0trk-c2', 'box every frame + memory', '#8a5cc2', '--')):
        yv, lo, hi = MF.median_curve(F[a], k)
        if a in ('b0-c2', 'b0trk-c2'):
            ax.fill_between(k, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(k, yv, ls, color=col, lw=3, label=f'{label} ({pct(a)})')
    ax.axhline(1, color='#52514e', lw=1, ls=':')
    ax.set_xlim(MF.K0, kmax); ax.set_ylim(0.4, 1.4)
    ax.set_xlabel('frames since the prompt'); ax.set_ylabel('mask / box area\n(relative to frame 6)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=14, handlelength=1.6)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    save(fig, 'masksize_sam.png')


def recogN4(ns=(1, 3, 5, 7, 11, 15, 21)):
    """poster_figures.recogN with SAMURAI instead of the memoryless tracker arm: YOLO11 AP50 vs N, 32 crossings."""
    PF.OUT = OUT
    PF.RECOG_ARMS = (('adapter-pf', 'Layer-SAM', C_OURS, '-'), ('b0-c2', 'SAM2.1', C_B0, '-'),
                     ('t1-c2', 'SAMURAI', C_SAMURAI, '-'), ('nomask', 'no mask (plain average)', C_GREY, '--'))
    PF.recogN(ns)
    os.replace(f'{OUT}/recogN.png', f'{OUT}/recogN4.png')
    print('renamed to', f'{OUT}/recogN4.png')


def changemap(seq='seq4', k=43):
    """The tracker's input on one frame: frame t with the kernel tracker's box, the empty scene B, and the change map
    m_t = 1 - cos(F_t, F_B) of SAM2.1 b+ stride-16 features (adapter/tracker.feat_maps), cropped around the person.
    seq4 frame 43 = the frame of the poster's input crops (plots/nfo/inputs_*.jpg)."""
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    from benchmark.nfo_vos.adapter.tracker import feat_maps, kernel16_track
    t = clip_of(seq, k)
    pred = image_predictor()
    c, (w, h) = kernel16_track(t, pred)
    w0, w1 = t['warmup']
    bg = np.median(np.stack([TR.native(i, seq) for i in range(w0, w1 + 1)]), 0).astype(np.float32)
    x = TR.native(k, seq).astype(np.float32)
    m = feat_maps(pred, x[None], bg)[0]
    cx, cy = c[t['frames'].index(k)]
    box = (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))
    x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.5, pad_y=0.08)
    rgb = lambda g: cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    heat = (plt.get_cmap('magma')(np.clip(m / max(np.percentile(m, 99.5), 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)
    panes = [rgb(x), rgb(bg), np.ascontiguousarray(heat)]
    for v in (panes[0], panes[2]):
        cv2.rectangle(v, box[:2], box[2:], (230, 40, 230), 3)
    fig, axes = plt.subplots(1, 3, figsize=(9, 4.6), gridspec_kw=dict(wspace=0.04))
    for ax, v in zip(axes, panes):
        ax.imshow(v[y0:y1, x0:x1], aspect='auto'); ax.axis('off')
    save(fig, 'changemap.png')


C_UNSURE = (230, 184, 0)     # the talk's single "unsure" colour (yellow); the poster uses grey


def yellow(prompt_pick=('seq2', 132)):
    """Talk copies of the poster's label figures with unsure in yellow: prompt.png (a frame with small unsure patches),
    examples.png, playsafe.png, qual.png (all to plots/talk/)."""
    global C_VOID
    C_VOID = PF.C_VOID = C_UNSURE
    PF.OUT = OUT
    prompt(*prompt_pick)
    PF.examples(); PF.playsafe(); PF.qual()


def annot_pair(picks=(('seq2', 132), ('seq3', 615))):
    """Annotator A vs B on frames near the median agreement (J about 0.65): frame | A | B | difference."""
    fig, axes = plt.subplots(len(picks), 4, figsize=(10, 3.2 * len(picks)), gridspec_kw=dict(wspace=0.03, hspace=0.06))
    for r, (seq, k) in enumerate(picks):
        g = TR.native(k, seq)
        pa, ka = TR.hand_label(seq, k)
        pb, kb = label_B(seq, k)
        x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.3, pad_y=0.05)
        def lab(p, keep):
            v = tint(g, p, C_HAND, 0.65).astype(np.float32)
            v[~keep] = 0.35 * v[~keep] + 0.65 * np.array(C_UNSURE, np.float32)
            return v.astype(np.uint8)
        d = np.repeat(g[..., None], 3, 2).astype(np.float32) * 0.6
        a_only, b_only = pa & ~pb & ka & kb, pb & ~pa & ka & kb
        d[a_only] = (230, 40, 200); d[b_only] = (0, 200, 230)
        m = ka & kb
        J = (pa & pb & m).sum() / ((pa | pb) & m).sum()
        for c, img in enumerate((cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), lab(pa, ka), lab(pb, kb), d.astype(np.uint8))):
            ax = axes[r, c]; ax.imshow(img[y0:y1, x0:x1], aspect='auto'); ax.axis('off')
            if r == 0:
                ax.set_title(('frame', 'annotator A', 'annotator B', 'difference')[c], fontsize=16, loc='center')
        axes[r, 3].text(0.97, 0.03, f'J {J:.2f}', transform=axes[r, 3].transAxes, ha='right', va='bottom', fontsize=15,
                        color='white', fontweight='bold', bbox=dict(facecolor='#0b0b0b', alpha=0.7, lw=0, pad=3))
    save(fig, 'annot_pair.png')


def clip(seq='seq3', k=287, w=480):
    """One NFO window as a frame sequence with the GT box, for \\animategraphics in the talk (plots/talk/clip/f<i>.jpg),
    plus clip.mp4 for viewers without JavaScript."""
    t = clip_of(seq, k)
    os.makedirs(f'{OUT}/clip', exist_ok=True)
    s = w / 800
    vw = cv2.VideoWriter(f'{OUT}/clip.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, int(600 * s)))
    boxes = BM.gt_boxes(seq)
    for i, f in enumerate(t['frames']):
        v = cv2.resize(cv2.cvtColor(TR.native(f, seq), cv2.COLOR_GRAY2BGR), (w, int(600 * s)), interpolation=cv2.INTER_AREA)
        if f in boxes:
            b = boxes[f]
            cv2.rectangle(v, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), (60, 220, 60), 2)
        cv2.imwrite(f'{OUT}/clip/f{i}.jpg', v, [cv2.IMWRITE_JPEG_QUALITY, 85])
        vw.write(v)
    vw.release()
    open(f'{OUT}/clip/n.tex', 'w').write(f'\\def\\ClipLast{{{len(t["frames"]) - 1}}}\n')
    print('wrote', len(t['frames']), 'frames of', t['id'], 'to', f'{OUT}/clip/')


if __name__ == '__main__':
    names = sys.argv[1:] or ['scenes', 'prompt', 'qual2', 'masksize_sam', 'davis']
    for n in names:
        globals()[n]()
