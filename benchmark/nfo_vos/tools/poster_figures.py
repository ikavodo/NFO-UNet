"""Figures of the conference poster (docs/poster/poster_latex/plots/nfo/). All numbers from the 32 whole crossings
(NFO_RUN=fullpass, findings §9, jobs 1950229 / 1950449), except the recognition-vs-N sweep (40-frame windows, seq2–4;
no whole-crossing sweep exists yet; the poster marks it).

    NFO_RUN=fullpass PYTHONPATH=. python -m benchmark.nfo_vos.tools.poster_figures [name ...]

names: examples, playsafe, bars, ablation, masksize, recogN, inputs, benchtiles (default: all). Needs the native hand
frames, hand labels, full-pass masks and the CSVs listed in CLAUDE.md; no GPU, no sam2 / ultralytics.
"""
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault('NFO_RUN', 'fullpass')
from benchmark.nfo_vos import trials as TR                       # noqa: E402  (reads NFO_RUN)
from benchmark.nfo_vos.eval import box_metrics as BM              # noqa: E402

OUT = 'docs/poster/poster_latex/plots/nfo'
# recog_vs_N / mask_fill palette
C_OURS, C_SAMURAI, C_B0, C_TRK, C_GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8984'
C_HAND = (40, 170, 60)                                           # visible person (RGB)
C_VOID = (200, 200, 200)                                         # "unsure" (RGB)
# figures are drawn at the poster's column width (≈ 9.3 in), so font sizes are print sizes in pt
plt.rcParams.update({'font.size': 18, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 19, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f'{OUT}/{name}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('wrote', f'{OUT}/{name}')


def tint(gray, mask, rgb, alpha=0.55):
    v = np.repeat(gray[..., None], 3, 2).astype(np.float32)
    v[mask] = (1 - alpha) * v[mask] + alpha * np.array(rgb, np.float32)
    return v.astype(np.uint8)


def label_view(seq, k):
    g = TR.native(k, seq)
    person, keep = TR.hand_label(seq, k)
    v = tint(g, person, C_HAND, 0.65).astype(np.float32)
    void = ~keep
    out = v
    out[void] = 0.25 * out[void] + 0.75 * np.array(C_VOID, np.float32)
    return out.astype(np.uint8)


def crop_box(seq, k, pad_x=0.9, pad_y=0.18):
    b = BM.gt_boxes(seq)[k]
    w, h = b[2] - b[0], b[3] - b[1]
    x0, x1 = int(max(b[0] - pad_x * w, 0)), int(min(b[2] + pad_x * w, 800))
    y0, y1 = int(max(b[1] - pad_y * h, 0)), int(min(b[3] + pad_y * h, 600))
    return x0, y0, x1, y1


def clip_of(seq, k):
    for t in json.load(open(TR.OUT)):
        if t['seq'] == seq and k in t['frames']:
            return t
    raise KeyError((seq, k))


def arm_mask(arm, t, k):
    d = np.load(f'{BM.MASKS}/{arm}/{t["id"]}.npz')
    return d['masks'][list(d['frames']).index(k)]


def jpr(m, person, keep):
    tp, fp, fn = (m & person & keep).sum(), (m & ~person & keep).sum(), (~m & person & keep).sum()
    return tp / (tp + fp + fn), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def examples(picks=(('seq2', 52), ('seq3', 287), ('seq4', 922))):
    """Top: whole frames (person box in green). Bottom: crop raw | crop with the hand label (green = visible person,
    grey = marked unsure by the annotator, ignored in scoring)."""
    fig = plt.figure(figsize=(9.3, 5.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.75, 1], hspace=0.2, wspace=0.04)
    for c, (seq, k) in enumerate(picks):
        g = TR.native(k, seq)
        b = BM.gt_boxes(seq)[k]
        v = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 4)
        ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2]); ax.imshow(v); ax.axis('off')
        ax.set_title(f'scene {seq[-1]}', fontsize=19, loc='left')
        x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.35, pad_y=0.06)
        person, keep = TR.hand_label(seq, k)
        unsure = (~keep).sum() / ((~keep).sum() + person.sum())
        for j, img in enumerate((cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), label_view(seq, k))):
            ax = fig.add_subplot(gs[1, 2 * c + j]); ax.imshow(img[y0:y1, x0:x1], aspect='auto'); ax.axis('off')
            if j:
                ax.set_title(f'{unsure:.0%} unsure', fontsize=13, color='#52514e', loc='right')
    save(fig, 'examples.png')


PLAYSAFE_OPTIONS = (('seq4', 293), ('seq4', 123), ('seq3', 615), ('seq3', 1110), ('seq4', 922), ('seq4', 486))


def playsafe_options():
    """Contact sheet of candidate frames for poster section 2 (SAM2.1 video vs the human label)."""
    playsafe(PLAYSAFE_OPTIONS, arms=(('b0-c2', 'SAM2.1 video', C_B0),), name='playsafe_options.png', label_rows=True)


def qual():
    """Results column: the section-2 frame again, now with our mask, plus a frame late in another crossing."""
    playsafe((('seq4', 293), ('seq4', 123)), arms=(('b0-c2', 'SAM2.1', C_B0), ('adapter-pf', 'ours', C_OURS)),
             name='qual.png', pad_x=0.5, row_h=3.6)


def playsafe(picks=(('seq4', 293),), arms=(('b0-c2', 'SAM2.1 video', C_B0),), name='playsafe.png', label_rows=False,
             pad_x=0.25, row_h=5.4):
    """Frame, human label, and each arm's mask with precision / recall, on hand frames inside a crossing."""
    n = 2 + len(arms)
    fig, axes = plt.subplots(len(picks), n, figsize=(9.3 * n / 4 if not label_rows else 7, row_h * len(picks) * (1 if not label_rows else 0.75)),
                             gridspec_kw=dict(wspace=0.04, hspace=0.15), squeeze=False)
    for r, (seq, k) in enumerate(picks):
        t = clip_of(seq, k)
        g = TR.native(k, seq)
        person, keep = TR.hand_label(seq, k)
        x0, y0, x1, y1 = crop_box(seq, k, pad_x=pad_x, pad_y=0.05)
        panes = [(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB),
                  f'{seq} {k} (option {r + 1})' if label_rows else f'frame {t["frames"].index(k)}', None),
                 (label_view(seq, k), 'human label', None)]
        for arm, aname, col in arms:
            m = arm_mask(arm, t, k)
            _, p, rc = jpr(m, person, keep)
            rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
            panes.append((tint(g, m, rgb, 0.65), aname, f'{p:.0%} right\nfinds {rc:.0%}'))
        for c, (img, title, note) in enumerate(panes):
            ax = axes[r, c]; ax.imshow(img[y0:y1, x0:x1], aspect='auto'); ax.axis('off')
            ax.set_title(title, fontsize=19, fontweight='bold' if note else 'normal')
            if note:
                ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=17, fontweight='bold', va='bottom',
                        color='white', bbox=dict(facecolor='#0b0b0b', alpha=0.7, lw=0, pad=4))
    save(fig, name)


# 32 whole crossings, seq1–4, one cluster env (findings §9, job 1950449); ours / noC = 3 seeds
SCORES = {  # name: (box AUC, hand J, recognition conf)
    'SAM2.1 video': (0.571, 0.733, 0.666),
    'SAMURAI': (0.579, 0.730, 0.663),
    'memoryless SAM2.1\n+ tracker hint': (0.628, 0.685, 0.596),
    'ours': (np.mean([0.674, 0.681, 0.678]), np.mean([0.757, 0.759, 0.752]), np.mean([0.662, 0.669, 0.671])),
}
SEEDS = {'ours': ((0.674, 0.681, 0.678), (0.757, 0.759, 0.752), (0.662, 0.669, 0.671))}
METRICS = ('follows the person\n(tracking success)', 'visible pixels right\n(overlap with human label)',
           'recognisable as a person\n(detector confidence)')


def bars():
    cols = (C_B0, C_SAMURAI, C_TRK, C_OURS)
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.2), gridspec_kw=dict(wspace=0.12))
    for m, ax in enumerate(axes):
        vals = [v[m] for v in SCORES.values()]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=cols, width=0.72)
        s = SEEDS['ours'][m]
        ax.errorbar(x[-1], vals[-1], yerr=[[vals[-1] - min(s)], [max(s) - vals[-1]]], color='#0b0b0b', capsize=5, lw=1.5)
        for xi, v in zip(x, vals):
            ax.annotate(f'{v:.2f}', (xi, v), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=16,
                        fontweight='bold' if xi == x[-1] else 'normal')
        ax.set_ylim(0, 0.85)                                # bars start at 0: no exaggerated differences
        ax.set_xticks([]); ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.set_title(METRICS[m].split('\n(')[0].replace(' ', '\n', 1), fontsize=18)
        ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    fig.legend(handles, ['SAM2.1 video', 'SAMURAI', 'SAM2.1 + tracker', 'ours'], loc='lower center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=16, columnspacing=0.8, handlelength=1)
    save(fig, 'bars.png')


# one ingredient at a time, 3-seed means (findings §9, job 1950449); line = 95% CI of the single comparison
# trk − SAMURAI, dots = the three seeds for the integral image
# chain from SAM2.1 video to ours; CIs: job 1953256 (b0trk-c2 − b0-c2, trk − b0trk-c2), findings §9
STEPS = (('+ tracker prompts\n(memory kept)', (0.089, 0.012, -0.029), ((0.062, 0.116), (-0.011, 0.036), (-0.054, 0.001)), None),
         ('− memory', (-0.033, -0.060, -0.040), ((-0.076, 0.004), (-0.113, -0.014), (-0.061, -0.021)), None),
         ('+ adapter, given the\nempty scene', (0.019, 0.055, 0.055), None, None),
         ('+ integral image\n(= ours)', (0.031, 0.016, 0.017), None,
          ((0.019, 0.035, 0.038), (0.005, 0.020, 0.022), (0.009, 0.022, 0.019))))


def ablation():
    """Rows: ingredients added one at a time; per row one bar per question (3-seed means). Line: 95% CI of the single
    comparison (none on this table, all steps from the SAM2.1 video baseline); dots: the three seeds (integral image)."""
    qs = (('tracking AUC', '#1f4e8c'), ('segmentation J', '#28aa3c'), ('recognition conf', '#8a5cc2'))
    fig, ax = plt.subplots(figsize=(9.3, 3.7))
    h = 0.26
    for r, (name, d, ci, seeds) in enumerate(STEPS):
        for m, (q, col) in enumerate(qs):
            y = -r - (m - 1) * h
            ax.barh(y, d[m], height=h * 0.9, color=col, label=q if r == 0 else None)
            if ci:
                ax.plot(ci[m], [y, y], color='#0b0b0b', lw=1.5)
            if seeds:
                ax.plot(seeds[m], [y] * 3, 'o', color='#0b0b0b', ms=4)
            r2 = np.sign(d[m]) * np.floor(abs(d[m]) * 100 + 0.5) / 100        # round half away from 0
            x = max(d[m], ci[m][1]) if ci and d[m] > 0 else min(d[m], ci[m][0]) if ci else d[m]
            x = max(x, *seeds[m]) if seeds else x
            ax.annotate(f'{r2:+.2f}', (x, y), xytext=(6 if d[m] > 0 else -6, 0), textcoords='offset points',
                        ha='left' if d[m] > 0 else 'right', va='center', fontsize=15)
    ax.axvline(0, color='#0b0b0b', lw=1)
    ax.set_yticks([-r for r in range(len(STEPS))]); ax.set_yticklabels([s[0] for s in STEPS], fontsize=17)
    ax.set_xlim(-0.15, 0.15); ax.set_xlabel('change vs the row above (first row: vs SAM2.1 video)')
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.35, 1.16), ncol=3, frameon=False, fontsize=15,
              handlelength=1, columnspacing=0.8)
    save(fig, 'ablation.png')


def masksize():
    """Mask area / person box area over a crossing, relative to frame 6: median over the 32 crossings."""
    from benchmark.nfo_vos.eval import mask_fill as MF
    rows = list(csv.DictReader(open(MF.CSV)))
    F = MF.fills(rows)
    lens = [len(f) for f in F['ours'].values()]
    kmax = max(k for k in range(MF.K0, max(lens)) if sum(L > k for L in lens) >= MF.MIN_N)
    k = np.arange(MF.K0, kmax + 1)
    # against memoryless SAM2.1 (trk) the late ratio is x0.87 [0.66, 1.13]: not resolved, so the poster compares to ours
    r = MF.late_ratio(F, 'b0-c2', list(F['b0-c2']))
    m, lo, hi, n, g = MF.boot_mean(r, list(r))
    print(f'SAM2.1 video / ours mask size, k >= {MF.K1}: x{np.exp(m):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}] ({n} clips)')
    fig, ax = plt.subplots(figsize=(9.3, 3.3))
    for a, label, col in (('b0-c2', 'SAM2.1', C_B0), ('b0trk-c2', 'SAM2.1 + tracker', '#8a5cc2'),
                          ('ours', 'ours, no memory', C_OURS)):
        yv, lo, hi = MF.median_curve(F[a], k)
        ax.fill_between(k, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(k, yv, color=col, lw=3.5, label=label)
    ax.axhline(1, color='#52514e', lw=1, ls=':')
    ax.set_xlim(MF.K0, kmax); ax.set_ylim(0.4, 1.4)
    ax.set_xlabel('frames since the prompt'); ax.set_ylabel('mask / box area\n(relative to frame 6)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=16,
              handlelength=1.2, columnspacing=1)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    save(fig, 'masksize.png')


def recogN():
    """Detector AP50 on masked N-frame stacks vs N (40-frame windows, seq2–4; recog_vs_N's CSVs)."""
    from benchmark.nfo_vos.eval import recog_vs_N as RN
    import importlib
    os.environ['NFO_RUN'] = 'rr'
    importlib.reload(TR); importlib.reload(BM)
    names = {'adapter-pf': 'ours, no memory', 't1-c2': 'SAMURAI', 'b0-c2': 'SAM2.1', 'trk': 'SAM2.1, no memory*',
             'nomask': 'no mask (plain average)'}
    fig, ax = plt.subplots(figsize=(9.3, 2.9))
    res = {}
    for N in RN.NS:
        rows = list(csv.DictReader(open(f'{BM.OUT}/recog_N{N}_uniform_meanfill_ablN.csv')))
        for a, *_ in RN.ARMS:
            res[a, N] = RN.ap_ci(rows, a, B=300)
    for a, _, col, ls in [x for x in RN.ARMS if x[0] != 't1-c2']:
        yv = np.array([res[a, N] for N in RN.NS])
        ax.plot(RN.NS, yv[:, 0], ls, color=col, lw=3 if a == 'adapter-pf' else 2.2, marker='o', ms=8, mec='white',
                mew=1.5, label=names[a], zorder=3 if a == 'adapter-pf' else 2)
    ax.axvline(5, color='#52514e', lw=1, ls=':')                     # N = 5: the headline's stack length
    ax.set_xticks(RN.NS); ax.set_xlabel('frames stacked N'); ax.set_ylabel('person detected (AP50)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=16)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    save(fig, 'recogN.png')


def inputs(row=2, cols=(0, 1, 2, 6)):
    """One row of the briefing's nfo_inputs.jpg (seq4 frame 43): frame, empty scene, integral image, our mask; the
    burnt-in labels are cut off (the poster labels the tiles)."""
    im = cv2.imread('docs/briefing/figs/nfo_inputs.jpg')
    h, w = im.shape[0] // 3, im.shape[1] // 7
    for c in cols:
        tile = im[row * h + 20:(row + 1) * h, c * w + 1:(c + 1) * w - 1]
        os.makedirs(OUT, exist_ok=True)
        cv2.imwrite(f'{OUT}/inputs_{c}.jpg', tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print('wrote', f'{OUT}/inputs_*.jpg')


def benchtiles(seq='seq4', k=293):
    """Three tiles for the three questions: person box vs mask box, hand label, a 7-frame recognition stack (from
    stack_N7_vs_N21.jpg, seq2). The mask is SAM2.1 video's: the poster introduces the benchmark before our method."""
    t = clip_of(seq, k)
    g = TR.native(k, seq)
    m = arm_mask('b0-c2', t, k)
    b = BM.gt_boxes(seq)[k]
    ys, xs = np.nonzero(m)
    v = tint(g, m, (42, 120, 214), 0.5)
    cv2.rectangle(v, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 3)
    cv2.rectangle(v, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (42, 120, 214), 3)
    x0, y0, x1, y1 = crop_box(seq, k, pad_x=0.25, pad_y=0.05)
    cv2.imwrite(f'{OUT}/bench_track.jpg', cv2.cvtColor(v[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{OUT}/bench_pixels.jpg', cv2.cvtColor(label_view(seq, k)[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    st = cv2.imread('images/benchmark/downstream/stack_N7_vs_N21.jpg')
    cv2.imwrite(f'{OUT}/bench_recog.jpg', st[20:296, 240:480])
    print('wrote', f'{OUT}/bench_*.jpg')


FIGS = dict(examples=examples, playsafe=playsafe, qual=qual, playsafe_options=playsafe_options, bars=bars, ablation=ablation, masksize=masksize, inputs=inputs,
            benchtiles=benchtiles, recogN=recogN)       # recogN last: it switches the run to rr

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
