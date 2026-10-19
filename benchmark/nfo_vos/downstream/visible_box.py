"""Depth-restricted "visible" box (docs/vos_benchmark/plan_downstream_benchmark.md, Task 2; D3b).

Inside the GT box (whole person), pixels whose empty-scene disparity (results/benchmark/rr/depth/seq*_bg_disp_1036.npy,
Depth Anything V2 L, 0..1, larger = nearer) exceeds τ = the q-quantile of the disparity inside the box are near foliage
O; support S = box \\ O. Rows are trimmed from the top and bottom, then columns from the left and right, while their support
fraction is < ρ. A quantile, not an absolute τ, because the disparity is only affine-relative. Everything trimmed → GT
box, flagged. (q, ρ) selected on seq1 hand frames, tested once on seq2–4 against the hand-mask boxes.

  python -m benchmark.nfo_vos.downstream.visible_box select    # grid on seq1 (20 frames), prints and stops
  python -m benchmark.nfo_vos.downstream.visible_box test      # frozen (Q, RHO) on seq2–4 (23 frames), once
"""
import functools
import sys

import numpy as np

from benchmark.nfo_vos.downstream import box_metrics as BM

Q, RHO = None, None      # frozen after `select` (date + values)


@functools.lru_cache(maxsize=None)
def disp(seq):
    return np.load(f'results/benchmark/rr/depth/{seq}_bg_disp_1036.npy')


def visible_box(gt, d, q, rho):
    """(xyxy, fell_back). gt: xyxy inclusive native pixels; d: [H, W] disparity."""
    H, W = d.shape
    x0, y0 = max(int(round(gt[0])), 0), max(int(round(gt[1])), 0)
    x1, y1 = min(int(round(gt[2])), W - 1), min(int(round(gt[3])), H - 1)
    patch = d[y0:y1 + 1, x0:x1 + 1]
    S = patch <= np.quantile(patch, q)
    rows, cols = np.nonzero(S.mean(1) >= rho)[0], None
    if len(rows) == 0:
        return np.asarray(gt, np.float64), True
    S = S[rows[0]:rows[-1] + 1]
    cols = np.nonzero(S.mean(0) >= rho)[0]
    if len(cols) == 0:
        return np.asarray(gt, np.float64), True
    return np.array([x0 + cols[0], y0 + rows[0], x0 + cols[-1], y0 + rows[-1]], np.float64), False


def _iou(a, b):
    return float(BM.ope([a], [b])['iou'][0])


def compare(seqs, q, rho):
    """Per hand frame: IoU(visible, hand box) and IoU(GT, hand box)."""
    out = []
    for seq, k, h in BM.hand_frames():
        g = BM.gt_boxes(seq).get(k)
        if seq not in seqs or g is None:
            continue
        hb = BM.mask_box(h)
        v, fb = visible_box(g, disp(seq), q, rho)
        out.append((seq, k, _iou(v, hb), _iou(g, hb), fb, g, v, hb))
    return out


def select():
    print(' q    ρ    IoU(vis,hand)  IoU(GT,hand)  fallbacks   (seq1, 20 frames)')
    for q in (0.5, 0.6, 0.7, 0.8):
        for rho in (0.1, 0.2, 0.3):
            r = compare(('seq1',), q, rho)
            print(f'{q:.1f}  {rho:.1f}   {np.mean([x[2] for x in r]):.3f}          {np.mean([x[3] for x in r]):.3f}         {sum(x[4] for x in r)}')


def test(B=2000):
    assert Q is not None, 'freeze (Q, RHO) from `select` first'
    r = compare(('seq2', 'seq3', 'seq4'), Q, RHO)
    d = np.array([x[2] - x[3] for x in r])
    bs = d[np.random.default_rng(0).integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.quantile(bs, [.025, .975])
    print(f'seq2-4 n={len(r)}  IoU vis {np.mean([x[2] for x in r]):.3f}  GT {np.mean([x[3] for x in r]):.3f}  '
          f'Δ {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  fallbacks {sum(x[4] for x in r)}  → {"PASS" if lo > 0 else "FAIL"}')
    _montage(r)
    return r


def _montage(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from benchmark.nfo_vos import velocity as V
    n = len(r)
    fig, ax = plt.subplots(int(np.ceil(n / 6)), 6, figsize=(12, 2.2 * np.ceil(n / 6)))
    for a in ax.flat:
        a.axis('off')
    for a, (seq, k, iv, ig, fb, g, v, hb) in zip(ax.flat, r):
        a.imshow(V.native(k, seq), cmap='gray')
        for bb, c in ((g, 'lime'), (v, 'cyan'), (hb, 'magenta')):
            a.add_patch(plt.Rectangle(bb[:2], bb[2] - bb[0], bb[3] - bb[1], fill=False, ec=c, lw=1.2))
        a.set_title(f'{seq} {k} vis {iv:.2f} / GT {ig:.2f}', fontsize=7)
    fig.suptitle('GT box (green), depth-restricted visible box (cyan), hand-mask box (magenta)', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{BM.IMG}/visible_box_seq234.png', dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    {'select': select, 'test': test}[sys.argv[1]]()
