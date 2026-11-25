"""Box-extent bias of mask arms vs GT boxes, on frames where every listed arm tracks (IoU > 0): mean IoU, log width / height
ratio (pred / GT), centre offset normalised by GT size. Separates extent error from tracking failure (box AUC mixes both).

    NFO_RUN=fullpass PYTHONPATH=. python -m benchmark.nfo_vos.tools.box_bias SEQ_PREFIX ARM1 ARM2 ...   (e.g. seq1, seq2-4: 'seq[234]')
"""
import json
import re
import sys

import numpy as np

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.eval.box_metrics import _iou, _xywh, gt_boxes, load_arm, mask_box


def main(seqs, *arms):
    st = {a: [] for a in arms}                                         # per clip: (iou, logw, logh, dx, dy) means
    for t in json.load(open(TR.OUT)):
        if not re.fullmatch(seqs, t['seq']):
            continue
        gtb, per = gt_boxes(t['seq']), {}
        for a in arms:
            la = load_arm(a, t['id'])
            if la is None:
                break
            ms, fr = la
            per[a] = {int(k): mask_box(ms[i]) for i, k in enumerate(fr) if int(k) in gtb}
        if len(per) < len(arms):
            continue
        fs = sorted(set.intersection(*[set(p) for p in per.values()]))
        g = np.stack([_xywh(gtb[f]) for f in fs])
        P = {a: np.stack([_xywh(per[a][f]) for f in fs]) for a in arms}
        ok = np.all([_iou(P[a], g) > 0 for a in arms], 0)
        for a in arms:
            p, gg = P[a][ok], g[ok]
            c = (p[:, :2] + p[:, 2:] / 2 - gg[:, :2] - gg[:, 2:] / 2) / gg[:, 2:]
            st[a].append([_iou(p, gg).mean(), np.log(p[:, 2] / gg[:, 2]).mean(), np.log(p[:, 3] / gg[:, 3]).mean(),
                          c[:, 0].mean(), c[:, 1].mean(), ok.mean()])
    rng = np.random.default_rng(0)
    print(f"{seqs}: {len(st[arms[0]])} clips; frames where all arms track; log w/h = log(pred/GT); dx, dy in GT widths")
    for a in arms:
        m = np.mean(st[a], 0)
        print(f"  {a:22s} IoU {m[0]:.3f}  log w {m[1]:+.3f}  log h {m[2]:+.3f}  dx {m[3]:+.3f}  dy {m[4]:+.3f}  (tracked {m[5]:.2f})")
    for a in arms[1:]:
        d = np.array(st[a]) - np.array(st[arms[0]])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        print(f"  {a} − {arms[0]}: " + '  '.join(f"{k} {d.mean(0)[j]:+.3f} [{lo[j]:+.3f},{hi[j]:+.3f}]"
                                                  for j, k in enumerate(('IoU', 'log w', 'log h', 'dx', 'dy'))))


if __name__ == '__main__':
    main(*sys.argv[1:])
