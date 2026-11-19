"""Hand J on the held-out seq2–4 polygon labels (<k>_handAv.png: 1 person, 2 void, void ignored), from the fullpass
crossing's mask at each hand frame; paired frame bootstrap of each arm − REF.

    NFO_RUN=fullpass PYTHONPATH=. python -m benchmark.nfo_vos.eval.heldout_j REF ARM [ARM ...]
"""
import json
import os
import sys

import numpy as np

from benchmark.nfo_vos import trials as TR
from benchmark.nfo_vos.eval.g2_score import paired


def hand_j(arm, seqs=('seq2', 'seq3', 'seq4')):
    trials = json.load(open(TR.OUT))
    out = {}
    for s in seqs:
        for k in TR.hand_frames(s):
            t = next((t for t in trials if t['seq'] == s and k in t['frames']), None)
            p = f'{TR.RES}/masks/{arm}/{t["id"]}.npz' if t else None
            if not p or not os.path.exists(p):
                continue
            m = np.load(p)['masks'][t['frames'].index(k)]
            H, keep = TR.hand_label(s, k)
            keep = np.ones_like(H) if keep is None else keep
            out[(s, k)] = (m & H & keep).sum() / max(((m | H) & keep).sum(), 1)
    return out


def main(ref, *arms):
    J = {a: hand_j(a) for a in (ref, *arms)}
    for a in (ref, *arms):
        d = '' if a == ref else '  Δ vs %s %+.3f [%+.3f, %+.3f] (n=%d)' % (ref, *paired(J[a], J[ref]))
        print(f'{a:20s} hand J {np.mean(list(J[a].values())):.3f} (n={len(J[a])}){d}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
