"""Evidence for the walk/run label of each NFO crossing (trials.segments: 2-means on speed_h, trials.py:142).

Two checks that do not depend on that threshold:
  1. cadence: step period P (frames) = first autocorrelation peak of the detrended GT box width. The box widens at
     every step as the legs spread, so P is the step period. Running has a shorter P and a longer step
     (speed_h * P, in body heights) than walking. Neither quantity needs the frame rate.
  2. montage: crops from the native frames of each crossing, for a visual check (flight phase, forward lean,
     bent arms). Saved to images/benchmark/gait/<seq>.jpg.

    PYTHONPATH=. NFO_RUN=rr python -m benchmark.nfo_vos.tools.gait_check
"""
import os
import cv2
import numpy as np
from benchmark.nfo_vos import trials as T

OUT = 'images/benchmark/gait'
NATIVE = 'data/nfo_final/nfo_final'


def step_period(w, lo=4, hi=40):
    w = np.asarray(w, float)
    d = w - np.convolve(w, np.ones(15) / 15, mode='same')
    d = d[7:-7] - d[7:-7].mean()
    ac = np.correlate(d, d, 'full')[len(d) - 1:]
    ac /= ac[0]
    pk = [k for k in range(lo, min(hi, len(ac) - 1)) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0]
    return (pk[0], float(ac[pk[0]])) if pk else (None, None)


def crops(seq, seg, n=8, H=110):
    b = T.bbs(seq)
    mid = (seg['start'] + seg['end']) // 2
    idx = [mid + 3 * (k - n // 2) for k in range(n)]                   # every 3rd frame around mid-crossing
    out = []
    for i in idx:
        im = cv2.imread(f'{NATIVE}/{seq}/{i:05d}.jpg')
        x0, y0, x1, y1 = [int(round(v)) for v in T.gt_to_native(b[i][0], im.shape[1], im.shape[0])]
        pad = int(0.15 * (y1 - y0))
        c = im[max(0, y0 - pad):y1 + pad, max(0, x0 - 2 * pad):x1 + 2 * pad]
        out.append(cv2.resize(c, (int(c.shape[1] * H / c.shape[0]), H)))
    w = max(c.shape[1] for c in out)
    return np.hstack([cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1] + 4, cv2.BORDER_CONSTANT, value=0) for c in out])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for seq in T.ALL_SEQS:
        b = T.bbs(seq)
        strips = []
        for s in T.segments(seq):
            w = [b[k][0].w for k in range(s['start'], s['end'] + 1)]
            P, r = step_period(w)
            step = s['speed_h'] * P if P else float('nan')
            rows.append((seq, s['idx'], s['dir'], s['gait'], s['L'], s['speed_h'], P, r, step))
            print(f"{seq} s{s['idx']} {s['dir']} {s['gait']:4s} L={s['L']:3d} v_h={s['speed_h']:.4f} "
                  f"P={P} (ac {r if r is None else round(r, 2)}) step={step:.2f} h")
            st = crops(seq, s)
            lab = f"s{s['idx']} {s['dir']} {s['gait'].upper()}  v={s['speed_h']:.3f} h/fr  L={s['L']}  P={P}"
            bar = np.zeros((22, st.shape[1], 3), np.uint8)
            cv2.putText(bar, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            strips.append(np.vstack([bar, st]))
        W = max(x.shape[1] for x in strips)
        cv2.imwrite(f'{OUT}/{seq}.jpg', np.vstack([np.pad(x, ((0, 4), (0, W - x.shape[1]), (0, 0))) for x in strips]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
    for g in ('walk', 'run'):
        P = np.array([x[6] for x in rows if x[3] == g and x[6]], float)
        S = np.array([x[8] for x in rows if x[3] == g and x[6]], float)
        print(f'{g}: n={len(P)}  P {P.min():.0f}-{P.max():.0f} (median {np.median(P):.0f}) frames, '
              f'step {S.min():.2f}-{S.max():.2f} (median {np.median(S):.2f}) body heights')


if __name__ == '__main__':
    main()
