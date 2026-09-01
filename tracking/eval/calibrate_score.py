"""Fit and audit the score -> P(person present) calibration, and plot whether it is honest.

    python -m tracking.eval.calibrate_score            # fit, audit, write the json + plot
    python -m tracking.eval.calibrate_score --no-save  # audit only

Platt scaling (Platt 1999): P = sigma(a * log1p(score) + b), two parameters, fitted by maximum
likelihood against the hand-labelled presence intervals.

A CALIBRATION MUST BE AUDITED OUT OF SAMPLE OR IT IS SELF-CONGRATULATION. Fitted and evaluated on
the same clip, the reliability curve is guaranteed to look good; it says nothing about whether
"P=0.8" means 80% on footage the fit has not seen. So this reports leave-one-clip-out numbers
alongside the in-sample ones, and the plot draws both. Three clips is a small basis and the honest
reading of any gap between the two is "this is what three clips supports", not a defect of Platt
scaling.

WHAT CALIBRATION DOES AND DOES NOT BUY. A logistic in a MONOTONE transform of the score cannot
reorder anything, so AUC is identical before and after by construction - printed as a check, not as
a result. Calibration buys interpretability of the threshold and of the number on screen. It does
not make the tracker better at telling a person from a plant, and reporting an AUC change here
would be a bug, not a finding.
"""
import argparse

import numpy as np
from scipy.optimize import minimize

from tracking.core.calibration import (DEFAULT_PATH, YOLO_PATH, Calibration, confidence, save,
                                       score_for_confidence)
from tracking.eval.presence_learning import LABELS, auc, features_for


def yolo_confidences(clip, weights, conf_floor=0.01, scale=0.5, device='cuda'):
    """Per-emission max person-confidence from YOLO, aligned to the tracker's emitted frames.

    Aligned deliberately: calibrating YOLO on ALL frames while the tracker is calibrated only on the
    frames it emits would compare the two on different denominators, and the tracker's warm-up
    suppression alone would shift the base rate. Same frames, same labels, same fit.

    conf_floor well below the runtime threshold because a calibration needs the LOW end of the
    range: fitting only on conf>=0.25 detections would leave the fit blind to exactly the region
    where 'probably nothing' lives, which is most of this footage.
    """
    import os as _os
    _os.environ.setdefault('YOLO_VERBOSE', 'False')
    from ultralytics import YOLO

    from tracking.eval.yolo_vs_tracker import yolo_boxes
    from tracking.stream.stream import (StreamPipeline, bootstrap_person_height,
                                        frames_from_source)
    frames = list(frames_from_source(f'data/{clip}.mkv', scale))
    h = bootstrap_person_height(np.stack(frames[:240]))
    model = YOLO(weights, task='detect' if weights.endswith('.engine') else None)
    pipe = StreamPipeline(h)
    present, out, lab = LABELS[clip], [], []
    pending = {}
    for i, f in enumerate(frames):
        dets = yolo_boxes(model, f, conf_floor, device if not weights.endswith('.engine') else None)
        pending[i] = max((c for _, c in dets), default=0.0)
        r = pipe.step(f)
        if r is None:
            continue
        out.append(pending.pop(r.frame_index, 0.0))
        lab.append(int(any(a <= r.frame_index <= b for a, b in present)))
    return np.array(out, float), np.array(lab, int)


def _lk(x, transform):
    from tracking.core.calibration import _link
    return _link(x, transform)


def fit_platt(u, y, l2: float = 1e-3, transform: str = 'log1p'):
    """Maximum-likelihood a, b for sigma(a*u + b). A whisper of L2 keeps a from running away if a
    clip happens to be perfectly separable, which would push the slope to infinity and turn every
    probability into 0 or 1 - confident nonsense being the classic failure of an uncalibrated fit."""
    def nll(w):
        z = np.clip(w[0] * u + w[1], -30, 30)
        return np.mean(np.log1p(np.exp(z)) - y * z) + l2 * w[0] ** 2

    w = minimize(nll, np.array([1.0, -1.0]), method='L-BFGS-B').x
    return Calibration(a=float(w[0]), b=float(w[1]), transform=transform)


def ece(p, y, bins=10):
    """Expected calibration error: mean |predicted - observed| over equal-width probability bins,
    weighted by bin population (Guo et al. 2017). 0 is perfect; it is the number that says whether
    'P=0.8' means 80%."""
    edges = np.linspace(0, 1, bins + 1)
    total, out = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
            out.append((0.5 * (lo + hi), float(y[m].mean()), int(m.sum())))
    return total / max(len(p), 1), out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--clips', nargs='+', default=sorted(LABELS))
    p.add_argument('--method', choices=('tracker', 'yolo'), default='tracker',
                   help="'yolo' recalibrates the detector's own confidence against the same labels "
                        'and the same emitted frames, so the two panels of the live split screen '
                        'show comparable numbers rather than numbers that merely look comparable')
    p.add_argument('--weights', default='data/yolo11m.pt')
    p.add_argument('--min-raw', type=float, default=0.01,
                   help="floor for --method yolo: conf=0 (no detection at all) is clamped up to "
                        "this before the logit, because logit(0) is -20.7 and one artificial "
                        "extreme dominates a two-parameter fit. Both panels answer the FRAME-level "
                        "question - is a person present - so no-detection frames stay in the fit "
                        "as its low end rather than being dropped. Conditioning on a detection "
                        "instead gives a useless-but-true P=0.99 for every box, because on this "
                        "footage YOLO's detections are 233/233 correct and its problem is recall.")
    p.add_argument('--no-save', dest='save', action='store_false')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    if a.out is None:
        a.out = f'images/stream/{"score" if a.method == "tracker" else "yolo"}_calibration.png'

    link = 'log1p' if a.method == 'tracker' else 'logit'
    per = {}
    for c in a.clips:
        if a.method == 'tracker':
            X, yy, _ = features_for(c)
            raw = np.expm1(X[:, 0])                # features_for stores log1p(score); undo it
        else:
            raw, yy = yolo_confidences(c, a.weights)
            fired = raw >= a.min_raw
            print(f'    {c}: YOLO fired on {fired.sum()}/{len(raw)} emissions '
                  f'({100 * fired.mean():.0f}%), and when it fired a person was present '
                  f'{yy[fired].sum()}/{fired.sum()} times')
            raw = np.maximum(raw, a.min_raw)
        per[c] = (np.array([_lk(v, link) for v in raw]), yy, raw)
        print(f'  {c}: {len(yy)} emissions, {yy.sum()} person-present '
              f'({100 * yy.mean():.0f}%), raw AUC {auc(per[c][0], yy):.3f}')

    u = np.concatenate([per[c][0] for c in a.clips])
    y = np.concatenate([per[c][1] for c in a.clips])
    raw_all = np.concatenate([per[c][2] for c in a.clips])
    cal = fit_platt(u, y, transform=link)
    p_in = np.array([confidence(v, cal) for v in raw_all])

    # leave-one-clip-out: the only honest read on whether P=0.8 means 80% on unseen footage
    p_out = np.zeros_like(p_in)
    at = 0
    for c in a.clips:
        uc, yc, rawc = per[c]
        tr = [d for d in a.clips if d != c]
        cal_c = fit_platt(np.concatenate([per[d][0] for d in tr]),
                          np.concatenate([per[d][1] for d in tr]), transform=link)
        p_out[at:at + len(uc)] = [confidence(v, cal_c) for v in rawc]
        at += len(uc)

    print(f'\nfit over {len(y)} rows from {len(a.clips)} clips: '
          f'P(present) = sigma({cal.a:.3f} * {link}(x) + {cal.b:.3f})')
    print(f'  AUC   raw score {auc(u, y):.3f}   calibrated {auc(p_in, y):.3f}   '
          f'(identical by construction - a monotone map cannot reorder)')
    e_in, curve_in = ece(p_in, y)
    e_out, curve_out = ece(p_out, y)
    print(f'  ECE   in-sample {e_in:.3f}   leave-one-clip-out {e_out:.3f}')
    print(f'  Brier in-sample {np.mean((p_in - y) ** 2):.3f}   '
          f'leave-one-clip-out {np.mean((p_out - y) ** 2):.3f}')

    raws = (0.0, 1.0, 5.0, 20.0, 100.0) if link == 'log1p' else (0.05, 0.25, 0.5, 0.75, 0.95)
    unit = 'score' if link == 'log1p' else 'raw conf'
    print(f'\n  what the raw {unit} values mean, now that they are calibrated:')
    for v in raws:
        print(f'    {unit} {v:6.2f}  ==  P(present) {confidence(v, cal):.2f}')
    for q in (0.25, 0.5, 0.75, 0.9):
        print(f'    P(present) {q:.2f}  ==  {unit} {score_for_confidence(q, cal):.2f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
    for curve, lab, mk in ((curve_in, f'in-sample (ECE {e_in:.3f})', 'o-'),
                           (curve_out, f'leave-one-clip-out (ECE {e_out:.3f})', 's--')):
        ax[0].plot([c[0] for c in curve], [c[1] for c in curve], mk, label=lab)
    ax[0].set_xlabel('predicted P(person present)')
    ax[0].set_ylabel('observed fraction present')
    ax[0].set_title(f'Reliability of the calibrated {a.method}')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    # x range 0-30, not 0-400: the logistic saturates by score ~20, so a wide axis is 95% empty
    # white space. The histograms behind the curve show where the scores ACTUALLY fall, which is
    # what makes the gate position meaningful rather than decorative.
    scores = raw_all
    top = 30 if link == 'log1p' else 1.0
    for m, lab, col in ((y == 1, 'person present', 'tab:green'),
                        (y == 0, 'absent', 'tab:red')):
        ax[1].hist(np.clip(scores[m], 0, top), bins=40, range=(0, top), density=True,
                   alpha=.35, color=col, label=f'{lab} (n={int(m.sum())})')
    ax[1].set_ylabel('density of observed scores')
    ax[1].legend(fontsize=8, loc='upper center')
    twin = ax[1].twinx()
    ss = np.linspace(0, top, 600)
    twin.plot(ss, [confidence(s, cal) for s in ss], 'tab:blue', lw=2)
    twin.set_ylabel('P(person present)', color='tab:blue')
    twin.set_ylim(0, 1.02)
    for q in (0.25, 0.5, 0.75, 0.9):
        s_q = score_for_confidence(q, cal)
        twin.plot([s_q, s_q], [0, q], ':', color='k', lw=1)
        twin.annotate(f'P={q:g} @ {s_q:.1f}', (s_q, q), fontsize=7, xytext=(4, -4),
                      textcoords='offset points')
    ax[1].set_xlabel('track score  (span * net_disp / (1 + resid_std))' if link == 'log1p'
                     else f"YOLO's own reported confidence (uncalibrated)")
    ax[1].set_title(f'{a.method}: the gate, in units anyone can read')
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f'\nwrote {a.out}')

    if a.save:
        cal = Calibration(a=cal.a, b=cal.b, transform=cal.transform, meta={
            'method': a.method, 'clips': list(a.clips), 'n': int(len(y)),
            'min_raw': a.min_raw if a.method == 'yolo' else None, 'auc': round(float(auc(u, y)), 4),
            'ece_in_sample': round(float(e_in), 4), 'ece_leave_one_clip_out': round(float(e_out), 4),
            'form': f'P(present) = sigmoid(a * {link}(x) + b)',
            'weights': a.weights if a.method == 'yolo' else None})
        path = DEFAULT_PATH if a.method == 'tracker' else YOLO_PATH
        save(cal, path)
        print(f'saved {a.method} calibration ({link} link): a={cal.a:.4f} b={cal.b:.4f} -> {path}')


if __name__ == '__main__':
    main()
