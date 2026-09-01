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

from tracking.core.calibration import Calibration, confidence, save, score_for_confidence
from tracking.eval.presence_learning import LABELS, auc, features_for


def fit_platt(u, y, l2: float = 1e-3):
    """Maximum-likelihood a, b for sigma(a*u + b). A whisper of L2 keeps a from running away if a
    clip happens to be perfectly separable, which would push the slope to infinity and turn every
    probability into 0 or 1 - confident nonsense being the classic failure of an uncalibrated fit."""
    def nll(w):
        z = np.clip(w[0] * u + w[1], -30, 30)
        return np.mean(np.log1p(np.exp(z)) - y * z) + l2 * w[0] ** 2

    w = minimize(nll, np.array([1.0, -1.0]), method='L-BFGS-B').x
    return Calibration(a=float(w[0]), b=float(w[1]))


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
    p.add_argument('--no-save', dest='save', action='store_false')
    p.add_argument('--out', default='images/stream/score_calibration.png')
    a = p.parse_args()

    per = {}
    for c in a.clips:
        X, y, _ = features_for(c)
        per[c] = (X[:, 0], y)                      # column 0 is already log1p(score)
        print(f'  {c}: {len(y)} emissions, {y.sum()} person-present '
              f'({100 * y.mean():.0f}%), score AUC {auc(X[:, 0], y):.3f}')

    u = np.concatenate([per[c][0] for c in a.clips])
    y = np.concatenate([per[c][1] for c in a.clips])
    cal = fit_platt(u, y)
    p_in = np.array([confidence(np.expm1(v), cal) for v in u])

    # leave-one-clip-out: the only honest read on whether P=0.8 means 80% on unseen footage
    p_out = np.zeros_like(p_in)
    at = 0
    for c in a.clips:
        uc, yc = per[c]
        tr = [d for d in a.clips if d != c]
        cal_c = fit_platt(np.concatenate([per[d][0] for d in tr]),
                          np.concatenate([per[d][1] for d in tr]))
        p_out[at:at + len(uc)] = [confidence(np.expm1(v), cal_c) for v in uc]
        at += len(uc)

    print(f'\nfit over {len(y)} emissions from {len(a.clips)} clips: '
          f'P(present) = sigma({cal.a:.3f} * log1p(score) + {cal.b:.3f})')
    print(f'  AUC   raw score {auc(u, y):.3f}   calibrated {auc(p_in, y):.3f}   '
          f'(identical by construction - a monotone map cannot reorder)')
    e_in, curve_in = ece(p_in, y)
    e_out, curve_out = ece(p_out, y)
    print(f'  ECE   in-sample {e_in:.3f}   leave-one-clip-out {e_out:.3f}')
    print(f'  Brier in-sample {np.mean((p_in - y) ** 2):.3f}   '
          f'leave-one-clip-out {np.mean((p_out - y) ** 2):.3f}')

    print('\n  what the existing thresholds mean, now that they have units:')
    for s in (0.0, 1.0, 5.0, 20.0, 100.0):
        print(f'    --min-score {s:5.0f}  ==  P(present) >= {confidence(s, cal):.2f}')
    for q in (0.25, 0.5, 0.75, 0.9):
        print(f'    --min-confidence {q:.2f}  ==  --min-score {score_for_confidence(q, cal):.1f}')

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
    ax[0].set_title('Reliability of the calibrated score')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    # x range 0-30, not 0-400: the logistic saturates by score ~20, so a wide axis is 95% empty
    # white space. The histograms behind the curve show where the scores ACTUALLY fall, which is
    # what makes the gate position meaningful rather than decorative.
    scores = np.expm1(u)
    top = 30
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
    ax[1].set_xlabel('track score  (span * net_disp / (1 + resid_std))')
    ax[1].set_title('The gate, in units anyone can read')
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f'\nwrote {a.out}')

    if a.save:
        cal = Calibration(a=cal.a, b=cal.b, meta={
            'clips': list(a.clips), 'n': int(len(y)), 'auc': round(float(auc(u, y)), 4),
            'ece_in_sample': round(float(e_in), 4), 'ece_leave_one_clip_out': round(float(e_out), 4),
            'form': 'P(present) = sigmoid(a * log1p(score) + b)'})
        save(cal)
        print(f'saved calibration: a={cal.a:.4f} b={cal.b:.4f}')


if __name__ == '__main__':
    main()
