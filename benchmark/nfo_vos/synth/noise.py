"""Input nuisance model for the feature-layer training plan (docs/layer_sam2/training/plan_feature_layers_training.md §2).

Applied to the input frames only, never to the targets (clean frame, S, V). Per clip:
  sway     scene B and occluder matte a translated by d_t = A·sin(2π t / P + φ)·(cos θ, sin θ), A ∈ U[0, A_max] px
           (passed to gpjatk_foliage.make_layered_clip(sway=...); geometry, so V follows the swayed matte)
  gain     X_t ← X_t · (1 + g_t),  g_t ~ U[−0.01, 0.01] per frame
  noise    X_t ← X_t + N(0, σ²),   σ ~ U[σ_lo, σ_hi] per clip
  JPEG     quality ~ U[70, 95] per clip
Calibration (`calibrate`): NFO warm-up residual r = f_t − median_t(f). Small-sway model f_t(x) ≈ B(x) + u_t·∇B(x) + n_t
gives Var_t r(x) = σ_n² + σ_u²·|∇B(x)|² (u isotropic with per-axis variance σ_u², n white), so a regression of the
per-pixel temporal variance on |∇B|² separates sensor noise (intercept) from sway (slope). Lag-1 temporal correlation
of r is a second discriminant: ≈ 0 for white sensor noise, large for slow sway.
"""
import json
import os

import cv2
import numpy as np

SIGMA = (1.0, 5.0)       # white noise; calibrate(): flat-region σ 0.3–4.8 on NFO, lag-1 corr 0.83–0.91 ⇒ the plan's
                         # U[3, 12] (total residual std) is mostly temporally correlated scene change, not sensor noise
GAIN = 0.01
SWAY_MAX = 1.0           # px amplitude
SWAY_PERIOD = (8, 40)    # frames
JPEG_Q = (70, 95)
FLUTTER_RHO = 0.97       # AR(1) coefficient. NFO residual lag-1 is 0.83–0.91 (seq2 0.868), but the textured photo responds
                         # nonlinearly to ~1–2 px displacements: ρ 0.87 gives residual lag-1 0.65; it saturates ≈ 0.79
                         # (calibrate_flutter sweep, 2026-10-15). 0.97 is the closest at seq2's std.
FLUTTER_SCALE = 12.0     # px, spatial σ of the displacement field (assumed: ≈ half a clump-24 leaf clump; not calibrated)
FLUTTER_AMP = 2.1        # px RMS per axis; calibrate_flutter at ρ 0.97: 1.9 → std_all 10.7, interpolated to seq2's 11.96
FLUTTER_DOWN = 4         # fields are generated at 1/4 resolution and bilinearly upsampled (smooth at σ = 12 px)


def sample(rng, sigma=SIGMA, sway_max=SWAY_MAX):
    return dict(sigma=float(rng.uniform(*sigma)), sway_amp=float(rng.uniform(0, sway_max)),
                sway_period=float(rng.uniform(*SWAY_PERIOD)), sway_phase=float(rng.uniform(0, 2 * np.pi)),
                sway_dir=float(rng.uniform(0, np.pi)), jpeg=int(rng.integers(JPEG_Q[0], JPEG_Q[1] + 1)),
                seed=int(rng.integers(2 ** 31)))


def sway_offsets(p, n, t0=0):
    """(n, 2) per-frame (dx, dy) in px for frames t0 … t0 + n − 1 (negative t0 = warm-up before the clip)."""
    t = np.arange(t0, t0 + n)
    a = p['sway_amp'] * np.sin(2 * np.pi * t / p['sway_period'] + p['sway_phase'])
    return np.stack([a * np.cos(p['sway_dir']), a * np.sin(p['sway_dir'])], 1)


def flutter_fields(p, n, amp, shape=(540, 960)):
    """Non-rigid foliage flutter: (n, h/4, w/4, 2) displacement fields in native px. Each field is spatially smoothed
    white noise (Gaussian σ = FLUTTER_SCALE) normalised to unit std; in time an AR(1) process
    Z_t = ρ Z_{t−1} + √(1 − ρ²) N_t, so every frame has per-axis RMS displacement `amp` and lag-1 corr ρ."""
    rng = np.random.default_rng(p['seed'] + 7)
    h, w, s = shape[0] // FLUTTER_DOWN, shape[1] // FLUTTER_DOWN, FLUTTER_SCALE / FLUTTER_DOWN

    def field():
        z = np.stack([cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), s) for _ in range(2)], -1)
        return z / z.std()
    Z, out = field(), []
    for _ in range(n):
        out.append(amp * Z)
        Z = FLUTTER_RHO * Z + np.sqrt(1 - FLUTTER_RHO ** 2) * field()
    return np.stack(out).astype(np.float32)


def displace(x, d):
    """d = (dx, dy): rigid translation (warpAffine, the original sway path). d = (h', w', 2) field: out(p) = x(p − d(p))
    (same sign as the translation), field upsampled to x's size."""
    Hh, Ww = x.shape
    if np.ndim(d) == 1:
        return cv2.warpAffine(x, np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (Ww, Hh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    d = cv2.resize(np.asarray(d, np.float32), (Ww, Hh), interpolation=cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.arange(Ww, dtype=np.float32), np.arange(Hh, dtype=np.float32))
    return cv2.remap(x, gx - d[..., 0], gy - d[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def photometric(X, p, salt=0):
    """Gain, Gaussian noise, clip, JPEG on float frames [T, H, W] → float32 [T, H, W] (integer-valued)."""
    rng = np.random.default_rng(p['seed'] + salt)
    out = []
    for x in X:
        y = x * (1 + rng.uniform(-GAIN, GAIN)) + rng.normal(0, p['sigma'], x.shape)
        y = np.clip(np.round(y), 0, 255).astype(np.uint8)
        y = cv2.imdecode(cv2.imencode('.jpg', y, [cv2.IMWRITE_JPEG_QUALITY, p['jpeg']])[1], 0)
        out.append(y.astype(np.float32))
    return np.stack(out)


def residual_stats(F):
    """Warm-up residual statistics of an empty-scene stack F [T, H, W] (see module docstring)."""
    med = np.median(F, 0)
    r = F - med
    spread = float(r.reshape(len(r), -1).mean(1).std())
    r = r - r.reshape(len(r), -1).mean(1)[:, None, None]                   # remove global gain
    var = (1.4826 * np.median(np.abs(r - np.median(r, 0)), 0)) ** 2       # robust per-pixel temporal variance
    gx, gy = cv2.Sobel(med, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(med, cv2.CV_32F, 0, 1) / 8
    g2 = (gx ** 2 + gy ** 2).ravel()
    q = np.quantile(g2, np.linspace(0, 1, 21))
    b = np.clip(np.searchsorted(q, g2, side='right') - 1, 0, 19)
    xs = np.array([np.median(g2[b == k]) for k in range(20)]); ys = np.array([np.median(var.ravel()[b == k]) for k in range(20)])
    slope, icpt = np.polyfit(xs, ys, 1)
    return dict(sigma_n=float(np.sqrt(max(icpt, 0))), sigma_u=float(np.sqrt(max(slope, 0))),
                sigma_flat=float(np.sqrt(np.median(var.ravel()[g2 <= q[2]]))), std_all=float(np.sqrt(np.median(var))),
                lag1=float((r[1:] * r[:-1]).sum() / (r ** 2).sum()), brightness_spread=spread,
                var_by_g2=[(float(a), float(c)) for a, c in zip(xs[::4], ys[::4])])


def calibrate(max_frames=150):
    """σ_n, σ_u, lag-1 correlation, global brightness spread per NFO sequence (longest warm-up)."""
    from benchmark.nfo_vos import trials as TR
    tr = json.load(open('results/benchmark/rr/trials.json'))
    out = {}
    for seq in ('seq1', 'seq2', 'seq3', 'seq4'):
        w0, w1 = max({tuple(t['warmup']) for t in tr if t['seq'] == seq}, key=lambda w: w[1] - w[0])
        idx = range(w0, min(w1, w0 + max_frames - 1) + 1)
        F = np.stack([TR.native(i, seq) for i in idx]).astype(np.float32)
        out[seq] = dict(frames=[idx[0], idx[-1]], **residual_stats(F))
        print(seq, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[seq].items() if k != 'var_by_g2'}, flush=True)
    return out


def calibrate_flutter(amps=(0.25, 0.5, 1.0, 1.5, 2.0), n_clips=6, n=40):
    """Synthetic warm-up stacks (B only, occluder ≡ B) with flutter at each amp, white σ 3, JPEG 85, no rigid sway:
    residual std_all / lag-1, to match NFO seq2 (std_all 11.96, lag-1 0.868; noise_calibration.json)."""
    from benchmark.nfo_vos.adapter import dataset as D
    from benchmark.nfo_vos.synth.gpjatk_foliage import foliage_pair
    res = {}
    for amp in amps:
        st = []
        for spec in D.clip_specs('val')[::20][:n_clips]:
            B = foliage_pair(np.random.default_rng(spec[3]))[0].astype(np.float32)
            p = dict(sigma=3.0, jpeg=85, seed=spec[3])
            fl = flutter_fields(p, n, amp)
            st.append(residual_stats(photometric(np.stack([displace(B, d) for d in fl]), p)))
        res[amp] = {k: float(np.mean([x[k] for x in st])) for k in ('std_all', 'lag1', 'sigma_flat')}
        print(f'amp {amp:.2f} px: ' + ', '.join(f'{k} {v:.3f}' for k, v in res[amp].items()), flush=True)
    return res


def montage(path='images/benchmark/gpjatk_foliage/noise.png', n_clips=3, crop=200):
    """Rows: val-subject clips at the middle frame, crop around the person. Columns: clean | noisy at the low end of
    the σ / sway range | noisy at the high end | plan's original σ = 12 (no sway) | |high − clean| × 4. Last row: NFO warm-up frame crops for reference."""
    from benchmark.nfo_vos.synth.gpjatk_foliage import make_layered_clip
    from benchmark.nfo_vos import trials as TR
    lo = dict(sigma=SIGMA[0], sway_amp=0.0, sway_period=20., sway_phase=0., sway_dir=0., jpeg=JPEG_Q[1], seed=1)
    hi = dict(sigma=SIGMA[1], sway_amp=SWAY_MAX, sway_period=8., sway_phase=0., sway_dir=0.7, jpeg=JPEG_Q[0], seed=2)
    p12 = dict(lo, sigma=12.0, jpeg=JPEG_Q[0], seed=3)
    rows = []
    for k, (seq, cam) in enumerate([('p21s1', 'c1'), ('p23s1', 'c2'), ('p25s1', 'c3')][:n_clips]):
        t = 21                                     # sin(2π·21/8) ≈ 0.71 → |d| ≈ 0.7 px in the high draw
        res = {}
        for name, p in (('lo', lo), ('hi', hi), ('s12', p12)):
            *_, L = make_layered_clip(seq, cam, 0.45, 500 + k, 'perlin', None, save=False, layers=True,
                                      sway=sway_offsets(p, 40))
            res[name] = photometric(L['X'][t:t + 1], p)[0]
        clean = L['clean'][t]
        ys, xs = np.nonzero(L['S'][t]); cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = np.clip(cy - crop // 2, 0, 540 - crop), np.clip(cx - crop // 2, 0, 960 - crop)
        c = lambda im: np.clip(im[y0:y0 + crop, x0:x0 + crop], 0, 255).astype(np.uint8)
        tiles = [c(clean), c(res['lo']), c(res['hi']), c(res['s12']), c(4 * np.abs(res['hi'] - clean))]
        for tl, lab in zip(tiles, ('clean', f'lo s{SIGMA[0]:g} q{JPEG_Q[1]}', f'hi s{SIGMA[1]:g} q{JPEG_Q[0]} sway1', 'plan s12 q70',
                                   '|hi-clean|x4')):
            cv2.putText(tl, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        rows.append(np.concatenate(tiles, 1))
    tr = json.load(open('results/benchmark/rr/trials.json'))
    tiles = []
    for seq in ('seq1', 'seq2'):
        w0, w1 = max({tuple(t['warmup']) for t in tr if t['seq'] == seq}, key=lambda w: w[1] - w[0])
        F = np.stack([TR.native(i, seq) for i in range(w0, min(w1, w0 + 60))]).astype(np.float32)
        f, med = F[len(F) // 2], np.median(F, 0)
        y0, x0 = 300 - crop // 2, 400 - crop // 2
        a, b = f[y0:y0 + crop, x0:x0 + crop].astype(np.uint8), np.clip(4 * np.abs(f - med)[y0:y0 + crop, x0:x0 + crop], 0, 255).astype(np.uint8)
        cv2.putText(a, f'NFO {seq}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        cv2.putText(b, f'|{seq}-median|x4', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        tiles += [a, b]
    rows.append(np.concatenate(tiles + [np.zeros_like(a)], 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.concatenate(rows, 0))
    print('wrote', path)


if __name__ == '__main__':
    import sys
    if 'flutter' in sys.argv:
        calibrate_flutter()
    elif 'calibrate' in sys.argv:
        json.dump(calibrate(), open('results/benchmark/rr/noise_calibration.json', 'w'), indent=1)
    else:
        montage()
