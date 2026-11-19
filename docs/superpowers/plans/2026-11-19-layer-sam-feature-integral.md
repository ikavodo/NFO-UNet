# Learned Feature-Space Integral Image: Implementation Plan (validity check)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide cheaply whether a learned integral image built from cached SAM2 features has merit. It should match
the pixel adapter on seq1 with one encoder pass per frame and ≤ 40 ms/frame.

**Architecture:**
- A small aggregator (`FeatIntegral`, logits at stride 16) takes a learned per-cell softmax over the encoder features of
  frames t−n_h..t−1, shifted onto frame t by the tracker's Theil–Sen x-shift (bilinear), plus a null key.
- It outputs Ĉ16/8/4 and a := 1 − α_∅. These replace `enc(C)` and the pixel mass as inputs of the unchanged `Fusion`
  adapter.
- Stage 0 pre-trains the aggregator alone on a clean-feature loss. Then Fusion is trained as today, with the aggregator
  fine-tuned jointly.

**Tech Stack:** PyTorch, frozen SAM2.1-b+ image predictor, existing `benchmark/nfo_vos` synthetic pipeline, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-layer-sam-feature-integral-design.md` (revision 3).

## Scope: the merit test only (user, 2026-11-19: "not a billion arms")

The plan runs exactly three checks, in order. Each can stop the direction.

| check | what | cost | stop if |
|---|---|---|---|
| **G0** | shift equivariance of SAM2 features under subject motion | minutes, laptop | p90 ε16 ≥ ½ median δ16 for both roll and bilinear |
| **G1** | stage 0: learned Ĉ vs fixed Ĉ vs enc(C) vs F_t on hidden-recoverable cells | ≈ 15 min, laptop | learned ≤ F_t at stride 16 |
| **G2** | one Fusion training, `feat` seed 0, vs the existing pixel checkpoint seed 0, seq1 only, one environment; plus timing | 1 cluster job | seq1 hand J or box AUC < pixel − 0.02; or > 40 ms/frame |

**Deferred until G2 passes. Do not build:**
- 3 seeds and the seq2–4 headline;
- the `feat-fixed` training arm (it becomes the main arm only if G1 shows fixed ≥ learned);
- the occluder mix arms;
- the background track (B-med / B-rec, cold start);
- the switch refit.

**Background in this plan:** B = `enc(pixel median of the warm-up)`, exactly as today. This holds the B input identical
to the pixel adapter, so G2 isolates the integral image. It deviates from the spec's §5 (per-cell median of encoded
warm-up frames), which moves to the deferred background track.

## Global Constraints

- seq2–4 is never touched: no selection or tuning on it, and no seq2–4 evaluation in this plan.
- The pixel (default) path stays bit-identical: `parity_snapshot save before.pkl` → change → `save after.pkl` → `diff`
  prints IDENTICAL. Run at the end of Tasks 4, 6 and 7.
- Every cosine involving stride-16 features is taken on features centred by subtracting `sam.no_mem_embed`; strides 8
  and 4 are not centred. Helper: `feat_integral.ccos`.
- Logits and the softmax are computed in float32 even under autocast (fp16 on pre-Ampere cluster GPUs).
- Shifts are in 1024-frame px: dx1024 = dx_native · 1024 / W; cells = dx1024 / s.
- Arms are compared only within one environment (laptop or cluster), all scored in one run.
- Outputs:
  - numbers → `results/benchmark/rr/feat_integral/`;
  - images → `images/benchmark/feat_integral/` (dpi ≈ 80, ~10 in wide);
  - one line per dead end → `docs/vos_benchmark/nfo_failure_log.md`.
- All commands: from the repo root, `PYTHONPATH=.`, `$PY` = `../samurai/.venv/bin/python`.

## Review Focus

1. **Person at the image border:** all shifted keys are invalid at some cells. Expected: α_∅ = 1 there, Ĉ = B, a = 0,
   no NaN. Test in Task 2.
2. **Mode-2 frames** (5%, "integral missing"): every key invalid. Expected: Ĉ = B, a = 0. Test in Task 4.
3. **Clips with no hidden-recoverable cells** (ρ = 0 in 10% of clips; early frames): Ω is empty. Expected: L_C = 0, not
   NaN. Test in Task 3.
4. **fp16 autocast:** expected finite α and Ĉ. Test in Task 2.
5. **Inference frames with < n_h history:** tracker prompts, as today. Expected: identical masks to the pixel path for
   j < 6. Test in Task 7.

---

### Task 1: Noise-matched, occluder-free target frame X̃

**Files:**
- Modify: `benchmark/nfo_vos/synth/gpjatk_foliage.py:259-262` (the `layers` dict of `make_layered_clip`)
- Modify: `benchmark/nfo_vos/adapter/dataset.py:69-71` (`make_clip` return)
- Modify: `benchmark/nfo_vos/adapter/fusion_data.py` (`rescale_clip` places `Xt`; `clip_raw` carries `Xt`)
- Test: `benchmark/nfo_vos/tests/test_feat_integral.py` (new)

**Interfaces:**
- Produces:
  - `make_layered_clip(..., layers=True)[-1]['Xt']`: float [T, H, W], defined as `mf * I + (1 - mf) * Bs`. Same sway
    as X, no occluder.
  - `dataset.make_clip(...)['Xt'] = noise.photometric(L['Xt'], p)`: same `p`, salt 0, so the same gain/noise
    realisation as X.
  - `clip_raw(spec, rng)['Xt']`: uint8 [T, H, W].

- [ ] **Step 1: Write the failing tests**

```python
def test_target_equals_frame_where_unoccluded():
    from benchmark.nfo_vos.synth.gpjatk_foliage import make_layered_clip
    from benchmark.nfo_vos.adapter import dataset as D
    seq, cam, *_ = D.clip_specs('val')[0]
    *_, L = make_layered_clip(seq, cam, 0.45, 3, 'perlin', None, n=6, save=False, layers=True)
    free = L['a'] < 1e-6                       # static occluder alpha (no sway passed)
    assert np.allclose(L['Xt'][:, free], L['X'][:, free])
    assert np.abs(L['Xt'] - L['X'])[:, ~free].max() > 0

def test_make_clip_target_shares_noise():
    from benchmark.nfo_vos.adapter import dataset as D
    seq, cam, _, seed = D.clip_specs('val')[0]
    d = D.make_clip(seq, cam, 0.45, seed)
    occ_free = cv2.dilate((d['a'] >= 1e-6).astype(np.uint8), np.ones((33, 33), np.uint8)) == 0   # ≥ 16 px from occluder
    assert d['Xt'].shape == d['X'].shape
    assert np.median(np.abs(d['Xt'].astype(float) - d['X'].astype(float))[:, occ_free]) <= 1.0

def test_clip_raw_carries_target():
    from benchmark.nfo_vos.adapter.fusion_data import clip_raw, specs
    r = clip_raw(specs('val')[0], np.random.default_rng(0))
    assert r['Xt'].shape == r['X'].shape and r['Xt'].dtype == np.uint8
```

The second test's mask: pixels at least 16 px from any occluder pixel, because JPEG couples 8×8 blocks (and sway moves the matte by a few px). `make_clip` exposes `d['a']` for it.
couples 8×8 blocks. Expose `a` from `make_clip` as `d['a']` to build it.

- [ ] **Step 2: Run:** `$PY -m pytest benchmark/nfo_vos/tests/test_feat_integral.py -q -k "target or carries"`.
  Expected: FAIL (`KeyError: 'Xt'`).
- [ ] **Step 3: Implement:**
  - add `Xt` to the `layers` dict;
  - `make_clip` returns `Xt=noise.photometric(L['Xt'], p)` and `a=L['a']`;
  - `rescale_clip` places `Xt` like `X`;
  - `clip_raw` returns `Xt=u8(d['Xt'])`.

  No other RNG draw may be added (the default clips stay bit-identical).
- [ ] **Step 4: Run** the same command. Expected: PASS. Then the full suite: `$PY -m pytest benchmark/nfo_vos/tests -q`,
  no new failures.
- [ ] **Step 5: Commit:** `git commit -m "Feature integral: noise-matched occluder-free target frame Xt in the synthetic clips"`

---

### Task 2: `feat_integral.py`: key shift, aggregator, loss helpers

**Files:**
- Create: `benchmark/nfo_vos/adapter/feat_integral.py`
- Test: `benchmark/nfo_vos/tests/test_feat_integral.py`

**Interfaces:**
- Produces:
  - `ccos(x, y, mu=None) -> Tensor`: cosine along dim −3 of (x − mu), (y − mu). mu broadcastable [1, C, 1, 1].
  - `shift_keys(F, dx1024, stride, mode='bilinear') -> (Fs, valid)`.
    - Inputs: F [n, k, C, h, w] (frames already gathered), dx1024 [n, k] float.
    - Fs: each key translated by +dx1024/stride cells in x (content moves right for positive dx), zero-filled outside.
    - valid: [n, k, 1, h, w] bool, True where the source cell lies inside the image.
    - `mode='roll'` rounds to whole cells.
    - Implement with `F.grid_sample(align_corners=False, padding_mode='zeros')` on the flattened n·k batch.
  - `class FeatIntegral(nn.Module)`: `__init__(self, mu16: Tensor, d=32, nh_max=8)`;
    `forward(self, K, valid16, Ft, B, age, fixed=False, tau=3.5) -> dict`.
    - Inputs:
      - K: {'16','8','4'} → [n, k, C_s, h_s, w_s] (shifted);
      - valid16: [n, k, 1, 64, 64];
      - Ft, B: {'16','8','4'} → [n, C_s, h_s, w_s];
      - age: [n, k] long in 1..nh_max.
    - Returns: `{'C16','C8','C4'}` [n, C_s, h_s, w_s], `'a'` [n, 256, 256] in [0, 1], `'alpha16'` [n, k+1, 64, 64]
      (last channel = ∅).
  - `feat_loss(C, T, Om, mu16) -> Tensor`: mean over strides of the mean over Ω cells of 1 − ccos(C^s, T^s).
    - Om: {s: [n, 1, h_s, w_s] bool}.
    - Returns 0 (with grad) when every Ω is empty.

**Learned logits** (stride 16):
- `P = Conv1x1(256, d)`; `e_age = Embedding(nh_max + 1, d)`;
  `h = Conv1x1(4d, d) → GELU → Conv3x3(d, d) → GELU → Conv1x1(d, 1)`, the last conv zero-initialised.
- ℓ_k = h([P F_k, P(F_k − F_t), P(F_k − B), e_age]).
- ℓ_∅ = learned scalar `self.null`, init 0. Invalid keys get −inf.
- α = softmax over the k + 1 channels, in float32.
- α^8, α^4 = bilinear upsampling of α^16 to 128² / 256².
- Ĉ^s = Σ_k α_k^s K_k^s + α_∅^s B^s; a = 1 − α_∅^4.

**Fixed logits** (`fixed=True`, the spec's Ĉ_fixed; no parameters used):
- w_k = e^{−age_k/τ} / Σ e^{−age/τ};
- G_k = σ(z_k − θ_k), where:
  - z_k = (m_k − median(m_k)) / (1.4826 · MAD(m_k)) per key over cells;
  - m_k = 1 − ccos(K_k^16, B^16, mu16);
  - θ_k = max(3, q90(z_k)).
- ℓ_k = log(w_k G_k) (−inf where invalid); ℓ_∅ = log(max(1 − Σ_k w_k G_k, 1e-6)).
- Hence α_k = w_k G_k and a = Σ w G: today's mass, in feature space.

- [ ] **Step 1: Write the failing tests** (CPU; small sizes: n = 2, k = 3, h16 = 8 means h8 = 16, h4 = 32; for these
  tests let `a` be returned at the stride-4 grid size):

```python
def test_shift_integer_equals_roll_interior():
    F = torch.randn(1, 1, 4, 8, 8)
    for mode in ('bilinear', 'roll'):
        Fs, v = shift_keys(F, torch.tensor([[32.]]), 16, mode)        # 2 cells right
        assert torch.allclose(Fs[..., 2:], F[..., :-2], atol=1e-5) and not v[..., :2].any() and v[..., 2:].all()

def test_shift_zero_is_identity():
    F = torch.randn(2, 3, 4, 8, 8)
    Fs, v = shift_keys(F, torch.zeros(2, 3), 16)
    assert torch.allclose(Fs, F, atol=1e-6) and v.all()

def test_weights_sum_to_one_and_init_uniform():
    m = FeatIntegral(torch.zeros(1, 256, 1, 1)); o = m(*_inputs())
    assert torch.allclose(o['alpha16'].sum(1), torch.ones(1), atol=1e-5)
    assert torch.allclose(o['alpha16'], torch.full_like(o['alpha16'], 1 / 4), atol=1e-6)   # k = 3 valid + ∅

def test_all_keys_invalid_gives_background():                          # Review Focus 1 and 2
    K, v, Ft, B, age = _inputs(); v = torch.zeros_like(v)
    o = FeatIntegral(torch.zeros(1, 256, 1, 1))(K, v, Ft, B, age)
    assert torch.allclose(o['C16'], B['16']) and o['a'].abs().max() < 1e-6 and torch.isfinite(o['C4']).all()

def test_fixed_mode_equals_recency_mass_when_gate_open():
    # make every key differ strongly from B so G ≈ 1: a ≈ Σ w = 1, Ĉ ≈ Σ w_k K_k
    ...  # implementer: assert a > 0.99 and C16 ≈ recency-weighted mean of K16 (atol 1e-3)

def test_fp16_autocast_finite():                                      # Review Focus 4 (skip without CUDA)
    ...  # forward under torch.autocast('cuda', torch.float16) on CUDA tensors; all outputs finite
```

`_inputs()` builds random K/Ft/B dicts at the three strides with channels 256/64/32, all-valid masks and age = [3, 2, 1].

- [ ] **Step 2: Run** `$PY -m pytest benchmark/nfo_vos/tests/test_feat_integral.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `feat_integral.py` as specified above.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit:** `git commit -m "Feature integral: bilinear key shift, learned/fixed per-cell softmax aggregator with null key, clean-feature loss"`

---

### Task 3: Feature path in `fusion_data.build`

**Files:**
- Modify: `benchmark/nfo_vos/adapter/fusion_data.py` (`build`)
- Test: `benchmark/nfo_vos/tests/test_feat_integral.py`

**Interfaces:**
- Consumes: `shift_keys` (Task 2), `clip_raw(...)['Xt']` (Task 1), `hist_warps` (existing).
- Produces: `build(pred, raw, ..., feat=False, with_pixel=False) -> (f, meta)`.
  - With `feat=True`:
    - the pixel integral and `enc(C)` are skipped (unless `with_pixel=True`, which adds the usual `C16/8/4` and `a_pix`
      for G1);
    - `f` gains:
      - `'K16','K8','K4'`: [n, nh, C_s, h_s, w_s], from the already computed `F16/F8/F4` gathered at `hist`, shifted
        by dx1024 = (M[0, 2]) · 1024/W;
      - `'Kvalid16'` [n, nh, 1, 64, 64]: all False for mode-2 targets;
      - `'age'` [n, nh] = nh..1;
      - `'T16','T8','T4'` = encoder features of `raw['Xt'][ts]`;
      - `'Om16','Om8','Om4'`: [n, 1, h_s, w_s] bool. Ω = area coverage > 0.5 of (O_t ∧ ¬V_t ∧ S_t), where O_t = ∪
        over hist of V_i warped by M_i (`cv2.warpAffine`, nearest);
      - `'Vis16'`: coverage > 0.5 of V_t (the floor);
      - `'mu16'` = `pred.model.no_mem_embed.view(1, -1, 1, 1)`.
  - `feat=False` is unchanged (bit-identical).

- [ ] **Step 1: Write the failing test** (GPU; `pytest.mark.skipif(not torch.cuda.is_available())`):

```python
def test_build_feat_shapes_and_masks():
    from benchmark.nfo_vos.adapter.fusion_adapter import image_predictor
    from benchmark.nfo_vos.adapter.fusion_data import build, clip_raw, specs
    raw = clip_raw(specs('val')[0], np.random.default_rng(0)); raw['modes'][:] = 0; raw['modes'][-1] = 2
    f, meta = build(image_predictor(), raw, all_targets=True, feat=True)
    n, nh = len(meta['ts']), raw['occ']['nh']
    assert f['K16'].shape == (n, nh, 256, 64, 64) and f['K4'].shape == (n, nh, 32, 256, 256)
    assert 'C16' not in f and f['T8'].shape == (n, 64, 128, 128)
    assert not f['Kvalid16'][-1].any()                                 # mode 2 → all keys invalid (Review Focus 2)
    S16 = torch.as_tensor(np.stack([a256(raw['S'][t]) for t in meta['ts']]))[:, None, ::4, ::4] > 0.5
    assert not (f['Om16'].cpu() & ~S16).any()                          # Ω ⊆ person
```

- [ ] **Step 2: Run** `$PY -m pytest benchmark/nfo_vos/tests/test_feat_integral.py::test_build_feat_shapes_and_masks -q`.
  Expected: FAIL (`TypeError: unexpected keyword 'feat'`).
- [ ] **Step 3: Implement** the `feat` branch in `build`.
- [ ] **Step 4: Run** the test, then the full suite and the parity snapshot. Expected: PASS; parity prints IDENTICAL.
- [ ] **Step 5: Commit:** `git commit -m "Feature integral: build(feat=True): shifted feature keys, Xt targets, hidden-recoverable mask Ω"`

---

### Task 4: G0, shift equivariance (no training)

**Files:**
- Create: `benchmark/nfo_vos/tools/shift_equivariance.py`

**Interfaces:**
- Consumes: `shift_keys`, `ccos` (Task 2); `make_layered_clip(..., layers=True)` for I, mf, B.

**Construction:**
- 30 val clip specs (`dataset.clip_specs('val')[:30]`), the middle frame of each.
- Inputs:
  - P0 = mf·I + (1 − mf)·B;
  - P_Δ = the same with I and mf translated by Δ native px in x (`cv2.warpAffine`, linear), over the **same** B.
- Δ ∈ {3, 6.25, 12.5, 25, 50}.
- Encode both. Shift E(P0) by Δ with `shift_keys` (roll and bilinear).
- On person cells of P_Δ (coverage > 0.5 at stride s):
  - ε = 1 − ccos(shift(E(P0)), E(P_Δ));
  - δ = 1 − ccos(E(P_Δ), E(B)).
- Print per stride, Δ and mode: median ε, p90 ε, median δ.
- Write `results/benchmark/rr/feat_integral/g0_shift.csv`.
- Save one montage `images/benchmark/feat_integral/g0_example.png`: P0, P_Δ, the ε map at s16 for Δ = 12.5.
- Last line: `G0 PASS (mode=...)` if p90 ε16 < ½ median δ16 at every Δ for at least one mode (prefer bilinear on a
  tie), else `G0 FAIL`.

- [ ] **Step 1: Run** `$PY -m benchmark.nfo_vos.tools.shift_equivariance`. Expected: the table plus a `G0 PASS/FAIL`
  line, in < 10 min.
- [ ] **Step 2: On FAIL,** stop. Log one line in the failure log and return to the user: RoPE fallback = design change.
  **On PASS,** record the chosen mode as the default in `shift_keys`.
- [ ] **Step 3: Commit:** `git commit -m "G0: SAM2 feature shift equivariance under subject motion (<result>)"` (script + csv).

---

### Task 5: Stage 0 and G1 (learned vs fixed vs pixel vs frame t)

**Files:**
- Create: `benchmark/nfo_vos/adapter/feat_stage0.py`
- Test: `benchmark/nfo_vos/tests/test_feat_integral.py`

**Interfaces:**
- Consumes: `build(feat=True, with_pixel=...)` (Task 3), `FeatIntegral`, `feat_loss`, `ccos` (Task 2), `Clips` (existing).
- Produces:
  - `main(steps=400, seed=0, eval_every=100, n_val=30)`;
  - checkpoint `results/benchmark/rr/feat_integral/stage0_seed{seed}.pt` (the best val L_C);
  - `g1_seed{seed}.json`.

**Procedure:**
- Training: one train clip per step, `build(pred, raw, max_targets=32, rng=rng, feat=True)`, loss `feat_loss`, AdamW
  3e-4, wd 1e-4, cosine schedule.
- Val: 30 fixed val clips with `with_pixel=True`, all targets.
- G1 table: mean ccos on Ω per stride for:
  - learned (best checkpoint);
  - fixed (`fixed=True`);
  - pixel (`C16/8/4` of `with_pixel`);
  - F_t (`X16/8/4`);
  - plus the floor: ccos(F_t, T) on `Vis16`.
- 95% clip-bootstrap CIs (2000 resamples) of learned − fixed and learned − F_t at s16.
- Verdict lines:
  - `G1 KILL` if learned ≤ F_t at s16;
  - else `G1 LEARNED>FIXED` if the CI of learned − fixed excludes 0 from above;
  - else `G1 FIXED>=LEARNED`. Then G2 uses the fixed aggregator (no Agg parameters, λ = 0).

- [ ] **Step 1: Write the failing test** (Review Focus 3):

```python
def test_feat_loss_zero_on_empty_omega():
    C = {s: torch.randn(2, c, h, h, requires_grad=True) for s, c, h in (('16', 256, 8), ('8', 64, 16), ('4', 32, 32))}
    Om = {s: torch.zeros(2, 1, x.shape[-2], x.shape[-1], dtype=torch.bool) for s, x in C.items()}
    l = feat_loss(C, {s: torch.randn_like(x) for s, x in C.items()}, Om, torch.zeros(1, 256, 1, 1))
    assert l.item() == 0 and l.requires_grad
```

- [ ] **Step 2: Run** the test. Expected: PASS if Task 2 implemented `feat_loss` as specified, else fix `feat_loss`.
- [ ] **Step 3: Implement** `feat_stage0.main` as above. Print `it {i} L_C {..} {s/it}` every 25 steps, unbuffered.
- [ ] **Step 4: Run** `$PY -m benchmark.nfo_vos.adapter.feat_stage0 400 0` on the laptop. Expected: the G1 table and one
  verdict line. Record the numbers in the spec's §6 as a result block.
- [ ] **Step 5: On `G1 KILL`,** stop: one line in the failure log, report to the user.
- [ ] **Step 6: Commit:** `git commit -m "Stage 0 + G1: learned feature integral vs fixed / pixel / frame t on hidden-recoverable cells (<verdict>)"`

---

### Task 6: Fusion training with the feature integral

**Files:**
- Modify: `benchmark/nfo_vos/adapter/fusion_adapter.py` (`main`, `evaluate`)
- Modify: `benchmark/nfo_vos/adapter/fusion_nfo.py` (`items`, `evaluate`)
- Modify: `benchmark/nfo_vos/adapter/fusion_data.py` (`encode`)
- Modify: `benchmark/nfo_vos/adapter/fusion_adapter.sbatch` (env `FEAT`, `AGG`, `FIXED`)
- Modify: `benchmark/nfo_vos/tools/remote_smoke.py` (one training step with `FEAT=1`)

**Interfaces:**
- Consumes: Tasks 2, 3 and 5 (`stage0_seed0.pt`).
- Produces:
  - `main(steps, seed, ..., feat=False, agg_ckpt=None, fixed=False, lam=1.0)`.
    - With `feat`, each step:
      1. `f, _ = build(..., feat=True)`;
      2. `o_agg = agg(f[K..], f['Kvalid16'], {X..}, {B..}, f['age'], fixed)`;
      3. `f.update(C16/8/4 = o_agg[...], a = o_agg['a'])`;
      4. loss = `loss_fn(...)` + `lam · feat_loss(o_agg C, f[T], f[Om], f['mu16'])`. With `fixed`: no Agg parameters,
         λ = 0.
    - Agg is its own AdamW parameter group, same lr/wd.
    - Checkpoints `results/benchmark/rr/layers_fusion_adapter/feat_seed{seed}/fusion_it{it}.pt` contain both state dicts
      (`{'fusion': ..., 'agg': ...}`).
    - Selection is unchanged: best seq1 SEL adapter J.
  - `fusion_nfo.items(split, pred, feat=False)`: with `feat`, each item also carries:
    - `'H'`: the 6 native history frames j−6..j−1;
    - `'dx1024'`: the same Theil–Sen shifts `items` already computes, converted to the 1024 frame.
  - `fusion_data.encode(pred, clips, feat=False)`: encodes `'H'` and returns `K*`, `Kvalid16`, `age`.
  - adapter-noC under `feat`: C := B, a := 0, as today.
- Defaults (`feat=False`) are unchanged.

- [ ] **Step 1: Write the failing test** (GPU): one `main`-equivalent step on one val clip with `feat=True`. Both the
  Fusion `out` conv and the Agg `h` last conv receive a non-zero gradient after a backward through `loss_fn + feat_loss`
  (the analogue of the existing `test_zero_convs_receive_gradient`). Expose the step as
  `fusion_adapter.train_step(pred, sam, net, agg, raw, rng, lam)` and use it in `main`.
- [ ] **Step 2: Run** it. Expected: FAIL (no `train_step`).
- [ ] **Step 3: Implement** `train_step`, the `feat` options of `main`/`evaluate`/`items`/`encode`, and the sbatch/env
  flags (`FEAT=1 AGG=<path> [FIXED=1]`).
- [ ] **Step 4: Run** the test, the full suite and the parity snapshot. Then `FEAT=1 $PY -m benchmark.nfo_vos.tools.remote_smoke`.
  Expected: PASS; IDENTICAL; the smoke prints one feat training step and its peak memory (expected < 12 GB).
- [ ] **Step 5: Commit:** `git commit -m "Fusion training on the learned feature integral (FEAT=1): train_step, seq1 selection items, sbatch flags"`

---

### Task 7: One-pass inference, timing

**Files:**
- Modify: `benchmark/nfo_vos/adapter/inference.py` (`masks_from_arrays`, `fast_masks`)
- Modify: `benchmark/nfo_vos/eval/dump_arms.py` (arm `adapter-feat` → `feat_seed0` checkpoint; `adapter-feat-noC`)
- Modify: `benchmark/nfo_vos/tools/timing.py` (methods `feat`, `trk`)
- Test: `benchmark/nfo_vos/tests/test_feat_integral.py`

**Interfaces:**
- Consumes: `FeatIntegral`, `shift_keys` (Task 2); `hist_warps` (existing).
- Produces: `masks_from_arrays(..., agg=None, fixed=False)`.
  - With `agg`:
    - skip `_gate_gpu`, the pixel integral and `enc(C)`;
    - for j ≥ hist:
      - keys = `F[j-hist:j]` per stride, shifted by dx1024 from `hist_warps(kc, j, hist)`. **Replace the inline
        Theil–Sen block with this call in both paths**, so dx is shared by construction;
      - `age` = hist..1;
      - B = `Bf`;
      - f gets `C*`, `a` from `agg`.
  - The `aux=` hook's `box_signals` gets `mass = a`. The switch is not refitted (deferred).

- [ ] **Step 1: Write the failing tests:**
  - `test_inference_feat_matches_pixel_before_history` (GPU, Review Focus 5): on one synthetic clip as arrays, masks
    for j < 6 from `agg=FeatIntegral(...)` equal the pixel path's masks exactly.
  - `test_inference_dx_equals_build_dx` (CPU): for a random smooth track, the dx1024 that inference computes at j
    equals `build`'s for target t = j.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** the `agg` path, the dump arm and the timing methods.
- [ ] **Step 4: Run** the tests, the full suite and the parity snapshot. Then the timing:

```bash
TC_BATCH=1 $PY -m benchmark.nfo_vos.tools.timing trk
TC_BATCH=1 $PY -m benchmark.nfo_vos.tools.timing feat
```

  Expected: PASS; IDENTICAL; `trk` (the floor) and `feat` ms/frame printed. **G-speed: feat ≤ 40 ms/frame.** Record
  both numbers in the spec.
- [ ] **Step 5: Commit:** `git commit -m "One-pass inference with the feature integral; timing: trk floor <x> ms, feat <y> ms/frame"`

---

### Task 8: G2, one cluster training and the seq1 decision

**Files:**
- Modify: `docs/vos_benchmark/nfo_vos_pilot_findings.md` (on pass) or `docs/vos_benchmark/nfo_failure_log.md` (on fail)

- [ ] **Step 1 (user, cluster):** run the smoke test, then the training:
  - `$PY -m benchmark.nfo_vos.tools.remote_smoke`;
  - `FEAT=1 AGG=results/benchmark/rr/feat_integral/stage0_seed0.pt SEED=0 STEPS=1000 sbatch benchmark/nfo_vos/adapter/fusion_adapter.sbatch`.
  - Copy back `feat_seed0/` and its log.
- [ ] **Step 2: Score both checkpoints in one environment.** On the laptop, for the seq1 windows only:
  - `dump_arms adapter-pf` (pixel seed 0) and `dump_arms adapter-feat`;
  - `box_metrics score adapter-pf adapter-feat` (seq1 rows);
  - hand J for both from `fusion_nfo.evaluate(pred, net, items('select'))`.
- [ ] **Step 3: Decision (pre-registered).**
  - **PASS** if feat ≥ pixel − 0.02 on seq1 hand J **and** on seq1 box AUC, **and** Task 7 timing ≤ 40 ms/frame.
  - On PASS, write a findings section with the G0/G1/G2 numbers and the next step: 3 seeds + the seq2–4 headline, then
    the deferred arms.
  - On FAIL, one line in the failure log with the numbers, plus a hypothesis and the test that would check it.
- [ ] **Step 4: Commit:** `git commit -m "G2: learned feature integral vs pixel adapter on seq1 (<verdict>)"`
