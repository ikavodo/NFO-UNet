# One-Pass Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A per-frame adapter on frozen SAM2.1-b+ that needs one image-encoder pass per frame. A feature-space
composite with a learned gate replaces the pixel composite, the hand-made evidence mass and the explicit differences.
It is trained on the existing synthetic pipeline and evaluated on NFO seq2–4 against the current adapter.

**Architecture:**
- New module `layers/onepass.py`:
  - shift and history helpers, shared by training and inference;
  - the `OnePass` network.
- The synthetic `build` gets an `onepass` flag that returns all clip-frame features plus a history table. The pixel
  composite is not built.
- NFO inference reuses `video_adapter.fast_masks`'s single-encode loop.
- Training, eval, dumps and the cluster scripts gain a `onepass` mode / arms.

**Tech Stack:** PyTorch (bf16/fp16 autocast), SAM2.1-b+ image predictor (frozen), numpy/scipy (Theil–Sen), pytest.
Interpreter: `../samurai/.venv/bin/python`; tests use `TC_BATCH=1 PYTHONPATH=.`.

**Spec:** `docs/superpowers/specs/2026-10-20-one-pass-adapter-design.md`, including its "Expert pipeline review" section.

## Global Constraints

- **History.** K = 6 at test. Training draws n_h ∈ {4, 5, 6} per clip: the w_k of the dropped oldest frames are zeroed
  and renormalised, and the tensor shape stays K = 6.
- **Shift.** Δx_k = fx[t] − fx[t−k] in native px, from Theil–Sen over [t−12, t] of the tracker x (`factorised_gtfree.theil_at`).
  - Converted to grid cells as Δx · 1024 / W_native / s for s ∈ {16, 8, 4}. W_native comes from the frame, never from `S1`.
  - Bilinear, border replicate, `align_corners=True`; dst(x) = src(x − Δ).
- **Order.** Gate first, on the unshifted features against the unshifted empty scene. Then shift the gate and the projected features.
- **Gate.** One map per past frame at stride 4: g = σ(φ([P⁴F⁴_{t−k}, P⁴F⁴_B])), with φ = Conv1×1(64→32) → GELU →
  Conv3×3(32→1). Its last-layer weights are 0 and its bias 0, so g = 0.5 at init. Average-pooled ×2 → stride 8, ×4 → stride 16.
- **Recency.** w_k ∝ exp(−k/τ) for k = 1 (newest) … 6. log τ is learned, init log 3.5.
- **Composite.** Ĉ^s = (Σ w g P^s(F_{t−k}) + ε Σ w P^s(F_{t−k})) / (Σ w g + ε) with ε = 1e-3, and m^s = Σ w g.
  This is computed in **fp32, outside autocast**, and so is the gate sigmoid.
- **Empty history** (mode 2 in training, noC at inference, or all w_k zeroed): Ĉ^s := P^s(F^s_B) and m^s := 0.
- **Adapter input** per stride: [P^s F^s_t, Ĉ^s, P^s F^s_B, m^s]. The blocks, widths (w = 64), projections P
  (`CH`/`D_` of `fusion_adapter`) and zero-init output convs are the same as `Fusion`.
- **Token.** The tracker point token is SAM2's own embedding (as `Fusion` with `trk`).
  - δ = zero-init MLP(proto).
  - proto = the m¹⁶-weighted mean, inside box_trk, of the raw 256-channel stride-16 history composite (same formula, raw features).
  - Scale 1 − exp(−Σ_box m¹⁶ / κ), log κ init log(500/16).
- **Weight decay.** log τ and log κ have weight decay 0.
- **Training.** LOSS=v34, STEPS=1000 (cosine), AdamW lr 3e-4, wd 1e-4, 32 targets per clip, eval every 100 steps. The
  primary checkpoint is step 200; step 600 is a declared secondary.
- **Unchanged:** the loss, data generator, tracker, prompts and SAM2 weights. The old `Fusion` checkpoints must still
  load and run.

## Review Focus

1. **A target with fewer than 6 predecessors, or a NaN / non-finite Δx** (tracker lost): use the trk decode (per-frame
   SAM2 with tracker box + point) for that frame, no exception. → test in Task 4.
2. **Mode-2 clip or noC at inference:** exactly Ĉ = P F_B and m = 0, with no NaN from 0/ε. → test in Task 1.
3. **fp16 autocast (Turing nodes):** the history arithmetic stays finite. The composite is fp32 even when the caller is
   under fp16 autocast. → test in Task 1.
4. **NFO frames at 800×600 vs synthetic 960×540:** the shift in cells uses each clip's own width. → covered by the parity
   test in Task 4 and a unit test in Task 1.
5. **Gate collapse** (flat 0.5 or saturated): visible in the training log, not discovered at test time. → logging in Task 3.

---

### Task 1: Shift and history helpers

**Files:**
- Create: `benchmark/nfo_vos/layers/onepass.py`
- Test: `benchmark/nfo_vos/tests/test_onepass.py`

**Interfaces:**
- Produces:
  - `shift_x(x: Tensor[N, C, h, w], dx_cells: Tensor[N]) -> Tensor[N, C, h, w]`: bilinear, border replicate, dst(x) = src(x − dx).
  - `history_table(kc_x: np.ndarray[T], ts: list[int], K: int = 6, n_h: int | None = None) -> (idx: np.ndarray[n, K] int, dx: np.ndarray[n, K] float32, keep: np.ndarray[n, K] bool)`.
    - Column k−1 is frame t−k.
    - dx in native px via Theil–Sen over [t−12, t].
    - keep[:, k−1] = k ≤ n_h (n_h None → K).
    - Rows with non-finite dx get keep all False.
  - `composite(P: dict[str, Tensor[n, K, C, h, w]], raw16: Tensor[n, K, 256, 64, 64], g4: Tensor[n, K, 1, 256, 256], PB: dict[str, Tensor[C, h, w]], rawB16: Tensor[256, 64, 64], dx: Tensor[n, K], keep: Tensor[n, K], log_tau: Tensor, W_native: int) -> (C: dict[str, Tensor[n, C, h, w]], m: dict[str, Tensor[n, 1, h, w]], C_raw16: Tensor[n, 256, 64, 64])`.
    - Inputs are **unshifted**; this function shifts.
    - fp32 inside `torch.autocast('cuda', enabled=False)`.
    - Empty-history rows: C = PB, C_raw16 = rawB16, m = 0.

- [ ] **Step 1: Write the failing tests**

```python
def test_shift_integer_is_roll_with_border():
    x = torch.arange(10.)[None, None, None].repeat(1, 2, 3, 1)
    y = shift_x(x, torch.tensor([2.]))
    assert torch.equal(y[0, 0, 0], torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7.]))

def test_shift_matches_video_adapter_pixel_shift():
    from benchmark.nfo_vos.layers.video_adapter import _shift_x
    x = torch.rand(3, 40, 60)
    assert torch.allclose(shift_x(x[:, None], torch.tensor([1.3, -2.7, 0.])) [:, 0], _shift_x(x, torch.tensor([1.3, -2.7, 0.])), atol=1e-5)

def test_history_table_columns_and_dropout():
    kc = np.arange(40.) * 2                                  # 2 px / frame
    idx, dx, keep = history_table(kc, [10, 20], n_h=4)
    assert idx[0].tolist() == [9, 8, 7, 6, 5, 4] and np.allclose(dx[0], [2, 4, 6, 8, 10, 12], atol=1e-4)
    assert keep[0].tolist() == [True] * 4 + [False] * 2

def test_history_table_nan_tracker_drops_row():
    kc = np.arange(40.); kc[15] = np.nan
    _, _, keep = history_table(kc, [16])
    assert not keep.any()

def test_composite_empty_history_is_empty_scene_and_zero_mass():
    # all keep False -> C == PB exactly, m == 0, no NaN
def test_composite_uniform_gate_is_recency_mean():
    # g = 0.5 everywhere, dx = 0: C^s equals Σ w_k P_k / Σ w_k within 1e-5 (ε term included)
def test_composite_fp32_under_fp16_autocast():
    # called inside torch.autocast('cuda', dtype=torch.float16): outputs are float32 and finite
def test_composite_cell_scale_uses_native_width():
    # a single bright column; dx = 16 native px, W_native 1024 -> stride-16 map shifts by 1 cell; W_native 800 -> 1.28 cells
```

- [ ] **Step 2: Run the tests and see them fail**
  - Run: `TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m pytest benchmark/nfo_vos/tests/test_onepass.py -v`
  - Expected: FAIL (ImportError).
- [ ] **Step 3: Implement** `shift_x`, `history_table` and `composite` in `onepass.py`.
  - `shift_x` generalises `video_adapter._shift_x` to [N, C, h, w]. One grid per row: `grid_sample`, border, `align_corners=True`.
  - `composite` pools g4 with `F.avg_pool2d` (2 → s8, 4 → s16), then shifts both P and g with dx · 1024 / W_native / s.
- [ ] **Step 4: Run the tests and see them pass** (same command; expected: all PASS).
- [ ] **Step 5: Commit** `git commit -m "onepass: shift / history table / feature composite helpers (shared by train and test)"`

### Task 2: OnePass network

**Files:**
- Modify: `benchmark/nfo_vos/layers/onepass.py`
- Test: `benchmark/nfo_vos/tests/test_onepass.py`

**Interfaces:**
- Consumes: Task 1's helpers. From `fusion_adapter`: `CH`, `D_`, `block`, `decode`.
- Produces: `class OnePass(nn.Module)`.
  - `__init__(self, w=64, K=6)`.
  - `frame_terms(self, F16, F8, F4) -> dict(P16, P8, P4, g4)`: per frame, no history. g4 is the gate logit-σ against the
    stored empty scene; call `set_scene(B16, B8, B4)` first.
  - `set_scene(self, B16, B8, B4) -> None`: caches P^s F_B and raw B16 for the clip or sequence.
  - `forward(self, f: dict, sam) -> dict`, where f holds:
    - `X16/X8/X4` [n] (target features);
    - `T16/T8/T4` [T] (all clip-frame features);
    - `idx, dx, keep` [n, K] (from `history_table`);
    - `W_native` (int);
    - `trk`, `box_trk` (1024 frame, as `build`).

    Returns `o` with `X16/X8/X4` (+ residuals), `token`, and `stats` = dict(m_sum, gate_std).
  - `param_groups(self, wd=1e-4) -> list[dict]`: log τ and log κ at wd 0.

- [ ] **Step 1: Write the failing tests**

```python
def test_step0_equals_trk(sam_pred, nfo_window_feats):
    # fresh OnePass: decode(sam, o, token=o['token']) == decode(sam, dict(f, box=f['box_trk']), point=True) (max |Δlogit| < 1e-3)
def test_gate_init_half(nfo_window_feats):
    # frame_terms(...)['g4'] == 0.5 everywhere
def test_kappa_and_tau_init_and_no_decay():
    net = OnePass(); assert abs(net.log_kappa.exp() - 500 / 16) < 1e-3 and abs(net.log_tau.exp() - 3.5) < 1e-4
    assert {id(p) for g in net.param_groups() if g['weight_decay'] == 0 for p in g['params']} == {id(net.log_tau), id(net.log_kappa)}
def test_old_fusion_checkpoint_still_loads():
    # Fusion(token=True).load_state_dict(torch.load(dump_arms.CKPT)) succeeds
```

  The fixtures (`sam_pred`, `nfo_window_feats`) go in `test_onepass.py`. They encode the first 12 frames of the first
  seq2 window once, using `video_adapter.fast_masks`'s `enc`; factor `enc` out to a module-level `encode_gray(pred, img)`
  in `video_adapter.py` so it can be imported.

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `OnePass`.**
  - z-inputs: `[P F_t, C^s, P F_B, m^s]`. Blocks: `block(3*D_[s]+1, w)` for s16, `block(3*D_[s]+1 + w, w)` for s8 and s4.
  - Zero-init out convs and token MLP, as `Fusion`.
  - Token: SAM2 point embedding at `trk` with `box_trk` (as `Fusion.forward`'s `trk` branch), corrected by δ on the point token.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `git commit -m "onepass: OnePass network (zero-init residuals, learned gate, box-restricted token prototype)"`

### Task 3: Training path and gate logging

**Files:**
- Modify: `benchmark/nfo_vos/layers/fusion_data.py` (`build`)
- Modify: `benchmark/nfo_vos/layers/fusion_adapter.py` (`main`, `evaluate`)
- Modify: `benchmark/nfo_vos/layers/fusion_nfo.py` (`evaluate`)
- Test: `benchmark/nfo_vos/tests/test_onepass.py`

**Interfaces:**
- Consumes: `history_table`, `OnePass`.
- Produces:
  - `build(pred, raw, max_targets=None, rng=None, dev='cuda', all_targets=False, onepass=False)`.
    - With `onepass=True` it skips the pixel composite and its encode.
    - It returns f with `T16/T8/T4` (all frames, from the encode that already exists), `X*` = `T*[ts]`, `idx`, `dx`,
      `keep`, `W_native`, `B*`, plus the existing `V`, `W`, `box`, `box_trk`, `trk`, `mode`.
    - n_h ~ {4, 5, 6} from `rng`. Mode-2 targets get keep all False.
  - `fusion_adapter.main(mode='onepass', ...)` builds `OnePass`, uses `param_groups`, and calls `build(..., onepass=True)`.
    The output dir is `onepass_seed{seed}_{LOSS}`.
  - `evaluate` logs `gate_std`, median Σm, and m on person (mean m¹⁶ over V) for onepass.
  - `fusion_nfo.evaluate(pred, net, sel)` dispatches on `isinstance(net, OnePass)` to `onepass_sel(pred, net) -> dict`,
    with the same keys as today (`adapter`, `trk-box`, `adapter-noC` → `{J, P, R}`) plus `m_sum`. It runs
    `onepass_masks` (Task 4) on the seq1 windows that contain the brush frames.

- [ ] **Step 1: Write the failing test.** `test_build_onepass_shapes`: one val clip, `build(..., onepass=True, max_targets=4)`.
  - Assert `f['idx'].shape == (4, 6)`.
  - Assert `T16.shape[0] == len(raw['X'])`.
  - Assert `'C16' not in f`.
  - Assert mode-2 rows have keep all False.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the `build` flag, `main` mode, eval logging and the `fusion_nfo` dispatch.
  - Print one `GATE it N: syn gate_std … Σm … m_on … | seq1 Σm …` line per eval.
  - Collapse is pre-declared in the spec: synthetic gate std < 0.05 at step 200, or seq1 median Σm outside [0.5×, 2×]
    the synthetic median. Print `COLLAPSE` when it holds.
- [ ] **Step 4: Run the test, expect PASS. Then smoke-train locally:**
  - Run: `LOSS=v34 TC_BATCH=1 PYTHONPATH=. ../samurai/.venv/bin/python -m benchmark.nfo_vos.layers.fusion_adapter onepass 4 0 2 32 2 0`
  - Expected: finite losses, `VAL/SEL/GATE` lines at it 0 and it 2, peak memory printed < 15 GB.
- [ ] **Step 5: Commit** `git commit -m "onepass: training mode, build(onepass=True), gate-collapse logging"`

### Task 4: NFO inference, parity, dumps, timing

**Files:**
- Modify: `benchmark/nfo_vos/layers/video_adapter.py`
- Modify: `benchmark/nfo_vos/downstream/dump_arms.py`
- Modify: `benchmark/nfo_vos/timing.py`
- Test: `benchmark/nfo_vos/tests/test_onepass.py`

**Interfaces:**
- Consumes: `OnePass`, `history_table`, `encode_gray`.
- Produces: `onepass_masks(pred, net, t, noC=False, internals=False) -> (masks [T, H, W] bool, sec_per_frame[, internals: dict j -> (C16, m16, proto)])`.
  - It runs the same single-encode loop as `fast_masks`: frames encoded once, the tracker on those F16, the empty
    scene once (`set_scene`).
  - Frame terms are computed per frame. Targets j < 6, rows with keep all False, or non-finite dx fall back to the trk
    decode; noC sets keep all False.
  - Arms in `dump_arms.ARM_CKPT`:
    - `onepass-s0/s1/s2` → `layers_fusion_adapter/onepass_seed{s}_v34/fusion_it200.pt`;
    - `onepass-s{0,1,2}-noC`;
    - `onepass-s0-it600` (declared secondary).

    Routed to `onepass_masks`.
  - `timing.py onepass` prints ms per frame over the 61 windows.

- [ ] **Step 1: Write the failing tests.**
  - `test_train_test_parity`: for one seq2 window and target j = 20, the internals from `onepass_masks(..., internals=True)`
    equal `OnePass.forward`'s on a batch assembled with `history_table` and the same features. Assert allclose for C16,
    m16 and proto (atol 1e-4).
  - `test_nan_tracker_falls_back_to_trk`: patch `kernel_track` to return NaN at frame 20. The mask at 20 equals the trk
    decode and no exception is raised.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `onepass_masks`, the `dump_arms` routing and `timing.py onepass`.
- [ ] **Step 4: Run the tests, expect PASS.** Then run `timing.py onepass` with an untrained net (masks equal trk; timing only).
  Expected: ≤ 35 ms per frame on the laptop RTX PRO 4000 (spec target).
- [ ] **Step 5: Commit** `git commit -m "onepass: NFO inference path (single encode), parity test, dump arms, timing"`

### Task 5: Cluster runs and the decision rule

**Files:**
- Modify: `benchmark/nfo_vos/layers/remote_smoke.py` (add one onepass training step to `_train`)
- Modify: `benchmark/nfo_vos/layers/fusion_adapter.sbatch` (comment: `MODES=onepass`)
- Create: `benchmark/nfo_vos/downstream/onepass_rule.py`

**Interfaces:**
- Consumes: masks of `adapter-pf`, `adapter-s1`, `adapter-s2`, `adapter-pf-noC`, `adapter-s1-noC`, `adapter-s2-noC`,
  `onepass-s{0,1,2}`, `onepass-s{0,1,2}-noC`, `t1-c2`. Also the box CSV, hand rows, and one recognition run via
  `recognition.main(7, 'uniform', arms)` with `NFO_RECOG_TAG=_onepass`.
- Produces: `onepass_rule.py` prints the three conditions of decision rule 1 with values and CIs, then `ADOPT` or
  `KEEP TWO-PASS`:
  - one-pass − one-pass-noC per seed, against the bar ½ · mean_s(adapter − adapter-noC), for box AUC and recognition conf;
  - one-pass − adapter, for hand J ≥ −0.010;
  - one-pass − SAMURAI box AUC > 0 with its CI excluding 0.

  It also prints the declared secondary (it600) and per-seed AMP dtype from the training logs.

- [ ] **Step 1: Write the failing test** `test_rule_logic`: synthetic per-window dicts where the one-pass gain is 0.6×
  the bar → `ADOPT`; 0.4× → `KEEP TWO-PASS`; hand J −0.02 → `KEEP TWO-PASS`.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `onepass_rule.py` (decision function `rule(...) -> str` plus a `main()` that loads masks and
  CSVs). Extend `remote_smoke._train` with one `OnePass` step.
- [ ] **Step 4: Run the test, expect PASS.** Then local smoke `remote_smoke` → `SMOKE OK`.
- [ ] **Step 5: Commit and push** `one-pass-adapter`. Hand the user the cluster commands:
  - `git checkout one-pass-adapter && git pull`;
  - `for S in 0 1 2; do LOSS=v34 STEPS=1000 SEED=$S MODES=onepass sbatch --export=ALL --array=0 benchmark/nfo_vos/layers/fusion_adapter.sbatch; done`;
  - after the it200 checkpoints exist, the dump arrays for `onepass-s{0,1,2}` and `-noC`;
  - after copying back, `python -m benchmark.nfo_vos.downstream.onepass_rule`.

### Conditional (only if the rule prints KEEP TWO-PASS): old-gate feature composite

Not built now (YAGNI). If needed, one arm `onepass-oldgate-s0` replaces `g4` with the pixel z-score gate (q90
threshold), area-pooled to 256², with τ fixed at 3.5. It reuses `composite` unchanged and is trained with seed 0 only.
It separates "features vs pixels" from "learned vs hand-made gate" (spec, review B2).
