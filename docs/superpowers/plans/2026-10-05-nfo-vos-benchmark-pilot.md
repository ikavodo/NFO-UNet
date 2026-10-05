# NFO VOS benchmark pilot: implementation plan

Spec: `docs/superpowers/specs/2026-10-02-nfo-vos-benchmark-pilot-design.md` (rev. 3). The spec is
binding. This plan only orders the work.

## Global constraints

- Code lives in `benchmark/nfo_vos/`, run from the repo root as `python -m benchmark.nfo_vos.<mod>`.
- Environments: `PY=../master_thesis/.venv/bin/python` (sam2 1.1.0, skimage, pytest) for
  everything except T1. T1 uses `../samurai/.venv/bin/python` (SAMURAI repo commit `76ba195`).
- Checkpoint for all SAM2 calls: `../samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt`.
- Native frames: `data/nfo_final/nfo_final/seq1/{idx:05d}.jpg` (800×600). 224 frames and GT:
  `data/nfo_processed/seq1_gt/`.
- Native→224 mapping of every native-resolution mask uses exactly the GT call
  `scale_and_pad_img_to_square((m*255).astype(uint8), BoundingBox(0,0,0,0), 224)`
  (`gen_nfo_pseudo_masks.py:346`), then `> 127`.
- Cache: `results/benchmark/pilot/masks/<method>/<trial_id>.npz` (key `masks`, [51,H,W] bool,
  plus `frames`). Results: `results/benchmark/pilot/`. Images: `images/benchmark/pilot/`.
- The order is set by spec §6 rule 1. B0 is run and scored before T1 and T4 are built. If B0
  J@50 ≳ 0.85 from every start, stop and report.

### Task 1: metrics (vendored DAVIS + p/r/DRE/NRE/P_norm/decay)

- Vendor `davis2017/metrics.py` from davisvideochallenge/davis2017-evaluation @
  `ac7c43fca936f9722837b7fbd337d284ba37004b` into `benchmark/nfo_vos/davis_metrics.py`, plus
  its LICENSE as `benchmark/nfo_vos/DAVIS_LICENSE`.
- `benchmark/nfo_vos/metrics.py`: `frame_metrics(P, G) -> dict(J, F, p, r, area_p, area_g)`;
  `dre_nre(P_stack, G_stack)`; `pnorm(P_stack, boxes_224)` (LaSOT normalised precision AUC over
  thresholds 0..0.5); `decay(values)` (DAVIS quartile decay); `native_to_224(mask)`.
- Tests `benchmark/nfo_vos/tests/test_metrics.py`: identity 1/J = 1/p + 1/r − 1 on random masks;
  empty/empty → J = 1 (DAVIS convention); a disjoint non-empty prediction counts toward DRE, an
  empty one toward NRE; `native_to_224` on a native GT-like mask reproduces a stored `_sammask.png`
  shape (224×224).
- Run: `$PY -m pytest benchmark/nfo_vos/tests -q`. Expected: all pass.

### Task 2: trials

- `benchmark/nfo_vos/trials.py`: segments via `find_segments`, pilot = seq1 segments 4 and 5.
  Starts every 10 frames while t₀+50 ≤ last `_sammask.png` frame. Admissibility D(p*) ≥ 2.
  Prompt: GT box (native, `gt_to_native`) and p* (224 → native through the same pad/scale). Also
  v(t₀), n_frag(t₀), and the warm-up range (person-free gap frames, > 40 from every segment).
  Writes `results/benchmark/pilot/trials.json`.
- Tests: segment 4 = 1154–1308 and segment 5 = 1476–1621; 11 and 10 starts; warm-up ranges
  946–1113 and 1349–1435 (spec §2).
- Run: `$PY -m pytest benchmark/nfo_vos/tests -q`, then `$PY -m benchmark.nfo_vos.trials`.

### Task 3: B0/T1 runner

- `benchmark/nfo_vos/run_sam2_video.py --method b0|t1`. Imports nothing from NFO-UNet that needs
  more than numpy/cv2, so it runs in both venvs. Per trial: stage native frames t₀..t₀+50, then
  `init_state`, `add_new_points_or_box(points=[p*], labels=[1], box)` at frame 0, and
  `propagate_in_video`. Masks are logits > 0. Same autocast (fp16) and offload settings for both
  methods (SAMURAI's `main_inference.py:76`), so B0 and T1 differ only in the predictor/config.
  Saves native masks, time per frame, and peak GPU memory.
- Smoke test: `--limit 1 --max-frames 3` writes an npz of shape [3, 600, 800].

### Task 4: scorer + rule-1 check

- `benchmark/nfo_vos/score.py`: maps native masks to 224 and writes one per-frame CSV
  `results/benchmark/pilot/per_frame.csv` (method, trial, t₀, t, Δt, J, F, p, r, areas,
  centroid error, v, n_frag). Per-trial summary `per_trial.csv` (J@10/25/50, mean J&F, decay,
  DRE, NRE, P_norm). Montage per trial under `images/benchmark/pilot/`.
- Run on B0 and evaluate rule 1.

### Task 5: T1 run (SAMURAI)

- `../samurai/.venv/bin/python -m benchmark.nfo_vos.run_sam2_video --method t1`, then re-score.

### Task 6: T4 family (T4, T4-1, T4-2 m=1..7, T4-3)

- `benchmark/nfo_vos/run_t4.py`, built on one causal tracker run per trial. MOG2 is warmed on the
  trial's warm-up frames, then the 224 frames t₀..t₀+50 go through the foreground pipeline,
  `detect_blobs`, and `track_blobs` (online). The track at t₀ is the detection whose centre lies in
  the GT box (largest area if several). It is followed by track id, and if the track dies, the
  method re-acquires the nearest detection to the last predicted position. person_height = GT box
  height at t₀ → `scale_relative_params`. Per t: buffer [max(t₀, t−6), t], OLS vx over the
  in-buffer track detections, `align_frames` on the buffer, then the variants below.
  - T4: median `fuse`, `segment_reference` (base_plus), `visibility_within_mask`,
    `project_to_frame` at the newest frame.
  - T4-1: the reference is the current frame alone.
  - T4-2(m): vote of the aligned person blobs, ≥ min(m, N) → same visibility step on T4's
    reference.
  - T4-3: the frame-t person blob (`restrict_to_nearby`).
- Tests: T4-1's visibility step is the identity (residual 0); the vote with m = 1 is the union and
  with m = N the intersection.

### Task 7: analysis

- `benchmark/nfo_vos/analyze.py`: the T4-2\* pick (best m by mean J@50 over the pilot), the
  method table, the §5 comparisons, the frame fixed-effects fit
  J = a_t + b·v + c·Δt + d·v·Δt + e·n_frag per method (numpy lstsq with frame dummies), the
  J@50-vs-v scatter, and `run.json`. Writes `results/benchmark/pilot/summary.md`.
