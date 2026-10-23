# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## What this repo is

Two things, separated by directory:

1. **The NFO VOS benchmark and the per-frame adapter** (`benchmark/nfo_vos/`, current work). Goal: a publishable
   benchmark for segmenting the *visible* part of a person under fragmented occlusion (foliage) on the NFO
   sequences, and a method that beats SAM2.1 video (B0) and SAMURAI (T1) there.
2. **The original spatio-temporal U-Net** of the paper "Person localization under fragmented occlusion" (root:
   `train_main.py`, `test_main.py`, `network/`, `dataset/`, `config/`, `eval/`, `utils/`, `gen_data/gen_*`). Kept
   as published; not used by (1) except `utils/bb_utils.py` and `gen_data/` helpers.

Results and their numbers live in `docs/vos_benchmark/nfo_vos_pilot_findings.md` (constructive) and
`docs/vos_benchmark/nfo_failure_log.md` (one line per dead end). The collaborator briefing is `docs/briefing/index.html`.
Code removed in the 2026-10-21 cleanup (≈ 90 experiment scripts, the blob tracker and live-tracker app `tracking/`,
the SAM2-memory arms) is at git tag **`archive/pre-cleanup-2026-10-21`**; docs older than that cite paths from there.

## The method (what produced the current results)

Frozen SAM2.1-b+ **image** predictor, per frame, no memory. Per frame t of an NFO window:

- **Tracker** (`adapter/tracker.py`): kernel-f16. A fixed-size box kernel matched on 1 − cos(F_t, F_B) (SAM2 stride-16
  features of frame t and of the empty scene B = warm-up median), Theil–Sen constant-velocity search window.
  Replaced the MOG2 blob tracker on 2026-10-16: it locates the person in 100% of synthetic frames vs 48%.
- **Composite** (`adapter/composite.py`): frames t−6..t−1, each shifted by the x-only Theil–Sen fit of the tracker
  track, change-gated against B (robust z-score, adaptive threshold max(3, q90)), recency-weighted (τ = 3.5):
  C = Σ w G A / Σ w G, evidence mass a = Σ w G. Ancestor: Airborne Optical Sectioning (arXiv 2111.06959, 2207.13344).
- **Adapter** (`adapter/fusion_adapter.py`, `Fusion`, 730k parameters): zero-initialised residuals at strides 16/8/4
  from (F_X, F_C, F_B, differences, a), plus a learned correction δ of the tracker point token. At step 0 it equals
  SAM2 with tracker prompts (arm `trk`).
- **Training** (`adapter/fusion_data.py`, `synth/`): synthetic GPJATK people behind procedural foliage
  (`synth/occluders.py`) on CanaTree100 / photo backgrounds, NFO-calibrated noise (`synth/noise.py`), regenerated
  online per clip. Loss **v3.4** (`fusion_adapter.loss_fn`): 20·focal_W(α 0.67, γ 2) + Focal Tversky (FP 0.3, FN 0.7, γ 0.75)
  with weights W (1 visible / background, 0 occluded within 8 px of a fragment, 0.3 far occluded). AdamW 3e-4,
  cosine, 32 targets per clip. **Checkpoint: step 200** (selected on seq1, never on seq2–4).
- **Inference** (`adapter/inference.py`, `fast_masks`): one encoder pass per frame shared by tracker, adapter and
  decoder: 61 ms/frame vs B0 53 ms (RTX PRO 4000 laptop). Frames with < 6 frames of history use the tracker prompts.

Things that mattered (each measured; details in the findings doc): the kernel-f16 tracker for composite alignment
(v3), the adaptive gate threshold (train/test parity), loss v3.4 over the balanced and simple variants, step 200,
and the composite itself (vs the `-noC` control, over 3 seeds: box AUC +0.024 ± 0.005, recognition conf
+0.025 ± 0.004, hand J +0.016 ± 0.013). Main result vs SAMURAI over 3 seeds on seq2–4 (recognition at N = 7; to be rescored at N = 5 on all sequences): box AUC 0.717 / 0.723 / 0.715
vs 0.646; hand J 0.761 / 0.763 / 0.757 vs 0.750; recognition AP50 0.926 / 0.915 / 0.926 vs 0.916.

Closed directions (do not reopen without new evidence): SAM2 memory variants (the oracle bound is +0.012 J), the
balanced loss, the one-pass feature-space adapter (designed, not pursued:
`docs/layer_sam2/archive/2026-10-20-one-pass-adapter-design.md`).

## Layout of `benchmark/nfo_vos/`

| path | purpose |
|---|---|
| `trials.py` | the test set: 61 admissible 40-frame windows over seq1–4 (`results/benchmark/rr/trials.json`, in git), GT boxes, native frames (`native`), hand labels (`hand_label`, `hand_frames`) |
| `baselines/run_sam2_video.py` (+ sbatch) | B0 (SAM2.1 video) and T1 (SAMURAI), same t0 box + point prompt |
| `adapter/` | the method: `tracker`, `composite`, `fusion_adapter` (model, loss, training `main`, `image_predictor`), `fusion_data` (synthetic batches), `dataset` (clip specs), `fusion_nfo` (NFO hand frames for checkpoint selection/test), `inference` (`fast_masks`), `fusion_adapter.sbatch` |
| `synth/` | synthetic training clips: `gpjatk_foliage`, `occluders` (vendored), `noise`, `photo_depth` (one-off: depth map of the foliage photo) |
| `eval/` | `dump_arms` (+ sbatch: masks of our arms), `box_metrics` (LaSOT one-pass AUC from masks), `recognition` (YOLO11m on GT-motion masked composites), `recog_vs_N`, `report` (the table + paired window bootstrap) |
| `tools/` | `remote_smoke` (cluster doctor: paths, GPU, one training step; run before long jobs), `timing`, `briefing_figures`, `cvat_io` (CVAT round trip for hand labels; the second annotator) |
| `tests/` | pytest suite; `parity_snapshot.py` = refactor guard (bit-exact before/after diff of the live paths) |

Arms (mask dirs under `results/benchmark/rr/masks/`): `b0-c2`, `t1-c2` (baselines, cluster rerun), `trk` (SAM2 +
tracker prompts), `trkbox` (the tracker box itself), `adapter-pf` / `adapter-s1` / `adapter-s2` (seeds 0/1/2),
`<arm>-noC` (composite removed at inference).

## Running

All commands from the repo root, `PYTHONPATH=.`; `$PY` = the benchmark env (see Environment).

```bash
$PY -m pytest benchmark/nfo_vos/tests -q                               # 29 tests, GPU ones skip without CUDA
$PY -m pytest benchmark/nfo_vos/tests/test_trials.py::test_each_frame_used_at_most_once   # one test
$PY -m benchmark.nfo_vos.tools.remote_smoke                            # before any cluster job
$PY -m benchmark.nfo_vos.adapter.fusion_adapter 1000 0                 # train (steps, seed); cluster: adapter/fusion_adapter.sbatch
$PY -m benchmark.nfo_vos.eval.dump_arms adapter-pf 0                   # masks of one arm, one window; cluster: eval/dump_arms.sbatch
$PY -m benchmark.nfo_vos.eval.box_metrics score b0-c2 t1-c2 trk adapter-pf ...   # -> results/benchmark/rr/downstream/box_gt.csv
$PY -m benchmark.nfo_vos.eval.recognition 5 uniform ARMS...             # see its docstring; -> recog_N5_*.csv (N = 5 since 2026-10-23)
$PY -m benchmark.nfo_vos.eval.report                                   # the table: seq1-4 (headline), seq1, seq2-4
$PY -m benchmark.nfo_vos.eval.mask_fill [score]                        # mask-size trend per clip (NFO_RUN=pass: clips to the end of the pass)
bash benchmark/nfo_vos/eval/pass_run.sh                                # cluster: the pass run's trials + mask arrays
TC_BATCH=1 $PY -m benchmark.nfo_vos.tools.timing ours|b0|t1
sbatch benchmark/nfo_vos/eval/rescore.sbatch                            # cluster: final table (box AUC + recognition for all final arms, one env, then report)
```

Refactor guard: `parity_snapshot save before.pkl`, change code, `save after.pkl`, `diff before.pkl after.pkl` must
print IDENTICAL (deterministic on one machine; measured 2026-10-21).

## Data manifest

Nothing large is in git. The cluster copy is the source of truth for everything marked *cluster*.

| path | what | in git? | how to get it |
|---|---|---|---|
| `results/benchmark/rr/trials.json` | the 61 windows | yes | `python -m benchmark.nfo_vos.trials` regenerates |
| `data/nfo_processed/seq*_gt/*_handmask_raw.png`, `*_handAv.png` | human visible-part labels (seq1 20 brush masks, seq2–4 23 frames) | yes (force-added: `data/` is ignored, so add new labels with `git add -f`) | irreplaceable |
| `data/nfo_final/nfo_final/seq{1..4}/*.jpg` | native 800×600 NFO frames (2.0 GB) | no | cluster |
| `data/nfo_processed/seq*_gt/` (rest) | `groundtruth.txt` (GT boxes), SAM2 pseudo-masks | no | cluster; masks regenerable with `gen_data/nfo_pseudo_masks` |
| `$NFO_GPJATK` (default `../master_thesis/GPJATK/Sequences`) | GPJATK person sequences, training (7.9 GB) | no | cluster |
| `data/CanaTree100/images/` | training backgrounds | no | cluster (public dataset) |
| `images/veronica111886-yakushima-2336735.jpg` | foliage template photo (Pixabay) | no | cluster |
| `data/synthetic/foliage_depth/veronica_disp.npy` | depth of that photo | no | cluster, or `synth/photo_depth.py` |
| `../samurai` @76ba195 + `sam2/checkpoints/sam2.1_hiera_base_plus.pt` | SAMURAI's sam2 fork + checkpoint | no | `requirements-benchmark.txt` header; cluster checkpoint `/storage/user/akid/checkpoints/` |
| `data/yolo11m.pt` | recognition detector | no | ultralytics downloads it on first use (assumed; or copy from cluster) |
| `results/benchmark/rr/layers_fusion_adapter/token_seed{0,1,2}/fusion_it200.pt` | the three adapter checkpoints | no | cluster (seed 2 trained there); poster tarball |
| `results/benchmark/rr/masks/<arm>/`, `downstream/*.csv`, `images/benchmark/downstream/` | masks, metric CSVs, figures | no | regenerable (commands above); poster tarball |

## Environment

- Laptop: `../samurai/.venv` (Python 3.12, torch 2.13 cu130); recognition (`ultralytics`) also runs there once
  installed from `requirements-benchmark.txt`.
- Cluster (SLURM): `/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3`, logs in `/storage/slurm/akid/logs/`.
  sbatch scripts use `--gres=gpu:1 --exclude=node1,node2,node6,node7,node22` (Pascal, 12 GB Titan, Blackwell sm 12.0
  without kernels) and set `AMP=fp16` automatically on pre-Ampere GPUs; training peaks at 9.8 GB (32 targets), so any GPU with ≥ 12 GB works.
- **Environment drift**: the same checkpoint gives masks differing by ~5.5% of pixels between laptop and cluster.
  Compare arms only within one environment and score recognition for all arms in one run.

## Rules for this benchmark

- seq2–4 is the single held-out test. Select checkpoints and tune on seq1 / synthetic data only.
- Synthetic val J is a sanity check, not a selection metric (it ranked the losses opposite to real data).
- Report means over 3 seeds and paired window-bootstrap CIs (`eval/report.py`).

## U-Net code (legacy, unchanged)

```bash
python3 -m train_main --config <config_name> --save-dir <dir>
python3 -m test_main --config <config_name> --load-dir <dir>
```

`requirements.txt` (PyTorch 1.8.1 era). `<config_name>` is a variable in `config/train_config.py` /
`config/test_config.py`, resolved with `eval()`; training pickles the resolved config (`train_cfg.pkl`) next to the
checkpoint and `test_main.py` applies overrides on top. Datasets (`dataset/`) walk `*_gt` folders with `_or`, `_gauss` /
`_circle` frames and `groundtruth.txt`; `seq_size` must be odd. Generators: `gen_data/gen_{noise,mnist,kth}_data`.
`set_cfg` / `load_cfg` use `eval()` and unpickle: trusted inputs only.
