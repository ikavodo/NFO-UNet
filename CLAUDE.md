# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Implementation of a spatio-temporal U-Net for heatmap-based person detection under fragmented occlusion,
accompanying the paper "Person localization under fragmented occlusion". The network takes a sequence of
grayscale frames stacked as channels and predicts a heatmap for the frame at the sequence center.

## Setup

```bash
python3 -m venv .venv
.venv/bin/activate
pip install -r requirements.txt
```

Also install PyTorch separately for your platform (project was built against PyTorch 1.8.1) — see
https://pytorch.org/.

There is no test suite, linter, or CI config in this repo.

## Running training / testing

```bash
python3 -m train_main.py --config <config_name> --save-dir <dir>
python3 -m test_main.py --config <config_name> --load-dir <dir_with_stored_model>
```

`<config_name>` refers to a variable name inside `config/train_config.py` / `config/test_config.py` that is
resolved with `eval()` (see Config system below). If `--config` is omitted, the base config is used, which
has most fields set to `None` and will fail at runtime unless filled in.

## Generating data

Three synthetic datasets can be generated via `gen_data/gen_noise_data`, `gen_data/gen_mnist_data`, and
`gen_data/gen_kth_data`. Each has a `*_config.py` to edit and a `main.py` to run. See README.md for what
each dataset is designed to test (temporal-only cues, spatio-temporal cues under occlusion, and real
person-detection footage from the KTH action dataset, respectively).

## Architecture

**Config system** (`config/config.py`, `config/train_config.py`, `config/test_config.py`): configs are
`AbstractConfig` instances built from a dict, holding a mutable module-level `config` singleton. Named
configs are defined as plain dicts/objects in the same file and picked up via `set_cfg(name)`, which does
`eval(name)` against the module namespace — so `--config` values must correspond to real Python identifiers
defined in `train_config.py`/`test_config.py`. Training persists the resolved config with `dill` next to the
checkpoint (`train_cfg.pkl`); `test_main.py` loads that pickle first, then applies `--config` overrides on
top, so a test config only needs to override what differs from training.

**Datasets** (`dataset/`): `AbstractDataSet` (in `abstract_dataset.py`) walks a root directory for
subfolders ending in `_gt`, and inside each expects per-frame files tagged `_or` (original), `_gauss` /
`_circle` (heatmap variants), and a `groundtruth.txt` of bounding boxes. Concrete subclasses
(`KthDataSet`, `MnistDataSet`, `TestingDataSet`) only need to implement `_construct_ds_entries`, which
builds `DatasetEntry` tuples and an `idx_mapping` (a dataset may contain sequence boundaries that must be
excluded from indexing, e.g. edges of a video root). `__getitem__` reads `seq_length` frames centered on
the target index (`margin = seq_length // 2`, must be odd), stacks them as channels, and passes them through
a `Transform` pipeline (`utils/transform_utils.py`, composed via `chain(...)`) before batching.
`config.hm_filter` (a `HeatMap` enum value) selects gauss vs. circle heatmap channel at train/eval time.

**Network** (`network/unet.py`, `network/unet_parts.py`): standard U-Net; `n_channels` is the sequence
length (each frame is one input channel), `n_classes=1` (single heatmap output). Checkpoints are the raw
`state_dict` saved via `torch.save(..., _use_new_zipfile_serialization=False)` for cross-version
compatibility, loaded by `UNet.load_checkpoint`.

**Evaluation** (`eval/`): `AbstractEval` thresholds/post-processes a predicted heatmap, extracts predicted
center points (`extract_centers`, implemented by subclasses `MaxEval` / `ThresholdEval`), and matches them
against ground-truth bounding boxes by squared distance (`max_dist_error`) to compute TP/FP/FN → F1, used
by `test_main.py`.

**Training loop** (`train_main.py`): standard loop with `EarlyStopping` (`early_stopping.py`) driving both
early termination and checkpoint saving (only saved on improvement). `logistic_loss.py` provides the
loss criterion referenced from configs.

## Gotchas

- `set_cfg`/`load_cfg` use `eval()` and unpickle arbitrary config objects — only pass trusted config
  names/files.
- `seq_size` (sequence length) must be odd; `AbstractDataSet.__init__` asserts this.
- Frame indexing in `__getitem__` uses `idx + i * nth_frame` directly against `self.entries`, so
  `idx_mapping` in each dataset subclass must already exclude any index too close to a sequence boundary.
