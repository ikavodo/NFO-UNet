# Diffusion-VAS: how to rebuild and run it (amodal baseline)

Diffusion-VAS (Chen, Ramanan & Khurana, CVPR 2025, arXiv 2412.04623) is an amodal video segmentation model: a Stable
Video Diffusion model that completes an object's mask behind occluders. We ran its amodal-segmentation stage once,
2026-10-14, on NFO seq1 (`nfo_failure_log.md` line "Diffusion-VAS ... on SAM2-ceiling": it adds pixels the annotators do
not count as visible, P 0.89 → 0.64). It is kept as a candidate **amodal baseline** for the paper; the laptop copy
(env `dvas`, 7.8 GB, and weights in `~/models/diffusion-vas`, 7.1 GB) was deleted on 2026-11-10. Rebuild it on the
machine that will run it (a GPU with ≥ 24 GB is assumed; not measured).

## What is pinned here

| file | what | use |
|---|---|---|
| `environment.lock.yml` | conda export of the env that ran it (Python 3.12.13, torch 2.12.1+cu130, diffusers 0.29.1, transformers 4.44.2) | `conda env create -f environment.lock.yml` |
| `requirements.lock.txt` | `pip freeze` of the same env (CUDA 13.0 torch index on line 1) | fallback: `pip install -r requirements.lock.txt` in a fresh Python 3.12 env |
| `conda.explicit.txt` | exact conda package URLs (linux-64) | bit-identical conda part: `conda create -n dvas --file conda.explicit.txt` |

The env was a clone of our `occ` env plus `diffusers==0.29.1 transformers==4.44.2`. Upstream instead asks for Python 3.10
and an unpinned `requirements.txt` (torch, diffusers 0.29.1, transformers, accelerate, ...); either works for inference
as far as we know (only the pinned one was run). torch `+cu130` needs a CUDA 13 driver: on an older driver install the
matching torch build from https://pytorch.org/get-started/locally/ instead and keep the other pins.

## Rebuild

```bash
# 1. code (upstream, MIT) at the commit we used, next to this repo
git clone https://github.com/Kaihua-Chen/diffusion-vas ../diffusion-vas && git -C ../diffusion-vas checkout bbea113
# 2. env
conda env create -f benchmark/external/diffusion_vas/environment.lock.yml      # -> env "dvas"
# 3. weights (Hugging Face; git-lfs needed). Only the amodal stage was used; the content-completion one is optional.
mkdir -p ~/models/diffusion-vas && cd ~/models/diffusion-vas
git clone https://huggingface.co/kaihuac/diffusion-vas-amodal-segmentation
# git clone https://huggingface.co/kaihuac/diffusion-vas-content-completion
# 4. depth: Depth Anything V2-L (HF id depth-anything/Depth-Anything-V2-Large-hf) downloads on first use
```

Check: `~/miniconda3/envs/dvas/bin/python -c "import torch, diffusers; print(torch.__version__, torch.cuda.is_available(), diffusers.__version__)"`.

## Run

Our runner was removed in the 2026-10-21 cleanup; it is at the archive tag:

```bash
git show archive/pre-cleanup-2026-10-21:benchmark/nfo_vos/amodal_dvas.py > benchmark/nfo_vos/amodal_dvas.py
PYTHONPATH=. ~/miniconda3/envs/dvas/bin/python -m benchmark.nfo_vos.amodal_dvas     # DVAS_SEQ / DVAS_TAG / DVAS_SEEDS env vars
```

It expects `../diffusion-vas` (code) and `~/models/diffusion-vas/diffusion-vas-amodal-segmentation` (weights); its
docstring lists its deviations from upstream `demo.py` (25-frame clips at stride 3, union with the modal mask at native
resolution, grayscale replicated to RGB). It imports `benchmark.nfo_vos.velocity`, which the cleanup also removed:
restore that from the same tag, or port the runner to the current code before a new run.
