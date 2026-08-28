#!/bin/bash

#SBATCH --job-name=kth_train_anisotropic
#SBATCH --nodes=1
# 8, not 4: kth_train/kth_train_anisotropic both set num_workers=min(8, available_cpus()) -
# 4 cpus-per-task left half the dataloader's own worker budget unused
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node4,node11,node14,node15,node16,node17,node19,node20
# --nodelist above is copied from integ.sbatch as a starting point - it does not target
# specific GPU types/VRAM. If this cluster supports a type constraint (common SLURM
# convention, verify with `sinfo`/`scontrol show node`), prefer that instead, e.g.:
# #SBATCH --gres=gpu:a100:1
# and drop --nodelist entirely once you know which nodes actually have A100s or 24GB+ cards.
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/akid/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/akid/logs/slurm-%j.out

# =============================================
# Environment Setup
# =============================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# `conda activate` inside a SLURM batch shell (non-interactive) fails silently-ish:
# most ~/.bashrc files early-return for non-interactive shells before reaching the
# conda-init block that defines the `conda` function, so `source ~/.bashrc; conda
# activate ...` errors out but doesn't halt the script - it then silently falls through
# to the base env's python3 (no torch installed). Call the env's interpreter directly
# instead, bypassing conda activate entirely.
PYTHON=/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3

# =============================================
# Main Execution
# =============================================
pwd; hostname; date
nvidia-smi
srun "$PYTHON" -m train_main --config kth_train_anisotropic \
    --save-dir out/kth_train_anisotropic_$(date +%Y%m%d_%H%M%S)
