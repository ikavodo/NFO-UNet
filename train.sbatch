#!/bin/bash

#SBATCH --job-name=nfo_unet_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node3,node4,node11,node13,node14,node15,node16,node17,node19,node20
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

source ~/.bashrc
conda activate comp_vis

# =============================================
# Main Execution
# =============================================
pwd; hostname; date
nvidia-smi
srun python3 -m train_main --config kth_train --save-dir out/kth_train_$(date +%Y%m%d_%H%M%S)
