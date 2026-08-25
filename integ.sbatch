#!/bin/bash

#SBATCH --job-name=motion_compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node3,node4,node11,node13,node14,node15,node16,node17,node19,node20
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/akid/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/akid/logs/slurm-%j.out

# =============================================
# Environment Setup
# =============================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

# Load environment
# module load cuda/12.1
source ~/.bashrc
conda activate comp_vis

# =============================================
# Main Execution
# =============================================
pwd; hostname; date
nvidia-smi
srun python train.py
