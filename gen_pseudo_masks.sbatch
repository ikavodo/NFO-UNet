#!/bin/bash

#SBATCH --job-name=nfo_pseudo_masks
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node4,node11,node14,node15,node16,node17,node19,node20
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/slurm/akid/logs/pseudo_masks-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/pseudo_masks-%A_%a.out

# One SLURM array task per (sequence, segment) pair - 4 sequences x 8 segments each = 32 tasks,
# each independently schedulable onto whatever node/GPU is free. This is the fix for the earlier
# OOM: 4 concurrent sam2.1-hiera-large instances on one 12GB card didn't fit (~3.1-3.7GB each);
# letting SLURM spread 32 single-segment jobs across the cluster's actual GPU count is both
# faster (real parallelism, not fighting over one card) and immune to that OOM entirely (each
# task loads exactly one model instance).

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3

SEQS=(seq1 seq2 seq3 seq4)
N_SEGMENTS=8  # verified: all 4 sequences have exactly 8 continuously-visible segments

seq_idx=$((SLURM_ARRAY_TASK_ID / N_SEGMENTS))
seg_idx=$((SLURM_ARRAY_TASK_ID % N_SEGMENTS))
SEQ=${SEQS[$seq_idx]}

pwd; hostname; date
nvidia-smi
srun "$PYTHON" -m gen_data.gen_nfo_pseudo_masks --seq "$SEQ" --segment-idx "$seg_idx" \
    --combine-method union_gt_outlier
