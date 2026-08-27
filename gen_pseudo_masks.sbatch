#!/bin/bash

#SBATCH --job-name=nfo_pseudo_masks
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node4,node11,node14,node15,node16,node17,node19,node20
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/slurm/akid/logs/pseudo_masks-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/pseudo_masks-%A_%a.out

# One SLURM array task per sequence (4 tasks, all 8 segments processed serially within each) -
# per-segment (32-task) granularity turned out to be overkill for this workload and only added
# staging-directory collision risk (see gen_nfo_pseudo_masks.py's --segment-idx comment) for no
# real speedup, since the cluster doesn't have 32 free GPUs to schedule onto simultaneously
# anyway. Kept: one model instance per task (avoids the earlier OOM from multiple sam2.1-hiera-
# large instances sharing one 12GB card), and array scheduling so SLURM still spreads the 4
# sequences across whatever GPUs are free.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3

SEQS=(seq1 seq2 seq3 seq4)
SEQ=${SEQS[$SLURM_ARRAY_TASK_ID]}

pwd; hostname; date
nvidia-smi
srun "$PYTHON" -m gen_data.nfo_pseudo_masks.gen_nfo_pseudo_masks --seq "$SEQ" \
    --combine-method union_gt_outlier
