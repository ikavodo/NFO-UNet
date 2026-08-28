#!/bin/bash
# Runs eval_nfo.py's three configs as three PARALLEL array tasks instead of one ~11-minute
# serial run. Each task is one config, ~3.5 min of pure CPU work.
#
#   sbatch tracking/eval/eval_nfo.sbatch
#
# Worth knowing before you bother: 11 minutes serial is not a lot. The reason to use this is
# convenience (fire and forget, logs per config) or a busy laptop, not a big speedup.

#SBATCH --job-name=nfo_eval
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_eval-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_eval-%A_%a.out

# No --gres=gpu: this pipeline is OpenCV/numpy only, there is nothing for a GPU to do.
# No --nodelist either, for the same reason - any CPU node will do, and pinning to the
# GPU nodelist copied around the other sbatch files would just queue behind GPU jobs.

# `conda activate` in a non-interactive batch shell usually fails silently (most ~/.bashrc
# files early-return before the conda-init block), then falls through to the base python.
# Call the env's interpreter directly - same reasoning as train.sbatch.
PYTHON=/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3

CONFIGS=(noshape fixed relative)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

pwd; hostname; date
echo "config: $CONFIG"

# OpenCV/BLAS spawn threads per core; without this they oversubscribe against
# --cpus-per-task and get slower, not faster.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENCV_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PYTHON" -m tracking.eval.eval_nfo "$CONFIG"
date
