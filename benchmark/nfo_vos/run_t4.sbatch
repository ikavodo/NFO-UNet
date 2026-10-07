#!/bin/bash
# T4-1n (tracker-prompted per-frame SAM2, no memory) on the run's trials, one array task per trial.
# Trials are independent; each writes results/benchmark/$NFO_RUN/masks/t4-1n/<trial>.npz.
#
#   N = number of admissible trials in results/benchmark/$NFO_RUN/trials.json (rr seq1 pilot: 15)
#   sbatch benchmark/nfo_vos/run_t4.sbatch        # task id = trial index (0..N-1)
# Needs on the cluster: this repo with data/nfo_processed/<seq>_gt (224 frames + _sammask.png), results/benchmark/$NFO_RUN/trials.json, and the base_plus checkpoint at NFO_SAM2_CKPT.
# T4 never reads the native frames.

#SBATCH --job-name=nfo_t4
#SBATCH --array=0-60          # = N_TRIALS - 1 (full rr run: 61 trials)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_t4-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_t4-%A_%a.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}

export NFO_RUN=${NFO_RUN:-rr}
IDX=$SLURM_ARRAY_TASK_ID

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "t4-1n trial-index=$IDX ckpt=$NFO_SAM2_CKPT"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENCV_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PYTHON" -m benchmark.nfo_vos.run_t4 --trial-index "$IDX" --force
date
