#!/bin/bash
# G1 baselines (plan_feature_layers_training.md step 3) under one nuisance arm per array task:
#   task 0 = flutter (non-rigid foliage flutter, calibrated to NFO seq2), task 1 = chunks (perlin clump 12 ∪ 24).
# Each writes results/benchmark/rr/layers_g1_<arm>/{clip_*.npz,g1_table.txt,g1_point.json} and
# images/benchmark/gpjatk_foliage/g1_smaps_<arm>.png. Resumable: finished clips are skipped.
#   sbatch benchmark/nfo_vos/layers/g1_baselines.sbatch          (or ARMS="base flutter chunks" with --array=0-2)
# Needs on the cluster: ../master_thesis/GPJATK/Sequences and ../master_thesis/src/occluders.py,
# images/veronica111886-yakushima-2336735.jpg (foliage photo, not in git), the base_plus checkpoint at NFO_SAM2_CKPT.
# ≈ 9 s per clip on an RTX-class GPU (local), 120 clips → ≈ 20 min + ≈ 3 min bootstrap.

#SBATCH --job-name=nfo_g1
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_g1-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_g1-%A_%a.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
ARMS=(${ARMS:-flutter chunks})
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "g1 baselines arm=$ARM ckpt=$NFO_SAM2_CKPT"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENCV_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=.

srun "$PYTHON" -m benchmark.nfo_vos.layers.g1_baselines --arm "$ARM"
date
