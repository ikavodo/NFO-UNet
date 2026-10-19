#!/bin/bash
# Cluster smoke test before adapter training (benchmark/nfo_vos/layers/remote_smoke.py), ≈ 3 min. Same env and GPU
# request as fusion_adapter.sbatch. Output ends with "SMOKE OK" or the list of failed checks.
#   sbatch benchmark/nfo_vos/layers/remote_smoke.sbatch            (LOSS=balanced sbatch --export=ALL ... to test a loss)
#SBATCH --job-name=nfo_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --exclude=node2,node22
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_smoke-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_smoke-%j.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export OMP_NUM_THREADS=2 PYTHONPATH=. TC_BATCH=1
hostname
srun "$PYTHON" -m benchmark.nfo_vos.layers.remote_smoke
