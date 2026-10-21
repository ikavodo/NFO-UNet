#!/bin/bash
# Cluster smoke test before adapter training (benchmark/nfo_vos/tools/remote_smoke.py), ≈ 3 min. Same env and GPU
# request as fusion_adapter.sbatch. Output ends with "SMOKE OK" or the list of failed checks.
#   sbatch benchmark/nfo_vos/tools/remote_smoke.sbatch         
#SBATCH --job-name=nfo_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# GPU: any free node with >= 12 GB (training peaks at 9.8 GB, 32 targets, measured 2026-10-21). Excluded: Pascal p6000 (node1, 7),
# 12 GB titan (node2, 6), Blackwell node22 (sm 12.0: no kernels in cu126 torch). Turing (rtx_8000/6000/5000) -> AMP=fp16 below.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_smoke-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_smoke-%j.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export AMP=$("$PYTHON" -c "import torch; print('fp16' if torch.cuda.get_device_capability() < (8, 0) else 'bf16')")
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) AMP=$AMP"
export OMP_NUM_THREADS=2 PYTHONPATH=. TC_BATCH=1
hostname
srun "$PYTHON" -m benchmark.nfo_vos.tools.remote_smoke
