#!/bin/bash
# Headroom checks on new datasets (tools/headroom.py), one job per (dataset, method):
#   sbatch --export=ALL,WHAT=cdnet,METHOD=b0 benchmark/nfo_vos/tools/headroom.sbatch
#   sbatch --export=ALL,WHAT=mvot,METHOD=t1  benchmark/nfo_vos/tools/headroom.sbatch
# then (CPU, any node): PYTHONPATH=. $PYTHON -m benchmark.nfo_vos.tools.headroom mvot-report
# Data on the cluster: /storage/user/akid/datasets/{CDnet_2014/dataset,MVOT} (override CDNET_ROOT / MVOT_ROOT).
# B0 = official sam2 package (comp_vis env); T1 = SAMURAI's sam2 fork put first on the path (as run_sam2_video.sbatch).
#SBATCH --job-name=nfo_headroom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_headroom-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_headroom-%j.out
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
SAMURAI_SAM2=${SAMURAI_SAM2:-/storage/user/akid/git_projects/samurai/sam2}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export CDNET_ROOT=${CDNET_ROOT:-/storage/user/akid/datasets/CDnet_2014/dataset}
export MVOT_ROOT=${MVOT_ROOT:-/storage/user/akid/datasets/MVOT}
WHAT=${WHAT:-cdnet}; METHOD=${METHOD:-b0}
if [ "$METHOD" = t1 ]; then export PYTHONPATH=.:$SAMURAI_SAM2; else export PYTHONPATH=.; fi
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date
echo "what=$WHAT method=$METHOD"
"$PYTHON" -m benchmark.nfo_vos.tools.headroom "$WHAT" "$METHOD"
date
