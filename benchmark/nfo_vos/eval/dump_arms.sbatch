#!/bin/bash
# Downstream benchmark, Task 3 (docs/vos_benchmark/plan_downstream_benchmark.md): masks of the missing arms, one window
# per array task (61 windows, trials.json index 0..60). One array per arm (arms: eval/dump_arms.py docstring):
#   for A in trkbox trk adapter-pf adapter-s1 adapter-s2 adapter-pf-noC adapter-s1-noC adapter-s2-noC; do ARM=$A sbatch --export=ALL benchmark/nfo_vos/eval/dump_arms.sbatch; done
# Writes results/benchmark/rr/masks/<arm>/<trial>.npz (≈ 40 KB each). Needs on the cluster (not in git):
#   results/benchmark/rr/trials.json
#   results/benchmark/rr/layers_fusion_adapter/token_seed{0,1,2}/fusion_it200.pt     (adapter arms)
#   data/nfo_final/nfo_final/seq{1..4}/   data/nfo_processed/seq{1..4}_gt/
#   the base_plus checkpoint at NFO_SAM2_CKPT
#SBATCH --job-name=nfo_dump
#SBATCH --array=0-60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# GPU: any free node (training peaks within 16 GB; it ran on a 16 GB laptop GPU). Excluded: Pascal p6000 (node1, 7),
# 12 GB titan (node2, 6), Blackwell node22 (sm 12.0: no kernels in cu126 torch). Turing (rtx_8000/6000/5000) -> AMP=fp16 below.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_dump-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_dump-%A_%a.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export OMP_NUM_THREADS=2 PYTHONPATH=. TC_BATCH=1
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; echo "arm=$ARM idx=$SLURM_ARRAY_TASK_ID"
srun "$PYTHON" -m benchmark.nfo_vos.eval.dump_arms "$ARM" "$SLURM_ARRAY_TASK_ID"
