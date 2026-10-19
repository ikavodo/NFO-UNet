#!/bin/bash
# Downstream benchmark, Task 3 (docs/vos_benchmark/plan_downstream_benchmark.md): masks of the missing arms, one window
# per array task (61 windows, trials.json index 0..60). One array per arm:
#   for A in vid-plain trkbox trk adapter-pf vid-prompts vid-adapter-pre vid-adapter-noC; do ARM=$A sbatch --export=ALL benchmark/nfo_vos/downstream/dump_arms.sbatch; done
# SC first (environment parity): ARM=vid-plain sbatch --array=0 ...; masks/b0 was made on this cluster in fp16, so
#   NFO_VID_FP16=1 vid-plain must reproduce masks/b0/seq1_s0_t11.npz to 0 px. Locally it differs by 1997 px (0.66%,
#   deterministic run-to-run), i.e. environment drift. If 0 px on the cluster, B0/T1 need no rerun.
# Writes results/benchmark/rr/masks/<arm>/<trial>.npz (≈ 40 KB each). Needs on the cluster (not in git):
#   results/benchmark/rr/trials.json
#   results/benchmark/rr/layers_fusion_adapter/token_seed0/fusion_it200.pt     (adapter arms)
#   data/nfo_final/nfo_final/seq{1..4}/   data/nfo_processed/seq{1..4}_gt/
#   the base_plus checkpoint at NFO_SAM2_CKPT
#SBATCH --job-name=nfo_dump
#SBATCH --array=0-60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_dump-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_dump-%A_%a.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export OMP_NUM_THREADS=2 PYTHONPATH=. TC_BATCH=1
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; echo "arm=$ARM idx=$SLURM_ARRAY_TASK_ID"
srun "$PYTHON" -m benchmark.nfo_vos.downstream.dump_arms "$ARM" "$SLURM_ARRAY_TASK_ID"
