#!/bin/bash
# Approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11–12): residual fusion adapter on frozen SAM2.
# One training run; override STEPS / SEED (the reported seeds: STEPS=1000 SEED=0|1|2, checkpoint step 200).
#   STEPS=1000 SEED=1 sbatch --export=ALL benchmark/nfo_vos/adapter/fusion_adapter.sbatch
# Writes results/benchmark/rr/layers_fusion_adapter/token_seed<seed>/{log.json,fusion_it*.pt} (≈ 3 MB per checkpoint)
# Nothing else is written: clips are regenerated online.
# Needs on the cluster (none of it is in git):
#   $NFO_GPJATK (default ../master_thesis/GPJATK/Sequences)  GPJATK source frames + silhouettes (synth/gpjatk_foliage.GP)
#   images/veronica111886-yakushima-2336735.jpg  foliage template photo (gpjatk_foliage.TEMPLATE)
#   data/nfo_final/nfo_final/seq{2,3,4}/         NFO native frames (real-frame validation, fusion_nfo)
#   data/nfo_processed/seq{1..4}_gt/            groundtruth.txt (+ hand labels, in git)
#   the base_plus checkpoint at NFO_SAM2_CKPT
# GPU: bf16 autocast → Ampere or newer (A100 / A40 / RTX 30xx+). On Turing (Quadro RTX 6000) set AMP=fp16.
# Measured locally (RTX PRO 4000 Blackwell laptop): ≈ 3.1 s per step (2 clips ≈ 24 frames), 1500 steps ≈ 80 min.

#SBATCH --job-name=nfo_fusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# GPU: any free node with >= 12 GB (training peaks at 9.8 GB, 32 targets, measured 2026-10-21). Excluded: Pascal p6000 (node1, 7),
# 12 GB titan (node2, 6), Blackwell node22 (sm 12.0: no kernels in cu126 torch). Turing (rtx_8000/6000/5000) -> AMP=fp16 below.
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_fusion-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_fusion-%j.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export AMP=$("$PYTHON" -c "import torch; print('fp16' if torch.cuda.get_device_capability() < (8, 0) else 'bf16')")
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) AMP=$AMP"

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "fusion adapter steps=${STEPS:-1500} seed=${SEED:-0} amp=${AMP:-bf16} ckpt=$NFO_SAM2_CKPT"
export OMP_NUM_THREADS=2 PYTHONPATH=.
# TEST=0: no seq2-4 test at the end (single held-out test is run separately on the selected checkpoint)
srun "$PYTHON" -m benchmark.nfo_vos.adapter.fusion_adapter "${STEPS:-1500}" "${SEED:-0}" 100 32 12 "${TEST:-0}"
date
