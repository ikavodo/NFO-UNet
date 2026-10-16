#!/bin/bash
# Approach A (docs/layer_sam2/training/plan_fusion_prompt_adapter.md §11–12): residual fusion adapter on frozen SAM2.
# One array task per sparse-prompt mode: 0 = token (default), 1 = box (ablation). Override MODES / STEPS / SEED.
#   sbatch benchmark/nfo_vos/layers/fusion_adapter.sbatch
#   MODES="token" STEPS=1500 SEED=1 sbatch --export=ALL --array=0 benchmark/nfo_vos/layers/fusion_adapter.sbatch
# Writes results/benchmark/rr/layers_fusion_adapter/<mode>_seed<seed>/{log.json,fusion_it*.pt} (≈ 3 MB per checkpoint)
# and images/benchmark/gpjatk_foliage/fusion_nfo_featdiff.png. Nothing else is written: clips are regenerated online.
# Needs on the cluster (none of it is in git):
#   ../master_thesis/GPJATK/Sequences            GPJATK source frames + silhouettes (synth/gpjatk_foliage.GP)
#   ../master_thesis/src/occluders.py            occ_perlin (imported by synth/gpjatk_foliage)
#   images/veronica111886-yakushima-2336735.jpg  foliage template photo (gpjatk_foliage.TEMPLATE)
#   data/nfo_final/nfo_final/seq{2,3,4}/         NFO native frames (real-frame validation, fusion_nfo)
#   data/nfo_processed/seq{2,3,4}_gt/            *_or.jpg (blob tracker), *_handAv.png, groundtruth.txt
#   the base_plus checkpoint at NFO_SAM2_CKPT
# GPU: bf16 autocast → Ampere or newer (A100 / A40 / RTX 30xx+). On Turing (Quadro RTX 6000) set AMP=fp16.
# Measured locally (RTX PRO 4000 Blackwell laptop): ≈ 3.1 s per step (2 clips ≈ 24 frames), 1500 steps ≈ 80 min.

#SBATCH --job-name=nfo_fusion
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_fusion-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_fusion-%A_%a.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
MODES=(${MODES:-token box})
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "fusion adapter mode=$MODE steps=${STEPS:-1500} seed=${SEED:-0} amp=${AMP:-bf16} ckpt=$NFO_SAM2_CKPT"
export OMP_NUM_THREADS=2 PYTHONPATH=.
srun "$PYTHON" -m benchmark.nfo_vos.layers.fusion_adapter "$MODE" "${STEPS:-1500}" "${SEED:-0}"
date
