#!/bin/bash
# Findings §10.x 26/27: train one seed of the light tied feature integral (+ amodal aux), the recipe frozen on 2026-11-26.
# Seed = array task id; submitted by adapter/feat_tie_seeds.sh. Writes layers_fusion_adapter/feat_tie_seed<s>/fusion_it200.pt
# and results/benchmark/rr/feat_integral/tie/train_cl_s<s>.log. STOP=200: returns before any test-set evaluation.
#SBATCH --job-name=nfo_tie_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_tie_train-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_tie_train-%A_%a.out
set -o pipefail
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export AMP=$("$PYTHON" -c "import torch; print('fp16' if torch.cuda.get_device_capability() < (8, 0) else 'bf16')")
export OMP_NUM_THREADS=2 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$SLURM_ARRAY_TASK_ID; A=results/benchmark/rr/layers_fusion_adapter; D=results/benchmark/rr/feat_integral/tie; mkdir -p $D
echo "$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) AMP=$AMP seed=$S commit $(git rev-parse --short HEAD) $(date)"
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 15000 ] || { echo "GPU has ${mem} MiB; FEAT training peaks ≈ 14 GB"; exit 1; }
AMODAL=1 FEAT=1 TIE=1 AGG=none LAM=1 RUN_DIR=$A/feat_tie_seed$S STOP=200 srun "$PYTHON" -u -m benchmark.nfo_vos.adapter.fusion_adapter 1000 $S 2>&1 \
  | grep --line-buffered -E "^(VAL|SEL|GAP|AMGAP|AMGAPC|TIE|it [0-9]+ (loss|modal)|params)|Error|Traceback" | tee $D/train_cl_s$S.log
[ -f $A/feat_tie_seed$S/fusion_it200.pt ] || { echo "no it200 checkpoint"; exit 1; }   # failed run → dependent dumps never start
date
