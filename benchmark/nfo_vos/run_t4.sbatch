#!/bin/bash
# T4 family on the pilot trials, one array task per (trial, variant). Trials are independent:
# each has its own warm-up, its own tracker run and its own causal buffer, and writes its own
# npz files (results/benchmark/$NFO_RUN/masks/<method>/<trial>.npz), so tasks share no state.
#
#   N = number of admissible trials in results/benchmark/$NFO_RUN/trials.json (rr seq1 pilot: 15)
#   sbatch --export=ALL --array=0-$((2*N-1)) benchmark/nfo_vos/run_t4.sbatch
#
# Task map: id 0..N-1 -> --variant point, trial id; id N..2N-1 -> --variant box, trial id-N.
# Needs on the cluster: this repo with data/nfo_processed/<seq>_gt (224 frames + _sammask.png), results/benchmark/$NFO_RUN/trials.json, and the base_plus checkpoint at NFO_SAM2_CKPT.
# T4 never reads the native frames.

#SBATCH --job-name=nfo_t4
#SBATCH --array=0-121         # = 2 x N_TRIALS - 1; full rr run: 61 trials (seq1 pilot alone: 0-29)
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
# number of admissible trials of this run (array ids: first N -> first method/variant, next N -> second)
N_TRIALS=$($PYTHON -c "import json; print(sum(t['admissible'] for t in json.load(open('results/benchmark/$NFO_RUN/trials.json'))))")
if [ "$SLURM_ARRAY_TASK_ID" -lt "$N_TRIALS" ]; then
  VARIANT=point; IDX=$SLURM_ARRAY_TASK_ID
else
  VARIANT=box; IDX=$((SLURM_ARRAY_TASK_ID - N_TRIALS))
fi

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "variant=$VARIANT trial-index=$IDX ckpt=$NFO_SAM2_CKPT"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENCV_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PYTHON" -m benchmark.nfo_vos.run_t4 --variant "$VARIANT" --trial-index "$IDX" --force
date
