#!/bin/bash
# B0 (SAM2.1 video) and T1 (SAMURAI) on the pilot trials, one array task per (method, trial).
# Trials are independent and each writes its own npz
# (results/benchmark/$NFO_RUN/masks/<method>[-cp-<tag>]/<trial>.npz), so tasks share no state.
#
#   sbatch benchmark/nfo_vos/run_sam2_video.sbatch                         # 42 tasks, raw prompt
#   PREPEND_DIR=results/benchmark/$NFO_RUN/composites_hybrid \
#     sbatch --export=ALL benchmark/nfo_vos/run_sam2_video.sbatch          # composite-prepend
#   sbatch --array=21-41 ...                                               # T1 only
#
# Task map: id 0-20 -> B0, trial id; id 21-41 -> T1, trial id-21.
# Needs on the cluster: this repo (branch vos-benchmark-pilot), results/benchmark/$NFO_RUN/trials.json
# and the composites dir (both committed), the NATIVE frames data/nfo_final/nfo_final/seq1/
# (only frames ~1140-1625 are read by the pilot), the base_plus checkpoint at NFO_SAM2_CKPT, a
# sam2 1.x env for B0 (PYTHON_B0) and the SAMURAI repo's sam2 fork for T1 (PYTHON_T1).

#SBATCH --job-name=nfo_sam2v
#SBATCH --array=0-29          # = 2 x N_TRIALS - 1; override with --array (rr seq1 pilot: 15 trials)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_sam2v-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_sam2v-%A_%a.out

PYTHON_B0=${PYTHON_B0:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
PYTHON_T1=${PYTHON_T1:-$HOME/samurai/.venv/bin/python}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-$HOME/checkpoints/sam2.1_hiera_base_plus.pt}

export NFO_RUN=${NFO_RUN:-rr}
# number of admissible trials of this run (array ids: first N -> first method/variant, next N -> second)
N_TRIALS=$($PYTHON_B0 -c "import json; print(sum(t['admissible'] for t in json.load(open('results/benchmark/$NFO_RUN/trials.json'))))")
if [ "$SLURM_ARRAY_TASK_ID" -lt "$N_TRIALS" ]; then
  METHOD=b0; PY=$PYTHON_B0; IDX=$SLURM_ARRAY_TASK_ID
else
  METHOD=t1; PY=$PYTHON_T1; IDX=$((SLURM_ARRAY_TASK_ID - N_TRIALS))
fi
PREPEND=${PREPEND_DIR:+--prepend-dir $PREPEND_DIR}

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "method=$METHOD trial-index=$IDX prepend=${PREPEND_DIR:-none} python=$PY"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PY" -m benchmark.nfo_vos.run_sam2_video --method "$METHOD" --trial-index "$IDX" $PREPEND --force
date
