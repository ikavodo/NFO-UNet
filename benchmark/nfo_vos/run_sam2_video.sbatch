#!/bin/bash
# B0 (SAM2.1 video) and T1 (SAMURAI) on the pilot trials, one array task per (method, trial).
# Trials are independent and each writes its own npz
# (results/benchmark/$NFO_RUN/masks/<method>[-cp-<tag>]/<trial>.npz), so tasks share no state.
#
#   sbatch benchmark/nfo_vos/run_sam2_video.sbatch                       # B0 + T1, plain
#   A4=fixed sbatch --export=ALL ...;  A4=gated TAU=0.6 sbatch --export=ALL ...   # A/4 (composites_a4 must exist)
#   N = number of admissible trials in results/benchmark/$NFO_RUN/trials.json; the default
#   --array=0-29 fits the rr seq1 pilot (N = 15). Paths for the checkpoint, SAMURAI and Python are
#   baked in below (cluster defaults).
#
# Task map: id 0..N-1 -> B0, trial id; id N..2N-1 -> T1, trial id-N.
# Needs on the cluster: this repo (branch vos-benchmark-pilot), results/benchmark/$NFO_RUN/trials.json
# and the composites dir (both committed), the NATIVE frames data/nfo_final/nfo_final/<seq>/, the base_plus checkpoint at NFO_SAM2_CKPT, a
# sam2 1.x env for B0 (PYTHON_B0) and the SAMURAI repo's sam2 fork for T1 (PYTHON_T1).

#SBATCH --job-name=nfo_sam2v
#SBATCH --array=0-121         # = 2 x N_TRIALS - 1; full rr run: 61 trials (seq1 pilot alone: 0-29)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_sam2v-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_sam2v-%A_%a.out

# cluster defaults (atcremers); every one can still be overridden from the environment
PYTHON_B0=${PYTHON_B0:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
PYTHON_T1=${PYTHON_T1:-$PYTHON_B0}                         # same env, SAMURAI fork put first on the path
SAMURAI_SAM2=${SAMURAI_SAM2:-/storage/user/akid/git_projects/samurai/sam2}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}

export NFO_RUN=${NFO_RUN:-rr}
# number of admissible trials of this run (array ids: first N -> first method/variant, next N -> second)
N_TRIALS=$($PYTHON_B0 -c "import json; print(sum(t['admissible'] for t in json.load(open('results/benchmark/$NFO_RUN/trials.json'))))")
if [ "$SLURM_ARRAY_TASK_ID" -lt "$N_TRIALS" ]; then
  METHOD=b0; PY=$PYTHON_B0; IDX=$SLURM_ARRAY_TASK_ID
  unset PYTHONPATH                                         # B0 = official sam2, never the fork
else
  METHOD=t1; PY=$PYTHON_T1; IDX=$((SLURM_ARRAY_TASK_ID - N_TRIALS))
  export PYTHONPATH=$SAMURAI_SAM2                          # T1 = SAMURAI's sam2 fork (configs/samurai)
fi
A4OPT=${A4:+--a4 $A4 --tau ${TAU:-0.6}}
INITOPT=${INIT_MASK:+--init-mask}                    # INIT_MASK=1: ORACLE GT-mask prompt at t0 (-> <method>-om)           # A4=fixed | A4=gated TAU=0.6 (composites_a4 must exist)

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "method=$METHOD trial-index=$IDX a4=${A4:-none} tau=${TAU:-} python=$PY"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PY" -m benchmark.nfo_vos.run_sam2_video --method "$METHOD" --trial-index "$IDX" $A4OPT $INITOPT --force
date
