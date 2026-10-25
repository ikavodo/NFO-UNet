#!/bin/bash
# B0 (SAM2.1 video) and T1 (SAMURAI) on the NFO windows, one array task per (method, window).
#   NFO_TAG=-c2 sbatch --export=ALL benchmark/nfo_vos/baselines/run_sam2_video.sbatch   # -> masks/b0-c2, masks/t1-c2
# NFO_TRK_PROMPTS=1 adds --trk-prompts (masks/b0trk<TAG>; submit ids 0..N-1 only = B0): eval/pass_run.sh vidtrk.
# Task map: id 0..N-1 -> B0, window id; id N..2N-1 -> T1, window id-N (N = 61 admissible windows).
# Needs on the cluster: data/nfo_final/nfo_final/<seq>/, the base_plus checkpoint at NFO_SAM2_CKPT, a sam2 1.x env
# for B0 (PYTHON_B0) and the SAMURAI repo's sam2 fork for T1 (SAMURAI_SAM2). GPU choice: see adapter/fusion_adapter.sbatch.

#SBATCH --job-name=nfo_sam2v
#SBATCH --array=0-121         # = 2 x N_TRIALS - 1; full rr run: 61 trials (seq1 pilot alone: 0-29)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
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

pwd; hostname; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TRK=${NFO_TRK_PROMPTS:+--trk-prompts}
echo "method=$METHOD trial-index=$IDX tag=${NFO_TAG:-} $TRK python=$PY"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PY" -m benchmark.nfo_vos.baselines.run_sam2_video --method "$METHOD" --trial-index "$IDX" --force $TRK
date
