#!/bin/bash
# Recognition vs stack length N on the 32 whole crossings (eval/recog_vs_N.py, NFO_RUN=fullpass): one array task per
# (N, arm), all in parallel. N = 5 is not rerun: it is the headline CSV recog_N5_uniform_meanfill_fp.csv.
# Writes results/benchmark/fullpass/downstream/recog_N<N>_uniform_meanfill_fpN_<arm>.csv.
#   sbatch benchmark/nfo_vos/eval/recog_sweep.sbatch          (array size = len(NS) x len(ARMS) = 36)
# 40-frame windows, one extra arm (eval/vidtrk_windows.sh): NFO_RUN=rr SWEEP_NS="1 3 5 7 11 15 21" SWEEP_ARMS=b0trk-c2
#   -> recog_N<N>_uniform_meanfill_ablN_<arm>.csv next to the _ablN CSVs of the other arms (--array=0-6).
# Then copy results/benchmark/fullpass/downstream/ back and run NFO_RUN=fullpass python -m benchmark.nfo_vos.eval.recog_vs_N
#SBATCH --job-name=nfo_recogN
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_recogN-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_recogN-%A_%a.out
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
NS=(${SWEEP_NS:-1 3 7 11 15 21})
ARMS=(${SWEEP_ARMS:-nomask adapter-pf t1-c2 b0-c2 trk b0trk-c2})       # default = recog_vs_N.ARMS (fullpass)
export NFO_RUN=${NFO_RUN:-fullpass}
TAG=$([ "$NFO_RUN" = fullpass ] && echo _fpN || echo _ablN)
N=${NS[$((SLURM_ARRAY_TASK_ID / ${#ARMS[@]}))]}
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID % ${#ARMS[@]}))]}
export OMP_NUM_THREADS=8 PYTHONPATH=. NFO_RECOG_TAG=${TAG}_$ARM NFO_RECOG_MONTAGE=0
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date; echo "run=$NFO_RUN N=$N arm=$ARM tag=$NFO_RECOG_TAG"
"$PYTHON" -m benchmark.nfo_vos.eval.recognition "$N" uniform "$ARM"
date
