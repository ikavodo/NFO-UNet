#!/bin/bash
# Full-pass protocol (findings §9): box AUC, recognition (N = 5) and hand labels on the 32 one-clip-per-pass clips.
# Needs the pass masks and the -noC arms (eval/pass_run.sh baselines / ours / fullpass; results/benchmark/fullpass/masks -> ../pass/masks), data/yolo11m.pt, frames.
#   sbatch benchmark/nfo_vos/eval/fullpass_score.sbatch
# Writes results/benchmark/fullpass/downstream/{box_gt.csv, recog_N5_uniform_meanfill_fp.csv, hand_rows.csv}; the
# table goes to the log.
#SBATCH --job-name=nfo_fullpass
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_fullpass-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_fullpass-%j.out
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export OMP_NUM_THREADS=8 PYTHONPATH=. NFO_RUN=fullpass NFO_RECOG_TAG=_fp
ARMS="b0-c2 t1-c2 trkbox trk adapter-pf adapter-s1 adapter-s2 adapter-pf-noC adapter-s1-noC adapter-s2-noC"
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date
"$PYTHON" -m benchmark.nfo_vos.eval.box_metrics score $ARMS
"$PYTHON" -m benchmark.nfo_vos.eval.recognition 5 uniform nomask gtbox $ARMS
"$PYTHON" -m benchmark.nfo_vos.eval.report
date
