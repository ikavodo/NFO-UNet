#!/bin/bash
# Final table in ONE environment: box AUC (box_metrics) and recognition (YOLO11m, N = 7, mean fill) for all final arms
# in one run each, then report.py. Earlier seeds were scored into separate CSVs; this puts them side by side.
#   sbatch benchmark/nfo_vos/eval/rescore.sbatch
# Writes results/benchmark/rr/downstream/box_gt.csv (the previous one is kept as box_gt_before_rescore.csv),
# recog_N7_uniform_meanfill_final.csv, hand_rows.csv; the table goes to the log.
# Needs: masks/<arm>/ for every arm below (cluster dumps), data/yolo11m.pt, ultralytics in $PYTHON's env.
#SBATCH --job-name=nfo_rescore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_rescore-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_rescore-%j.out
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export OMP_NUM_THREADS=8 PYTHONPATH=. NFO_RECOG_TAG=_final
ARMS="b0-c2 t1-c2 trkbox trk adapter-pf adapter-s1 adapter-s2 adapter-pf-noC adapter-s1-noC adapter-s2-noC"
D=results/benchmark/rr/downstream
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date
"$PYTHON" -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
for a in $ARMS; do n=$(ls results/benchmark/rr/masks/$a 2>/dev/null | wc -l); echo "masks/$a: $n windows"; [ "$n" -ge 61 ] || { echo "MISSING masks for $a"; exit 1; }; done
[ -f $D/box_gt.csv ] && [ ! -f $D/box_gt_before_rescore.csv ] && cp $D/box_gt.csv $D/box_gt_before_rescore.csv
"$PYTHON" -m benchmark.nfo_vos.eval.box_metrics score $ARMS
"$PYTHON" -m benchmark.nfo_vos.eval.recognition 7 uniform nomask gtbox sam2ceil $ARMS
"$PYTHON" -m benchmark.nfo_vos.eval.report
date
