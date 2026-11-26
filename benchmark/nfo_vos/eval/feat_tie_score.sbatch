#!/bin/bash
# Findings §10.x 27: held-out seq2–4 (fullpass clips 8–31) box AUC + recognition (N = 5) for the 3 tied-integral seeds, their
# -noC controls, B0, T1 and the pixel adapter seeds, in ONE cluster run each (one environment). Submitted by
# adapter/feat_tie_seeds.sh after all dumps. Missing arms are reported and dropped (not fatal), so one bad baseline dir does not
# cost the run. Writes fullpass/downstream/box_gt.csv (backup box_gt_before_tieheld.bak.csv) and recog_N5_uniform_meanfill_heldout_tie_cl.csv.
#SBATCH --job-name=nfo_tie_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_tie_score-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_tie_score-%j.out
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export OMP_NUM_THREADS=8 PYTHONPATH=. NFO_RUN=fullpass
M=results/benchmark/fullpass/masks; B=results/benchmark/fullpass/downstream
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date; "$PYTHON" -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
WANT="adapter-feattie@cl adapter-feattie-s1@cl adapter-feattie-s2@cl adapter-feattie-noC@cl adapter-feattie-s1-noC@cl adapter-feattie-s2-noC@cl
      b0-c2 t1-c2 adapter-pf adapter-s1 adapter-s2 adapter-pf-noC adapter-s1-noC adapter-s2-noC"
L=""
for a in $WANT; do n=$(ls $M/$a 2>/dev/null | grep -cE "^seq[234]_"); echo "masks/$a: $n held-out crossings"
  [ "$n" -ge 24 ] && L="$L $a" || echo "  DROPPED $a (needs 24)"; done
cp $B/box_gt.csv $B/box_gt_before_tieheld.bak.csv
echo "== box $(date +%T)"; "$PYTHON" -m benchmark.nfo_vos.eval.box_metrics score $L 2>&1 | tail -20
echo "== recognition $(date +%T)"; NFO_RECOG_TAG=_heldout_tie_cl NFO_RECOG_MONTAGE=0 "$PYTHON" -m benchmark.nfo_vos.eval.recognition 5 uniform $L 2>&1 | tail -20
echo "== done $(date +%T)"
