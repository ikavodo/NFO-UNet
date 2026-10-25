#!/bin/bash
# SAM2.1 video + tracker prompts (b0trk-c2) on the 61 40-frame windows, then its recognition-vs-N sweep and the plot
# with the other arms (recog_vs_N.png, seq2–4): three chained SLURM jobs, one command.
#   bash benchmark/nfo_vos/eval/vidtrk_windows.sh
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_RUN=rr
M=$(NFO_TAG=-c2 NFO_TRK_PROMPTS=1 sbatch --parsable --export=ALL --array=0-60 benchmark/nfo_vos/baselines/run_sam2_video.sbatch)
S=$(SWEEP_NS="1 3 5 7 11 15 21" SWEEP_ARMS=b0trk-c2 sbatch --parsable --export=ALL --dependency=afterok:$M --array=0-6 benchmark/nfo_vos/eval/recog_sweep.sbatch)
P=$(sbatch --parsable --dependency=afterok:$S --job-name=nfo_recogN_plot --cpus-per-task=4 --mem=16G --time=00:30:00 \
  --output=/storage/slurm/akid/logs/nfo_recogN_plot-%j.out \
  --wrap="cd $PWD && NFO_RUN=rr PYTHONPATH=. $PYTHON -m benchmark.nfo_vos.eval.recog_vs_N")
echo "masks $M (61 tasks) -> sweep $S (7 tasks) -> plot $P: images/benchmark/downstream/recog_vs_N.png"
