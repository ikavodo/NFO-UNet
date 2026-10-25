#!/bin/bash
# Pass run: the 61 benchmark windows (same t0, prompt, warm-up) run to the end of their pass, for eval/mask_fill.py, and
# its subset 'fullpass' (earliest start per pass: 32 clips, every frame once) for the full-pass scoring protocol.
# Writes results/benchmark/{pass,fullpass}/; fullpass/masks is a link to pass/masks. The benchmark run (rr) is untouched.
#   bash benchmark/nfo_vos/eval/pass_run.sh baselines                 # trials + B0/T1 arrays (independent of our arms)
#   bash benchmark/nfo_vos/eval/pass_run.sh ours trk adapter-pf ...   # our arms (e.g. adapter-pf-h4 once chosen)
#   bash benchmark/nfo_vos/eval/pass_run.sh fullpass adapter-pf-noC ...  # arms needed only on the 32 fullpass clips
#   bash benchmark/nfo_vos/eval/pass_run.sh vidtrk                    # B0 + tracker prompts on the 32 clips -> b0trk-c2
# Then (cluster or locally after copying results/benchmark/pass/masks back):
#   NFO_RUN=pass  python -m benchmark.nfo_vos.eval.mask_fill score && NFO_RUN=pass python -m benchmark.nfo_vos.eval.mask_fill
#   NFO_RUN=fullpass python -m benchmark.nfo_vos.eval.box_metrics score <arms>; ...eval.recognition 5 uniform <arms>; ...eval.report
# Needs what eval/dump_arms.sbatch and baselines/run_sam2_video.sbatch need (frames, checkpoints, results/benchmark/rr/
# trials.json and the adapter checkpoints under results/benchmark/rr/).
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export PYTHONPATH=.
STAGE=$1; shift
if [ ! -f results/benchmark/pass/trials.json ]; then
  NFO_RUN=pass "$PYTHON" -m benchmark.nfo_vos.trials --to-pass-end
  NFO_RUN=fullpass "$PYTHON" -m benchmark.nfo_vos.trials --to-pass-end --first-per-pass
  ln -sfn ../pass/masks results/benchmark/fullpass/masks
fi
export NFO_RUN=pass
N=$("$PYTHON" -c "import json; print(len(json.load(open('results/benchmark/pass/trials.json'))))")
if [ "$STAGE" = baselines ]; then
  # task ids 0..N-1 -> B0, N..2N-1 -> T1; named b0-c2 / t1-c2 like the benchmark arms
  NFO_TAG=-c2 sbatch --export=ALL --array=0-$((2 * N - 1)) benchmark/nfo_vos/baselines/run_sam2_video.sbatch
elif [ "$STAGE" = ours ]; then
  for A in "$@"; do ARM=$A sbatch --export=ALL --array=0-$((N - 1)) --time=01:00:00 benchmark/nfo_vos/eval/dump_arms.sbatch; done
elif [ "$STAGE" = fullpass ]; then
  # writes through fullpass/masks -> pass/masks; the fullpass clips are a subset of the pass clips (same ids)
  N=$("$PYTHON" -c "import json; print(len(json.load(open('results/benchmark/fullpass/trials.json'))))")
  for A in "$@"; do NFO_RUN=fullpass ARM=$A sbatch --export=ALL --array=0-$((N - 1)) --time=01:00:00 benchmark/nfo_vos/eval/dump_arms.sbatch; done
elif [ "$STAGE" = vidtrk ]; then
  # SAM2.1 video (memory) + tracker box/point every frame; needs fullpass/masks/trkbox (boxes). Ids 0..N-1 = B0 only.
  N=$("$PYTHON" -c "import json; print(len(json.load(open('results/benchmark/fullpass/trials.json'))))")
  NFO_RUN=fullpass NFO_TAG=-c2 NFO_TRK_PROMPTS=1 sbatch --export=ALL --array=0-$((N - 1)) benchmark/nfo_vos/baselines/run_sam2_video.sbatch
else
  echo "usage: $0 baselines | ours ARM... | fullpass ARM... | vidtrk"; exit 1
fi
