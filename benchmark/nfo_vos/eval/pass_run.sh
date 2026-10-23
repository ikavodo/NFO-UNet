#!/bin/bash
# Pass run for eval/mask_fill.py: the 61 benchmark windows (same t0, prompt, warm-up) run to the end of their pass.
# Writes results/benchmark/pass/{trials.json, masks/<arm>/}; the benchmark run (rr) is not touched. From the repo root:
#   bash benchmark/nfo_vos/eval/pass_run.sh
# then, when the arrays are done (or locally after copying results/benchmark/pass/masks back):
#   NFO_RUN=pass PYTHONPATH=. python -m benchmark.nfo_vos.eval.mask_fill score && NFO_RUN=pass ... mask_fill
# Needs what eval/dump_arms.sbatch and baselines/run_sam2_video.sbatch need (frames, checkpoints, results/benchmark/rr/
# trials.json and the adapter checkpoints under results/benchmark/rr/).
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_RUN=pass PYTHONPATH=.
"$PYTHON" -m benchmark.nfo_vos.trials --to-pass-end
N=$("$PYTHON" -c "import json; print(len(json.load(open('results/benchmark/pass/trials.json'))))")
# B0 + T1 (task ids 0..N-1 -> B0, N..2N-1 -> T1), named b0-c2 / t1-c2 like the benchmark arms
NFO_TAG=-c2 sbatch --export=ALL --array=0-$((2 * N - 1)) benchmark/nfo_vos/baselines/run_sam2_video.sbatch
for A in trk adapter-pf adapter-s1 adapter-s2; do
  ARM=$A sbatch --export=ALL --array=0-$((N - 1)) --time=01:00:00 benchmark/nfo_vos/eval/dump_arms.sbatch
done
