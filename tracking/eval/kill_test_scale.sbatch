#!/bin/bash
# Runs every kill-test variant as one parallel array task each. This one IS worth
# scheduling: each variant is 10-25 minutes of CPU, and there are six of them.
#
#   sbatch tracking/eval/kill_test_scale.sbatch
#   sbatch --array=4 tracking/eval/kill_test_scale.sbatch   # just the separated-roles run
#
# Task map (see VARIANTS below):
#   0 static      occluder fixed in the image, no moving distractor. The original kill test.
#   1 sway        the occluder itself sways: adds distractors but also weakens the occlusion,
#                 so the two effects are confounded. Kept for comparison only.
#   2 sway-frozen as 1, with the segmentation front-end frozen at its default pixel values
#                 instead of scaled per bucket - checks the result is not an artifact of how
#                 the front-end was scaled.
#   3 sweep       2-D sweep of the two scale-relative coefficients at every bucket, to check
#                 one coefficient pair can serve all scales.
#   4 separated   THE CLEAN DESIGN: occluder over the person's path is perfectly static (so
#                 occlusion strength is identical to task 0), and an independent swaying
#                 branch band sits strictly above the person's ground-truth path, so it can
#                 never occlude them. Occlusion and distraction are finally independent
#                 variables. Compare against task 0 to read off the distractor's true cost.
#   5 separated-competitive  as 4, but with the sway amplitude raised to ~the person's own
#                 per-window displacement (~40px at 1x). This matters: at 8px the distractor
#                 oscillates ~6x less than the person translates, so the existing
#                 span*net_displacement term rejects it with no help from the shape term, and
#                 the experiment measures nothing. A distractor is only a distractor if its
#                 motion is competitive.

#SBATCH --job-name=kill_test
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/kill_test-%A_%a.out
#SBATCH --error=/storage/slurm/akid/logs/kill_test-%A_%a.out

# No --gres=gpu: OpenCV/numpy only. --mem is 32G because the 2x bucket holds a few
# ~450x450xT uint8 stacks at once.

PYTHON=/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3

VARIANTS=(
  ""                                  # 0 static
  "--sway 8"                          # 1 sway
  "--sway 8 --frozen-preprocess"      # 2 sway-frozen
  "--alpha-sweep --sway 8"            # 3 sweep
  "--distractor 8"                    # 4 separated-weak
  "--distractor 40"                   # 5 separated-competitive
)
NAMES=(static sway sway-frozen sweep separated-weak separated-competitive)

ARGS=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

pwd; hostname; date
echo "variant: $NAME   args: $ARGS"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENCV_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun "$PYTHON" -m tracking.eval.kill_test_scale $ARGS
date
