#!/bin/bash
# Findings §10.x 26: light, tied (a = 1 − α_∅), teacher-free feature integral trained from scratch (+ amodal aux head),
# then the seq1-only screen (8 crossings) vs item 24's featbigam. Every arm is dumped and scored HERE (one environment, tag @cl);
# the laptop's @lap masks are never mixed in.
#   sbatch benchmark/nfo_vos/adapter/feat_tie.sbatch
# Needs on the cluster, beyond fusion_adapter.sbatch's list:
#   results/benchmark/rr/layers_fusion_adapter/feat_bigam_seed0/fusion_it200.pt   (item 24 checkpoint; copy from the laptop)
#   the fullpass trials (NFO_RUN=fullpass, eval/pass_run.sh), data/yolo11m.pt, ultralytics in $PYTHON's env
# Kill rule (pre-registered): AMGAP at it 200 ≤ 0 or positive on < 8/12 synthetic clips → no NFO screen.
# Writes RUN_DIR=layers_fusion_adapter/feat_tie_seed0/, results/benchmark/rr/feat_integral/tie/train_cl.log,
# masks/adapter-{feattie,featbigam}{,-noC}@cl/, fullpass box_gt.csv (backup box_gt_before_tie_cl.bak.csv), recog *_seq1_tie_cl.
#SBATCH --job-name=nfo_feattie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/storage/slurm/akid/logs/nfo_feattie-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_feattie-%j.out

PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export NFO_SAM2_CKPT=${NFO_SAM2_CKPT:-/storage/user/akid/checkpoints/sam2.1_hiera_base_plus.pt}
export AMP=$("$PYTHON" -c "import torch; print('fp16' if torch.cuda.get_device_capability() < (8, 0) else 'bf16')")
export OMP_NUM_THREADS=2 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=results/benchmark/rr/layers_fusion_adapter; D=results/benchmark/rr/feat_integral/tie; mkdir -p $D
echo "$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) AMP=$AMP commit $(git rev-parse --short HEAD) $(date)"
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$mem" -ge 15000 ] || { echo "GPU has ${mem} MiB; FEAT training peaks ≈ 14 GB: resubmit on a >= 16 GB GPU"; exit 1; }
[ -f $A/feat_bigam_seed0/fusion_it200.pt ] || { echo "MISSING $A/feat_bigam_seed0/fusion_it200.pt (copy from the laptop)"; exit 1; }

echo "== train $(date +%T)"
AMODAL=1 FEAT=1 TIE=1 AGG=none LAM=1 RUN_DIR=$A/feat_tie_seed0 STOP=200 srun "$PYTHON" -u -m benchmark.nfo_vos.adapter.fusion_adapter 1000 0 2>&1 \
  | grep --line-buffered -E "^(VAL|SEL|GAP|AMGAP|AMGAPC|TIE|it [0-9]+ (loss|modal)|params)|Error|Traceback" | tee $D/train_cl.log
g=$(grep "AMGAP it 200" $D/train_cl.log); echo "$g"; grep "TIE it 200\|AMGAPC it 200" $D/train_cl.log
"$PYTHON" -c "import re,sys; m=re.search(r'on Ω ([+-][0-9.]+) \(pos ([0-9.]+)', sys.argv[1]); sys.exit(0 if m and float(m[1])>0 and float(m[2])>=8/12-1e-6 else 1)" "$g" \
  || { echo "== KILL: aux head does not need the integral on synthetic; no NFO screen"; exit 0; }

export NFO_RUN=fullpass MASK_TAG=@cl
echo "== dumps seq1 (crossings 0-7) $(date +%T)"
for i in 0 1 2 3 4 5 6 7; do for a in adapter-feattie adapter-feattie-noC adapter-featbigam adapter-featbigam-noC; do
  "$PYTHON" -m benchmark.nfo_vos.eval.dump_arms $a $i 2>&1 | tail -1; done; done
L="adapter-feattie@cl adapter-feattie-noC@cl adapter-featbigam@cl adapter-featbigam-noC@cl"
B=results/benchmark/fullpass/downstream; cp $B/box_gt.csv $B/box_gt_before_tie_cl.bak.csv
echo "== box $(date +%T)"; "$PYTHON" -m benchmark.nfo_vos.eval.box_metrics score $L 2>&1 | tail -6
echo "== recognition $(date +%T)"; NFO_RECOG_TAG=_seq1_tie_cl NFO_RECOG_MONTAGE=0 "$PYTHON" -m benchmark.nfo_vos.eval.recognition 5 uniform $L 2>&1 | tail -6
echo "== done $(date +%T)"
