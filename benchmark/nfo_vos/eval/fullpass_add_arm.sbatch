#!/bin/bash
# Full-pass protocol: add ONE new arm to an existing fullpass_score.sbatch run without rescoring recognition for the
# others (recognition is per stack and per arm, so the other arms' rows do not change; same env, same YOLO).
# Box AUC is cheap and rescored for all arms (box_gt.csv is rewritten whole); recognition runs for ARM only and its rows
# replace ARM's rows in recog_N5_uniform_meanfill_fp.csv (a copy of the old file is kept as ..._fp_prev.csv).
#   ARM=b0trk-c2 sbatch --export=ALL benchmark/nfo_vos/eval/fullpass_add_arm.sbatch
#SBATCH --job-name=nfo_fullpass_add
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=node1,node2,node6,node7,node22
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/slurm/akid/logs/nfo_fullpass_add-%j.out
#SBATCH --error=/storage/slurm/akid/logs/nfo_fullpass_add-%j.out
set -e
PYTHON=${PYTHON:-/home/wiss/akid/miniconda3/envs/comp_vis/bin/python3}
export OMP_NUM_THREADS=8 PYTHONPATH=. NFO_RUN=fullpass
ARM=${ARM:?set ARM, e.g. ARM=b0trk-c2}
ARMS="b0-c2 t1-c2 $ARM trkbox trk adapter-pf adapter-s1 adapter-s2 adapter-pf-noC adapter-s1-noC adapter-s2-noC"
D=results/benchmark/fullpass/downstream
[ -f $D/recog_N5_uniform_meanfill_fp.csv ] || { echo "no $D/recog_N5_uniform_meanfill_fp.csv: run fullpass_score.sbatch"; exit 1; }
for A in $ARMS; do                # box and the merged table need all 32 clips of every arm
  n=$(ls results/benchmark/fullpass/masks/$A/ | grep -cxFf <("$PYTHON" -c "import json; [print(t['id'] + '.npz') for t in json.load(open('results/benchmark/fullpass/trials.json'))]")) || true
  [ "$n" -eq 32 ] || { echo "arm $A has $n of 32 fullpass clips; wait for its array"; exit 1; }
done
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date
"$PYTHON" -m benchmark.nfo_vos.eval.box_metrics score $ARMS
NFO_RECOG_TAG=_fp_add "$PYTHON" -m benchmark.nfo_vos.eval.recognition 5 uniform "$ARM"
"$PYTHON" - "$ARM" <<'EOF'
import csv, shutil, sys
arm, d = sys.argv[1], 'results/benchmark/fullpass/downstream'
old, new = f'{d}/recog_N5_uniform_meanfill_fp.csv', f'{d}/recog_N5_uniform_meanfill_fp_add.csv'
shutil.copy(old, old.replace('_fp.csv', '_fp_prev.csv'))
rows = [r for r in csv.DictReader(open(old)) if r['arm'] != arm]
add = list(csv.DictReader(open(new)))
stacks = lambda rs: {(r['id'], r['frame']) for r in rs}
ref = stacks([r for r in rows if r['arm'] == 't1-c2'])
assert stacks(add) == ref, f'{arm}: {len(stacks(add))} stacks vs t1-c2 {len(ref)}; not the same stacks'
with open(old, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows + add)
print(f'merged {len(add)} rows of {arm} into {old}')
EOF
NFO_RECOG_TAG=_fp "$PYTHON" -m benchmark.nfo_vos.eval.report
date
