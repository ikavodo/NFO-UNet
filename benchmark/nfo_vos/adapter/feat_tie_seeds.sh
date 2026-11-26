#!/bin/bash
# Findings §10.x 27: 3 seeds of the frozen tied-integral recipe on the held-out seq2–4, spread across nodes:
#   train seeds 1, 2 (2-task array; seed 0 = the §10.x 26 run, same recipe) ->
#   6 mask arms × 24 held-out crossings (fullpass clips 8–31), one array task per seed and crossing (arm + -noC share the encoder) (seed-0 arms start at once) ->
#   one scoring job (box AUC + recognition, all arms together).
#   bash benchmark/nfo_vos/adapter/feat_tie_seeds.sh        (from the repo root, on the cluster login node)
# Needs: layers_fusion_adapter/feat_tie_seed0/fusion_it200.pt (from the §10.x 26 job), fullpass trials + B0/T1/pixel-adapter masks.
set -e
A=results/benchmark/rr/layers_fusion_adapter
[ -f $A/feat_tie_seed0/fusion_it200.pt ] || { echo "MISSING $A/feat_tie_seed0/fusion_it200.pt"; exit 1; }
export NFO_RUN=fullpass MASK_TAG=@cl
TJ=$(sbatch --parsable --array=1-2 benchmark/nfo_vos/adapter/feat_tie_train.sbatch); echo "train array $TJ"
DJ=""
for s in 0 1 2; do
  sfx=$([ $s = 0 ] && echo "" || echo "-s$s"); dep=$([ $s = 0 ] && echo "" || echo "--dependency=afterok:${TJ}_$s")
  arm=adapter-feattie$sfx,adapter-feattie$sfx-noC        # one task per crossing dumps both: the encoder pass is shared
  j=$(ARM=$arm sbatch --parsable --export=ALL $dep --array=8-31 --time=01:00:00 --job-name=nfo_dump_feattie$sfx benchmark/nfo_vos/eval/dump_arms.sbatch)
  echo "dump $arm: $j"; DJ="$DJ:$j"
done
SJ=$(sbatch --parsable --dependency=afterok$DJ benchmark/nfo_vos/eval/feat_tie_score.sbatch); echo "score $SJ (after$DJ)"
