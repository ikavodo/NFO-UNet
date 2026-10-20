#!/bin/bash
# Submit an sbatch file on the best currently free GPU type that meets the training criterion (2026-10-20):
# Ampere or newer (sm >= 8.0: native bf16, tensor cores) and >= 24 GB. #SBATCH lines cannot express "any of these
# types, whichever is free", so this reads sinfo, picks the first ALLOWED type (in order of preference) with a free GPU
# on a usable node (idle/mix, not excluded), and passes it as --gres (command-line flags override #SBATCH). If none is
# free, it queues on the first allowed type that exists. Extra arguments are passed through to sbatch.
#
#   bash benchmark/nfo_vos/submit.sh benchmark/nfo_vos/layers/remote_smoke.sbatch
#   DRY=1 bash benchmark/nfo_vos/submit.sh ...        # print the choice and the sinfo table, submit nothing
#   LOSS=balanced bash benchmark/nfo_vos/submit.sh benchmark/nfo_vos/layers/fusion_adapter.sbatch token 1000 0
set -euo pipefail
# preference order; substrings of the gres type name. Excluded on purpose: Pascal (p6000, p100, titan_x, gtx_1080),
# Volta/Turing (v100, titan_v, rtx_2080, quadro_rtx_6000/8000, t4): no native bf16. 'a6000' does not match quadro_rtx_6000.
ALLOWED=(6000_ada l40s l40 a100 h100 a40 a6000 rtx_4090 rtx_3090 a5000)
EXCLUDE_NODES=${EXCLUDE_NODES:-node2,node22}

TABLE=$(sinfo -N -h -O "NodeList:30,Gres:80,GresUsed:100,StateCompact:12")
pick() {   # $1 = 1: require a free GPU
    for want in "${ALLOWED[@]}"; do
        while read -r node gres used state; do
            [[ ",$EXCLUDE_NODES," == *",$node,"* ]] && continue
            [[ $state =~ ^(idle|mix) ]] || continue
            type=$(sed -E 's/^gpu:([^:]+):([0-9]+).*/\1/' <<<"$gres")
            [[ $type == *"$want"* ]] || continue
            total=$(sed -E 's/^gpu:[^:]+:([0-9]+).*/\1/' <<<"$gres")
            busy=$(sed -E 's/^gpu:[^:]+:([0-9]+).*/\1/' <<<"$used")
            if [[ $1 == 0 ]] || (( total - busy > 0 )); then echo "$type"; return; fi
        done < <(grep '^\S\+\s\+gpu:' <<<"$TABLE")
    done
}
TYPE=$(pick 1); [[ -z $TYPE ]] && TYPE=$(pick 0) && echo "no allowed GPU free now: queueing on $TYPE" >&2
[[ -z $TYPE ]] && { echo "no allowed GPU type on this cluster; table:" >&2; echo "$TABLE" >&2; exit 1; }
echo "GPU type: $TYPE (exclude $EXCLUDE_NODES)" >&2
if [[ -n ${DRY:-} ]]; then echo "$TABLE"; exit 0; fi
sbatch --gres="gpu:$TYPE:1" --exclude="$EXCLUDE_NODES" "$@"
