#!/bin/bash
# P2-1b execution chain (after READY_FOR_P2_1B_EXECUTION). --kill-on-invalid-dep=yes everywhere.
#   train {CE43, CE44, KL42, KL43, KL44 [, CE42 if the equivalence gate failed]} -> select -> evaluate -> verify -> report
#                                                                               \-> prefix ----------------------/
# DRY_RUN=1 prints the plan without submitting.
set -euo pipefail
cd /home/tungnx/cs-asr-reft
ROOT=/mnt/data/tungnx/cs-asr-reft/p2_1b
DRY=${DRY_RUN:-0}
PRE=$(ls -t $ROOT/preflight_*.json | head -1)
VERDICT=$(python3 -c "import json;print(json.load(open('$PRE'))['verdict'])")
REUSE=$(python3 -c "import json;print(json.load(open('$PRE'))['ce42_reuse'])")
[ "$VERDICT" = "READY_FOR_P2_1B_EXECUTION" ] || { echo "preflight not READY: $VERDICT" >&2; exit 3; }
[ -f "$ROOT/AUTHORIZATION.json" ] || { echo "no authorization" >&2; exit 3; }
sub() { local n=$1; shift; if [ "$DRY" = 1 ]; then echo "sbatch --parsable --kill-on-invalid-dep=yes $*" >&2; echo "<$n>"; else sbatch --parsable --kill-on-invalid-dep=yes "$@"; fi; }
RUNS="CE:43 CE:44 KL:42 KL:43 KL:44"
[ "$REUSE" = "True" ] || RUNS="CE:42 $RUNS"
TJ=""
for r in $RUNS; do a=${r%%:*}; s=${r##*:}; j=$(sub "train_${a}${s}" slurm/reft_p2_1b.sbatch train $a $s); TJ="$TJ:$j"; echo "train $a $s = $j"; done
SEL=$(sub select --dependency=afterok$TJ slurm/reft_p2_1b_cpu.sbatch select)
EVA=$(sub evaluate --dependency=afterok:$SEL slurm/reft_p2_1b_cpu.sbatch evaluate)
VER=$(sub verify --dependency=afterok:$EVA slurm/reft_p2_1b.sbatch verify)
PRF=$(sub prefix --dependency=afterok:$SEL slurm/reft_p2_1b.sbatch prefix)
REP=$(sub report --dependency=afterok:$VER:$PRF slurm/reft_p2_1b_cpu.sbatch report)
echo "select=$SEL evaluate=$EVA verify=$VER prefix=$PRF report=$REP ce42_reuse=$REUSE"
[ "$DRY" = 1 ] || echo "$(date -u +%FT%TZ) trains$TJ select=$SEL evaluate=$EVA verify=$VER prefix=$PRF report=$REP" >> $ROOT/submission.log
