#!/bin/bash
# P2-1 submission (prepared in P2-0; NOT executed there). Dependency chain:
#   train R2 ─afterok─> select R2 ─┐
#                                  ├─afterok(both)─> evaluate R2 L1
#   train L1 ─afterok─> select L1 ─┘
# --kill-on-invalid-dep=yes: if a training or selection job fails, every dependent job is cancelled instead of waiting.
# Requires the project owner's /mnt/data/tungnx/cs-asr-reft/p2_1/AUTHORIZATION.json. DRY_RUN=1 prints the plan only.
set -euo pipefail
cd /home/tungnx/cs-asr-reft
ROOT=/mnt/data/tungnx/cs-asr-reft
DRY=${DRY_RUN:-0}
if [ "$DRY" != 1 ] && [ ! -f "$ROOT/p2_1/AUTHORIZATION.json" ]; then
  echo "P2-1 not authorized: $ROOT/p2_1/AUTHORIZATION.json is absent" >&2; exit 3
fi
sub() {  # sub NAME sbatch-args...
  local name=$1; shift
  if [ "$DRY" = 1 ]; then echo "sbatch --parsable --kill-on-invalid-dep=yes $*" >&2; echo "<$name>"
  else sbatch --parsable --kill-on-invalid-dep=yes "$@"; fi
}
TR=$(sub train_R2 slurm/reft_p2.sbatch train R2)
TL=$(sub train_L1 slurm/reft_p2.sbatch train L1)
SR=$(sub select_R2 --dependency=afterok:$TR slurm/reft_p2_cpu.sbatch select $ROOT/p2_1/R2)
SL=$(sub select_L1 --dependency=afterok:$TL slurm/reft_p2_cpu.sbatch select $ROOT/p2_1/L1)
EV=$(sub evaluate --dependency=afterok:$SR:$SL slurm/reft_p2_cpu.sbatch evaluate $ROOT/p2_1/R2 $ROOT/p2_1/L1)
echo "train_R2=$TR train_L1=$TL select_R2=$SR select_L1=$SL evaluate=$EV"
if [ "$DRY" != 1 ]; then echo "$(date -u +%FT%TZ) train_R2=$TR train_L1=$TL select_R2=$SR select_L1=$SL evaluate=$EV" >> "$ROOT/p2_1/submission.log"; fi
