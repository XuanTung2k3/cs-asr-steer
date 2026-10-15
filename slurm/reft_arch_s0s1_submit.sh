#!/bin/bash
# S1 execution chain (only after preflight ARCH_S0_READY_FOR_S1; every job re-checks the freeze, train re-checks the preflight).
# train x6 -> select -> {evaluate, verify, mech-anchors -> mech-gpu} -> report ; afterok + kill-on-invalid-dep.
set -euo pipefail
cd /home/tungnx/cs-asr-reft
ROOT=/mnt/data/tungnx/cs-asr-reft/arch_s0s1
PRE=$ROOT/$(cd $ROOT && ls preflight_*.json | sort -t_ -k2 -n | tail -1)
VERDICT=$(python3 -c "import json;d=json.load(open('$PRE'));print(d['verdict'] if not d['dry_run'] else 'DRY')")
[ "$VERDICT" = "ARCH_S0_READY_FOR_S1" ] || { echo "preflight not READY: $VERDICT ($PRE)" >&2; exit 3; }
[ -f "$ROOT/AUTHORIZATION.json" ] || { echo "no authorization" >&2; exit 3; }
D="--kill-on-invalid-dep=yes"
TJ=""
for a in R3 M1; do for s in 42 43 44; do j=$(sbatch --parsable slurm/reft_arch_s0s1.sbatch train $a $s); TJ="$TJ:$j"; echo "train $a $s = $j"; done; done
SEL=$(sbatch --parsable $D --dependency=afterok$TJ slurm/reft_arch_s0s1_cpu.sbatch select)
EVA=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_arch_s0s1_cpu.sbatch evaluate)
VER=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_arch_s0s1.sbatch verify)
MA=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_arch_s0s1_cpu.sbatch mech-anchors)
MG=$(sbatch --parsable $D --dependency=afterok:$MA slurm/reft_arch_s0s1.sbatch mech-gpu)
REP=$(sbatch --parsable $D --dependency=afterok:$EVA:$VER:$MG slurm/reft_arch_s0s1_cpu.sbatch report)
echo "$(date -u +%FT%TZ) preflight=$(basename $PRE) trains$TJ select=$SEL evaluate=$EVA verify=$VER mech-anchors=$MA mech-gpu=$MG report=$REP" | tee -a $ROOT/submission.log
