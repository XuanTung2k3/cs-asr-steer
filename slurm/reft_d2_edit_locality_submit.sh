#!/bin/bash
# D2 execution chain (only after mu_freeze.json and preflight D2_READY_FOR_TRAINING; every job re-checks the freeze and mu,
# train re-checks the preflight). train 42, 43 (then 44 after 42: at most two GPU jobs) -> select -> {evaluate, verify, mech}
# -> prefix (after verify) -> report ; afterok + kill-on-invalid-dep.
set -euo pipefail
cd /home/tungnx/cs-asr-reft
ROOT=/mnt/data/tungnx/cs-asr-reft/d2_edit_locality
PRE=$ROOT/$(cd $ROOT && ls preflight_*.json | sort -t_ -k2 -n | tail -1)
VERDICT=$(python3 -c "import json;d=json.load(open('$PRE'));print(d['verdict'] if not d['dry_run'] else 'DRY')")
[ "$VERDICT" = "D2_READY_FOR_TRAINING" ] || { echo "preflight not READY: $VERDICT ($PRE)" >&2; exit 3; }
[ -f "$ROOT/AUTHORIZATION.json" ] || { echo "no authorization" >&2; exit 3; }
[ -f "$ROOT/mu_freeze.json" ] || { echo "mu not frozen" >&2; exit 3; }
D="--kill-on-invalid-dep=yes"
T42=$(sbatch --parsable slurm/reft_d2_edit_locality.sbatch train 42)
T43=$(sbatch --parsable slurm/reft_d2_edit_locality.sbatch train 43)
T44=$(sbatch --parsable $D --dependency=afterok:$T42 slurm/reft_d2_edit_locality.sbatch train 44)
SEL=$(sbatch --parsable $D --dependency=afterok:$T42:$T43:$T44 slurm/reft_d2_edit_locality_cpu.sbatch select)
EVA=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_d2_edit_locality_cpu.sbatch evaluate)
VER=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_d2_edit_locality.sbatch verify)
MEC=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_d2_edit_locality.sbatch mech)
PFX=$(sbatch --parsable $D --dependency=afterok:$VER slurm/reft_d2_edit_locality.sbatch prefix)
REP=$(sbatch --parsable $D --dependency=afterok:$EVA:$VER:$MEC:$PFX slurm/reft_d2_edit_locality_cpu.sbatch report)
echo "$(date -u +%FT%TZ) preflight=$(basename $PRE) train42=$T42 train43=$T43 train44=$T44 select=$SEL evaluate=$EVA verify=$VER mech=$MEC prefix=$PFX report=$REP" | tee -a $ROOT/submission.log
