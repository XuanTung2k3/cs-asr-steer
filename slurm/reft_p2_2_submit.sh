#!/bin/bash
# P2-2B execution chain (only after preflight READY_FOR_P2_2_TRAINING; each job re-checks freeze and preflight).
# train x3 -> select -> {evaluate, verify, mech-anchors -> mech-replay} -> report ; afterok + kill-on-invalid-dep.
set -euo pipefail
cd /home/tungnx/cs-asr-reft
D="--kill-on-invalid-dep=yes"
T42=$(sbatch --parsable slurm/reft_p2_2.sbatch train 42)
T43=$(sbatch --parsable slurm/reft_p2_2.sbatch train 43)
T44=$(sbatch --parsable slurm/reft_p2_2.sbatch train 44)
SEL=$(sbatch --parsable $D --dependency=afterok:$T42:$T43:$T44 slurm/reft_p2_2_cpu.sbatch select)
EVA=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_p2_2_cpu.sbatch evaluate)
VER=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_p2_2.sbatch verify)
MA=$(sbatch --parsable $D --dependency=afterok:$SEL slurm/reft_p2_2_cpu.sbatch mech-anchors)
MR=$(sbatch --parsable $D --dependency=afterok:$MA slurm/reft_p2_2.sbatch mech-replay)
REP=$(sbatch --parsable $D --dependency=afterok:$EVA:$VER:$MR slurm/reft_p2_2_cpu.sbatch report)
echo "train=$T42,$T43,$T44 select=$SEL evaluate=$EVA verify=$VER mech-anchors=$MA mech-replay=$MR report=$REP" | tee -a /mnt/data/tungnx/cs-asr-reft/p2_2/submission.log
