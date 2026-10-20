#!/usr/bin/env bash
# Repeatable, non-destructive transfer. Divergent Git states require inspection.
set -euo pipefail
cd "$(dirname "$0")/../.."
local_head=$(git rev-parse HEAD)
remote_head=$(ssh A5000 'if test -d /home/ubuntu/tung.nx/cs-asr-reft/.git; then git -C /home/ubuntu/tung.nx/cs-asr-reft rev-parse HEAD; fi')
if [[ -n "$remote_head" && "$remote_head" != "$local_head" ]]; then
  echo 'Remote HEAD differs; preserve it and inspect before deployment.' >&2
  exit 1
fi
ssh A5000 'if test -d /home/ubuntu/tung.nx/cs-asr-reft/.git; then git -C /home/ubuntu/tung.nx/cs-asr-reft diff --quiet && git -C /home/ubuntu/tung.nx/cs-asr-reft diff --cached --quiet; fi'
rsync -az --update --stats \
  --rsync-path='env LD_LIBRARY_PATH=/home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/lib/x86_64-linux-gnu /home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/bin/rsync' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.DS_Store' \
  --exclude='*.egg-info/' --exclude='.venv/' --exclude='venv/' \
  --exclude='.env' --exclude='.env.*' --exclude='*.pt' --exclude='*.safetensors' \
  --exclude='*.wav' --exclude='*.parquet' --exclude='reft_artifacts/' --exclude='runs/' --exclude='logs/' \
  ./ A5000:/home/ubuntu/tung.nx/cs-asr-reft/
test "$(ssh A5000 'git -C /home/ubuntu/tung.nx/cs-asr-reft rev-parse HEAD')" = "$local_head"
