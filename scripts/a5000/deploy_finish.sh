#!/usr/bin/env bash
# Deploy only a clean, committed tree, preserving existing remote work.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit or preserve local edits before deployment.' >&2; exit 1; }
commit=$(git rev-parse HEAD)
tree=$(git rev-parse 'HEAD^{tree}')
bundle=$(mktemp -t a5000-finish).bundle
trap 'rm -f "$bundle"' EXIT
git bundle create "$bundle" a5000/e0-e1-pure-ce
remote_rsync='env LD_LIBRARY_PATH=/home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/lib/x86_64-linux-gnu /home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/bin/rsync'
ssh A5000 'cd /home/ubuntu/tung.nx/cs-asr-reft && test -z "$(git status --porcelain --untracked-files=no)"'
rsync -az --update --rsync-path="$remote_rsync" --exclude='.git/' --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.egg-info/' ./ A5000:/home/ubuntu/tung.nx/cs-asr-reft/
rsync -az --rsync-path="$remote_rsync" "$bundle" A5000:/home/ubuntu/tung.nx/a5000-bootstrap/a5000-e1-source.bundle
ssh A5000 "set -e
cd /home/ubuntu/tung.nx/cs-asr-reft
git add reft/a5000_followup/finish.py reft/a5000_followup/final_reports.py scripts/a5000/finish.sh scripts/a5000/finish_control.sh scripts/a5000/deploy_finish.sh docs/A5000_REPRODUCIBILITY_LEDGER.md
test \"\$(git write-tree)\" = '$tree'
git fetch /home/ubuntu/tung.nx/a5000-bootstrap/a5000-e1-source.bundle a5000/e0-e1-pure-ce
test \"\$(git rev-parse FETCH_HEAD)\" = '$commit'
git reset --soft FETCH_HEAD"
scripts/a5000/finish_control.sh launch
scripts/a5000/finish_control.sh status
