#!/usr/bin/env bash
# Additive immutable snapshot deployment; never modifies active/historical checkout.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ -z "$(git status --porcelain)" ]] || { echo 'Commit intended source before snapshot deployment.' >&2; exit 1; }
commit=$(git rev-parse HEAD)
branch=$(git branch --show-current)
bundle=$(mktemp -t a5000-campaign).bundle
trap 'rm -f "$bundle"' EXIT
git bundle create "$bundle" "$branch"
remote_rsync='env LD_LIBRARY_PATH=/home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/lib/x86_64-linux-gnu /home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/bin/rsync'
ssh -o ConnectTimeout=12 A5000 'mkdir -p /home/ubuntu/tung.nx/cs-asr-reft-campaign-snapshots /home/ubuntu/tung.nx/a5000-bootstrap'
rsync -az --rsync-path="$remote_rsync" "$bundle" "A5000:/home/ubuntu/tung.nx/a5000-bootstrap/campaign-$commit.bundle"
ssh -o ConnectTimeout=12 A5000 "set -e
snapshot=/home/ubuntu/tung.nx/cs-asr-reft-campaign-snapshots/$commit
if [ ! -d \"\$snapshot\" ]; then
 git clone --single-branch --branch '$branch' /home/ubuntu/tung.nx/a5000-bootstrap/campaign-$commit.bundle \"\$snapshot\"
fi
cd \"\$snapshot\"
test \"\$(git rev-parse HEAD)\" = '$commit'
test -z \"\$(git status --porcelain)\"
printf '%s\\n' \"\$snapshot\""
