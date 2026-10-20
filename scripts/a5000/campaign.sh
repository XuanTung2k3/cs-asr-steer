#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${A5000_CAMPAIGN_PYTHON:-/home/ubuntu/tung.nx/reft-a5000/bin/python}"
exec "$PY" -m reft.a5000_campaign.cli "$@"
