#!/usr/bin/env bash
# Run locally on the Mac. No secrets are transferred. The remote tmux session survives SSH disconnects.
set -euo pipefail
case "${1:-status}" in
  launch|resume)
    if [[ "${1:-}" == resume ]]; then
      ssh A5000 'rm -f /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/STOP'
    fi
    ssh A5000 'set -e
      cd /home/ubuntu/tung.nx/cs-asr-reft
      test -f /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/freeze.json
      if test -f /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/comparison.json; then echo "Campaign already complete"; exit 0; fi
      if tmux has-session -t a5000-e1 2>/dev/null; then echo "Existing session a5000-e1; use status/logs"; exit 0; fi
      mkdir -p /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/logs
      tmux new-session -d -s a5000-e1 "cd /home/ubuntu/tung.nx/cs-asr-reft; scripts/a5000/e1.sh preflight >> /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/logs/campaign.log 2>&1 && scripts/a5000/e1.sh campaign >> /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/logs/campaign.log 2>&1; code=\$?; printf '\''{\"exit_code\":%d}\\n'\'' \"\$code\" > /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/exit.json"
      tmux list-panes -t a5000-e1 -F "session=#{session_name} pane_pid=#{pane_pid}"'
    ;;
  status)
    ssh A5000 'cd /home/ubuntu/tung.nx/cs-asr-reft; scripts/a5000/e1.sh status; cat /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/process.json /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/exit.json 2>/dev/null; tmux list-sessions 2>/dev/null || true'
    ;;
  logs)
    ssh A5000 'tail -30 /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/logs/campaign.log'
    ;;
  stop)
    ssh A5000 'touch /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/STOP; echo "Graceful stop requested; resume from last optimizer boundary"'
    ;;
  *) echo 'Usage: e1_control.sh launch|status|logs|resume|stop' >&2; exit 2 ;;
esac
