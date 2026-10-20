#!/usr/bin/env bash
# Local Mac controls. The supervisor and all GPU work run entirely on A5000.
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-status}" in
  launch|resume)
    if [[ "${1:-}" == resume ]]; then ssh A5000 'rm -f ~/shared/tung.nx/a5000-e1-pilot-v1/AUTO_STOP ~/shared/tung.nx/a5000-e1-pilot-v1/STOP'; fi
    ssh A5000 'set -e
      if tmux has-session -t a5000-e1-finish 2>/dev/null; then echo "Supervisor already active"; exit 0; fi
      mkdir -p ~/shared/tung.nx/a5000-e1-pilot-v1/logs ~/shared/tung.nx/a5000-e1-pilot-v1/auto
      tmux new-session -d -s a5000-e1-finish "cd /home/ubuntu/tung.nx/cs-asr-reft; scripts/a5000/finish.sh >> /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/logs/supervisor.log 2>&1; code=\$?; printf '\''{\"exit_code\":%d}\\n'\'' \"\$code\" > /home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/auto/exit.json"
      tmux list-panes -t a5000-e1-finish -F "session=#{session_name} pane_pid=#{pane_pid}"'
    ;;
  status)
    ssh A5000 'cat ~/shared/tung.nx/a5000-e1-pilot-v1/auto/status.json ~/shared/tung.nx/a5000-e1-pilot-v1/status.json ~/shared/tung.nx/a5000-e1-pilot-v1/auto/child.json 2>/dev/null; tmux list-sessions 2>/dev/null || true'
    ;;
  logs)
    ssh A5000 'tail -15 ~/shared/tung.nx/a5000-e1-pilot-v1/logs/supervisor.log; tail -5 ~/shared/tung.nx/a5000-e1-pilot-v1/logs/campaign.log; ls -lt ~/shared/tung.nx/a5000-e1-pilot-v1/logs/auto* 2>/dev/null | head -5'
    ;;
  stop)
    ssh A5000 'touch ~/shared/tung.nx/a5000-e1-pilot-v1/AUTO_STOP ~/shared/tung.nx/a5000-e1-pilot-v1/STOP; echo "Graceful stop requested at worker checkpoint boundary"'
    ;;
  fetch)
    mkdir -p ../cs-asr-reft-reports
    rsync -az --update --rsync-path='env LD_LIBRARY_PATH=/home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/lib/x86_64-linux-gnu /home/ubuntu/tung.nx/a5000-bootstrap/rsync-package/extracted/usr/bin/rsync' \
      A5000:/home/ubuntu/shared/tung.nx/a5000-e1-pilot-v1/reports/ ../cs-asr-reft-reports/
    ;;
  *) echo 'Usage: finish_control.sh launch|status|logs|stop|resume|fetch' >&2; exit 2 ;;
esac
