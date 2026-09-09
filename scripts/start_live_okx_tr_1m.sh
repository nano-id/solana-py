#!/usr/bin/env bash
# Start live OKX TR 1m grow150 bot in tmux (decision @ 59s).
set -euo pipefail
cd "${HOME}/bot"
export PATH="${HOME}/.local/bin:${PATH}"
# shellcheck disable=SC1090
source "${HOME}/.bashrc" 2>/dev/null || true
if [[ -f "${HOME}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.venv/bin/activate"
fi
export PYTHONPATH="${HOME}/bot${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "${HOME}/bot/logs"
tmux has-session -t live-okx-tr 2>/dev/null && tmux kill-session -t live-okx-tr
tmux new -s live-okx-tr -d "bash -lc 'cd ~/bot; source ~/.venv/bin/activate 2>/dev/null; export PYTHONPATH=~/bot; python -m market_making.run_live_okx_tr_1m_grow150 --config config/live_okx_tr_1m_grow150.yaml --duration-sec 0 2>&1 | tee -a ~/bot/logs/live_okx_tr_1m_grow150.log'"
sleep 2
tmux ls
tmux capture-pane -t live-okx-tr -p | tail -30
