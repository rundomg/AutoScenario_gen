#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-/home/zx/code/LMDrive/scenario_runner}"
LMDRIVE_ROOT="${LMDRIVE_ROOT:-/home/zx/code/LMDrive}"
CARLA_ROOT="${CARLA_ROOT:-/home/zx/code/autodirve/CARLA_0.9.15}"
PYTHON_BIN="${PYTHON_BIN:-/home/zx/miniconda3/envs/simlingo/bin/python}"
ANTLR_WHEEL="${ANTLR_WHEEL:-/home/zx/.cache/pip/wheels/b1/a3/c2/6df046c09459b73cc9bb6c4401b0be6c47048baf9a1617c485/antlr4_python3_runtime-4.9.3-py3-none-any.whl}"
IOPATH_WHEEL="${IOPATH_WHEEL:-/home/zx/.cache/pip/wheels/9a/a3/b6/ac0fcd1b4ed5cfeb3db92e6a0e476cfd48ed0df92b91080c1d/iopath-0.1.10-py3-none-any.whl}"
FAIRSCALE_WHEEL="${FAIRSCALE_WHEEL:-/home/zx/.cache/pip/wheels/78/a4/c0/fb0a7ef03cff161611c3fa40c6cf898f76e58ec421b88e8cb3/fairscale-0.4.13-py3-none-any.whl}"
CONFIG_XML="${1:-${ROOT_DIR}/results/003_success/dynamic/s0000_c0_lmdrive_scenario.xml}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"

export SCENARIO_RUNNER_ROOT
export CARLA_ROOT
export AUTOSCENARIO_CONFIG_XML="${CONFIG_XML}"
export AUTOSCENARIO_AGENT_TRACE="${AUTOSCENARIO_AGENT_TRACE:-${CONFIG_XML%_scenario.xml}_agent_trace.jsonl}"
export AUTOSCENARIO_REALTIME="${AUTOSCENARIO_REALTIME:-1}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ANTLR_WHEEL}:${IOPATH_WHEEL}:${FAIRSCALE_WHEEL}:${ROOT_DIR}:${ROOT_DIR}/tools:${CARLA_ROOT}/PythonAPI/carla:${LMDRIVE_ROOT}/LAVIS:${LMDRIVE_ROOT}/vision_encoder:${LMDRIVE_ROOT}/leaderboard:${LMDRIVE_ROOT}/leaderboard/team_code:${SCENARIO_RUNNER_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/tools/lmdrive_scenario_runner_compat.py" \
  --scenario AutoScenario_s0000_c0 \
  --configFile "${CONFIG_XML}" \
  --additionalScenario "${ROOT_DIR}/tools/lmdrive_dynamic_scenario.py" \
  --agent "${LMDRIVE_ROOT}/leaderboard/team_code/lmdriver_agent.py" \
  --agentConfig "${ROOT_DIR}/tools/lmdriver_config_local.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --reloadWorld \
  --sync \
  --trafficManagerPort "${TM_PORT}" \
  --timeout 60 \
  --output
