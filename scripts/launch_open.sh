#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/open_config_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export VLLM_DISABLE_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1

pids=()

start_service() {
  local name="$1"
  shift
  echo "[open-config] starting $name"
  "$@" >>"$LOG_DIR/${name}.log" 2>&1 &
  local pid=$!
  pids+=("$pid")
  echo "[open-config] $name PID=$pid"
}

cd "$ROOT_DIR"

start_service qwen2_5omni python vllm_models/qwen2_5omni_server.py
sleep 5
start_service qwen3_instruct uvicorn audiotoolagent.apis.qwen3_instruct_api:app --host 0.0.0.0 --port 4014
sleep 5
start_service audioflamingo uvicorn af3.app:app --host 0.0.0.0 --port 4010

cleanup() {
  echo "[open-config] shutting down services"
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
  done
}

trap cleanup EXIT

echo "[open-config] services running. Logs: $LOG_DIR"
wait
