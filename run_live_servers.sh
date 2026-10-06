#!/usr/bin/env bash
# Starts Remote A2A Agent 1 (port 8001) & Remote A2A Agent 2 (port 8002)
# as live Uvicorn HTTP processes and runs the Orchestrator over TCP sockets.

set -euo pipefail

cleanup() {
  echo "Stopping background A2A servers..."
  kill "${PID1:-}" "${PID2:-}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting Remote A2A Agent 1 (Clinical Triage) on http://localhost:8001 ..."
uvicorn remote_agents:agent1_app --host localhost --port 8001 --log-level warning &
PID1=$!

echo "Starting Remote A2A Agent 2 (Scheduling & Copay) on http://localhost:8002 ..."
uvicorn remote_agents:agent2_app --host localhost --port 8002 --log-level warning &
PID2=$!

sleep 2

echo "Running Orchestrator against live HTTP A2A servers..."
python3 run_demo.py --live-http "$@"
