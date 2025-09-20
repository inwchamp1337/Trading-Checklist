#!/usr/bin/env bash
set -euo pipefail

# Check if web mode is enabled first
if [ "${WEB_MODE:-false}" = "true" ]; then
  echo "Starting in web mode..."
  exec python -u web/app.py
fi

# allow override of command via env RUN_COMMAND
if [ -n "${RUN_COMMAND:-}" ]; then
  echo "Running custom command: $RUN_COMMAND"
  exec /bin/sh -c "$RUN_COMMAND"
fi

# Default: run run_batch.py
echo "Starting in default batch mode..."
exec python -u scripts/run_batch.py