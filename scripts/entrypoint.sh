#!/usr/bin/env bash
set -euo pipefail

# allow override of command via env RUN_COMMAND
if [ -n "${RUN_COMMAND:-}" ]; then
  echo "Running custom command: $RUN_COMMAND"
  exec /bin/sh -c "$RUN_COMMAND"
fi

# Default: run run_batch.py
exec python -u scripts/run_batch.py