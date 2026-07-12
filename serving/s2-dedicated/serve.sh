#!/usr/bin/env bash
# S2 - Dedicated full checkpoint. There is no catalog base to attach to, so this needs its own
# GPU memory and its own instance. Per-token multi-tenant economics stop applying here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

CKPT="checkpoints/full-ft"   # the rung-2b (or 2e) checkpoint

vllm serve "$CKPT" --port 8000 --max-model-len 8192 --served-model-name dedicated &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python -m common.eval --rung S2 --technique dedicated --model dedicated \
  --base-url http://localhost:8000/v1 --notes "own instance, no shared base" --no-record

echo "S2: a full checkpoint occupies a whole instance. This is where BYOC begins."
wait
