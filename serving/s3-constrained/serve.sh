#!/usr/bin/env bash
# S3 - Constrained / grammar decoding. Same checkpoint as S2, but the server only lets the model
# emit valid SQL. Valid-SQL rate jumps with no weight change.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

CKPT="checkpoints/grpo"

vllm serve "$CKPT" --port 8000 --max-model-len 32768 --served-model-name dedicated \
  --guided-decoding-backend xgrammar &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python serving/s3-constrained/eval_constrained.py --model dedicated \
  --base-url http://localhost:8000/v1

echo "S3: compare valid-SQL rate to S2. That lift lives in the container, not the checkpoint."
wait
