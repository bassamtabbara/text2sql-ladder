#!/usr/bin/env bash
# S5 - Verifier pipeline: generate -> execute -> repair, scored end to end.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

CKPT="checkpoints/full-ft"   # or checkpoints/grpo

# Serve the model the pipeline talks to.
vllm serve "$CKPT" --port 8000 --max-model-len 32768 --served-model-name dedicated &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

# Drive the pipeline (must run from its own dir so the sibling `pipeline` import resolves).
( cd serving/s5-verifier && PYTHONPATH="$OLDPWD" python eval_verifier.py \
    --model dedicated --base-url http://localhost:8000/v1 --max-repairs 2 )

echo "S5: EX should rise from the self-repair loop. The product is now the pipeline + the weights,"
echo "not a checkpoint you could upload to a per-token API. That is tier 2 in one sentence."
wait
