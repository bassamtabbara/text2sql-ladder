#!/usr/bin/env bash
# Rung 2d: build preference pairs from the 2b model, DPO on them, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

# 1. Serve the 2b checkpoint so we can sample candidates from it.
vllm serve checkpoints/full-ft --port 8000 --max-model-len 32768 --served-model-name full-ft &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

# 2. Generate executor-labeled preference pairs, then stop the sampling server.
python rung2d-dpo/gen_pairs.py --model full-ft --samples 6 --out rung2d-dpo/pairs.jsonl
kill $VLLM_PID 2>/dev/null || true

# 3. DPO train.
python rung2d-dpo/train_dpo.py --base checkpoints/full-ft --out checkpoints/dpo

# 4. Serve + eval the DPO checkpoint.
vllm serve checkpoints/dpo --port 8000 --max-model-len 32768 --served-model-name dpo &
VLLM_PID=$!
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done
python -m common.eval --rung 2d --technique dpo --model dpo \
  --base-url http://localhost:8000/v1 --notes "beta=0.1"

echo "Rung 2d complete. Watch for a class of repeated mistakes (e.g. wrong join direction) fading."
