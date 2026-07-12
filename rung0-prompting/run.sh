#!/usr/bin/env bash
# Rung 0: serve the untouched base model, then run zero-shot, few-shot, and RAG.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

MODEL="Qwen/Qwen2.5-3B-Instruct"

# Serve the base model with vLLM (OpenAI-compatible on :8000). Runs in the background.
vllm serve "$MODEL" --port 8000 --max-model-len 8192 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
# wait for the server to accept requests
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python rung0-prompting/prompt.py --mode zero-shot --model "$MODEL"
python rung0-prompting/prompt.py --mode few-shot  --k 5 --model "$MODEL"
python rung0-prompting/prompt.py --mode rag       --k 5 --model "$MODEL"

echo "Rung 0 complete. See results/results.csv for the three numbers and where RAG plateaus."
