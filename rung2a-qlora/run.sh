#!/usr/bin/env bash
# Rung 2a: train a QLoRA adapter, serve base+adapter with vLLM, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation

BASE="Qwen/Qwen2.5-3B-Instruct"
ADAPTER="outputs/qlora"

rm -rf "$ADAPTER"   # start clean so a re-run doesn't mix stale checkpoints / old-rank shards
python rung2a-qlora/train_qlora.py --base "$BASE" --out "$ADAPTER" --rank 32 --epochs 3

# Serve the base once and load the adapter on top (this is exactly the multi-LoRA pattern).
# --max-lora-rank must be >= the adapter's rank (we trained rank 32; vLLM defaults to 16)
vllm serve "$BASE" --port 8000 --max-model-len 32768 \
  --enable-lora --max-lora-rank 32 --lora-modules qlora="$ADAPTER" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

# eval hits the adapter by name
python -m common.eval --rung 2a --technique qlora --model qlora \
  --base-url http://localhost:8000/v1 --notes "rank=32, completion-only, 3ep"

echo "Rung 2a complete. Compare EX/valid-SQL against rung 0's RAG plateau."
