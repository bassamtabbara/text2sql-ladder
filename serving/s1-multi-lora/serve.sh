#!/usr/bin/env bash
# S1 - Multi-LoRA serving. One base in GPU memory, many tiny adapters swapped on top.
# This is exactly the economics that let per-token platforms serve LoRA customers cheaply.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

BASE="Qwen/Qwen2.5-3B-Instruct"

# Load several adapters at once. Here we show the text-to-SQL adapter alongside a second one to
# make the point that they share a single base. Train/point the second adapter at any bounded task.
vllm serve "$BASE" --port 8000 --max-model-len 8192 \
  --enable-lora --max-loras 4 --max-lora-rank 32 \
  --lora-modules sql=outputs/qlora other=outputs/qlora &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

echo "Serving base '$BASE' with adapters loaded on top; list them:"
curl -s http://localhost:8000/v1/models | python -m json.tool

# Evaluate by adapter name; note GPU memory is ~one base, not one-per-tenant.
python -m common.eval --rung S1 --technique multi-lora --model sql \
  --base-url http://localhost:8000/v1 --notes "shared base, hot-swapped adapter" --no-record

echo "S1: same base memory serves many tenants' adapters. This is why tier-1 is cheap."
wait
