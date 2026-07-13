#!/usr/bin/env bash
# Rung 2d: build preference pairs from the 2b model, DPO on them, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# clear any stray vLLM server hogging the GPU before we start
pkill -f vllm 2>/dev/null || true
sleep 3

# Build on our best SFT model: merge the rung-2a QLoRA adapter into a standalone checkpoint.
MERGED="checkpoints/qlora-merged"
[ -d "$MERGED" ] || python rung2d-dpo/merge_qlora.py --out "$MERGED"

# 1. Serve the merged (best SFT) model so we can sample candidates from it.
vllm serve "$MERGED" --port 8000 --max-model-len 32768 --served-model-name sft &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

# 2. Generate executor-labeled preference pairs, then stop the sampling server.
python rung2d-dpo/gen_pairs.py --model sft --samples 6 --out rung2d-dpo/pairs.jsonl
kill $VLLM_PID 2>/dev/null || true

# 3. DPO from the merged (best SFT) model.
python rung2d-dpo/train_dpo.py --base "$MERGED" --out checkpoints/dpo

# 4. Serve + eval the DPO checkpoint.
vllm serve checkpoints/dpo --port 8000 --max-model-len 32768 --served-model-name dpo &
VLLM_PID=$!
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done
python -m common.eval --rung 2d --technique dpo --model dpo \
  --base-url http://localhost:8000/v1 --notes "DPO on QLoRA-merged, beta=0.1"

echo "Rung 2d complete. Watch for a class of repeated mistakes (e.g. wrong join direction) fading."
