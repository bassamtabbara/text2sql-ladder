#!/usr/bin/env bash
# Rung 2b: full fine-tune, serve the dedicated checkpoint, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation

# clear any stray vLLM server hogging the GPU (e.g. from a prior ad-hoc serve+eval) before training
pkill -f vllm 2>/dev/null || true
sleep 3

CKPT="checkpoints/full-ft"

python rung2b-full-ft/train_full.py --out "$CKPT" --lr 2e-5 --epochs 3

# No catalog base to attach to: this checkpoint needs its own dedicated instance (serving/s2).
vllm serve "$CKPT" --port 8000 --max-model-len 32768 --served-model-name full-ft &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python -m common.eval --rung 2b --technique full-ft --model full-ft \
  --base-url http://localhost:8000/v1 --notes "lr=2e-5, 3ep, completion-only"

echo "Rung 2b complete. Compare against 2a on the hard BIRD slice; probe forgetting on a few"
echo "non-SQL prompts. You now hold a checkpoint that no per-token multi-tenant API can host."
