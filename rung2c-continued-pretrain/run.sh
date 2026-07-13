#!/usr/bin/env bash
# Rung 2c (optional): continued pretraining, then re-run SFT on top to see if it helped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

CPT="checkpoints/continued-pt"
python rung2c-continued-pretrain/continued_pretrain.py --out "$CPT"

# Task-tune on the shifted base and evaluate, so the comparison to 2b is apples-to-apples.
python rung2b-full-ft/train_full.py --base "$CPT" --out checkpoints/cpt-then-sft --lr 1e-5

vllm serve checkpoints/cpt-then-sft --port 8000 --max-model-len 32768 --served-model-name cpt-sft &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python -m common.eval --rung 2c --technique continued-pt --model cpt-sft \
  --base-url http://localhost:8000/v1 --notes "cpt then sft"

echo "Rung 2c complete. For Spider this is likely marginal. The lesson is the judgment call."
