#!/usr/bin/env bash
# Rung 2e: GRPO against the execution reward, then serve + evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

# Train from the DPO checkpoint (fall back to full-ft if you skipped 2d).
BASE="checkpoints/dpo"
[ -d "$BASE" ] || BASE="checkpoints/full-ft"

python rung2e-grpo/train_grpo.py --base "$BASE" --out checkpoints/grpo --steps 500

vllm serve checkpoints/grpo --port 8000 --max-model-len 32768 --served-model-name grpo &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python -m common.eval --rung 2e --technique grpo --model grpo \
  --base-url http://localhost:8000/v1 --notes "reward=execution"

echo "Rung 2e complete. EX should push past the SFT/DPO ceiling on the hardest queries."
echo "The bigger takeaway is operational: you just ran generate-score-update, not a submitted job."
