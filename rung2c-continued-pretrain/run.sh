#!/usr/bin/env bash
# Rung 2c (optional): continued pretraining, then re-run SFT on top to see if it helped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# clear any stray vLLM server hogging the GPU before training
pkill -f vllm 2>/dev/null || true
sleep 3

CPT="checkpoints/continued-pt"
python rung2c-continued-pretrain/continued_pretrain.py --out "$CPT"

# Task-tune the shifted base with QLoRA (our best method from 2a), so the comparison is to plain
# QLoRA (73.3): did continued pretraining move the base enough to beat that?
ADAPTER="outputs/cpt-then-qlora"
rm -rf "$ADAPTER"
python rung2a-qlora/train_qlora.py --base "$CPT" --out "$ADAPTER" --rank 32 --epochs 3

vllm serve "$CPT" --port 8000 --max-model-len 32768 \
  --enable-lora --max-lora-rank 32 --lora-modules cpt-qlora="$ADAPTER" &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

python -m common.eval --rung 2c --technique continued-pt-qlora --model cpt-qlora \
  --base-url http://localhost:8000/v1 --notes "cpt then qlora rank32; compare vs plain qlora 73.3"

echo "Rung 2c complete. Our in-domain corpus is tiny (~1-2M tokens), so expect ~no gain over plain"
echo "QLoRA -- which IS the lesson: continued pretraining needs massive in-domain text to matter."
