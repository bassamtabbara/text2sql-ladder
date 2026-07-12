#!/usr/bin/env bash
# Rung 1: vendor fine-tune. Lightweight, run mainly to feel the constraints. Needs OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"

python rung1-vendor-ft/finetune_openai.py --n 500 --base-model gpt-4o-mini-2024-07-18

echo "Rung 1 complete. The number matters less than the three facts you just met:"
echo "  1) you can't download the weights  2) it runs only on the vendor  3) the door is closing."
