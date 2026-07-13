#!/usr/bin/env bash
# Rung 1: real supervised fine-tuning of Gemini Flash on Vertex -- the working "what does vendor
# fine-tuning buy" number. Optionally also runs the OpenAI attempt, which now just records the
# closed-door 403 as an aside (the rug-pull).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

: "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
: "${T2S_GCS_BUCKET:?set T2S_GCS_BUCKET (a writable GCS bucket for the training data)}"

python rung1-vendor-ft/finetune_vertex.py --n 2000 --epochs 3

# Optional aside: the OpenAI door. If OPENAI_API_KEY is set, this records base-mini then (likely)
# prints the 403 that proves rented customization can be revoked out from under you.
if [ -n "${OPENAI_API_KEY:-}" ]; then
  echo "--- Optional: OpenAI fine-tune attempt (the closed door) ---"
  python rung1-vendor-ft/finetune_openai.py --n 500 --base-model gpt-4o-mini-2024-07-18 || true
fi

echo "Rung 1 complete. The lift from base-gemini to vendor-ft-gemini is what SFT buys. And note:"
echo "the tuned weights live in Vertex -- you rented the customization, you don't own it."
