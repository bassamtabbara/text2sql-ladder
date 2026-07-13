#!/usr/bin/env bash
# S4 - Speculative decoding. Same checkpoint, faster tokens/sec. Here we use n-gram speculation
# (no separate model needed) to make the point cheaply; a trained draft / MTP head is the
# production path (that is the Cursor multi-token-prediction story).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"

CKPT="checkpoints/full-ft"

# n-gram speculative decoding (self-speculation from the prompt); good for the repetitive,
# schema-echoing tokens common in SQL. Swap in --speculative-model <draft> for a trained draft.
vllm serve "$CKPT" --port 8000 --max-model-len 32768 --served-model-name dedicated \
  --speculative-model "[ngram]" --num-speculative-tokens 5 --ngram-prompt-lookup-max 4 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

# Reuse the frozen eval purely for its p50 latency number; compare to S2's p50.
python -m common.eval --rung S4 --technique speculative --model dedicated \
  --base-url http://localhost:8000/v1 --notes "ngram spec decode; compare p50 to S2"

echo "S4: EX/valid unchanged (same weights); p50 latency should drop ~2-3x. That speed is serving"
echo "logic a per-token API will not run for you."
wait
