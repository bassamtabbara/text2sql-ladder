#!/usr/bin/env bash
# Rung 0: serve the untouched base model, then run zero-shot, few-shot, and RAG.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

MODEL="Qwen/Qwen2.5-3B-Instruct"

# Serve the base model with vLLM (OpenAI-compatible on :8000). Runs in the background.
vllm serve "$MODEL" --port 8000 --max-model-len 32768 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
# wait for the server to accept requests
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 3; done

# --- Baseline for the fine-tuning arc: the UNTOUCHED base Qwen (this is what rungs 2a-2e are
#     measured against, so the comparison is apples-to-apples and not confounded by model size).
python rung0-prompting/prompt.py --mode zero-shot --model "$MODEL"
python rung0-prompting/prompt.py --mode few-shot  --k 5 --model "$MODEL"
python rung0-prompting/prompt.py --mode rag       --k 5 --model "$MODEL"

# --- Frontier REFERENCE panel (optional): several rented frontier models, few-shot+RAG, NO tuning.
#     These are the ceilings the owned model chases, not the fine-tuning baseline. Each runs only if
#     its key is set; each hits an OpenAI-compatible endpoint so the frozen eval works unchanged.
#     The client reads OPENAI_API_KEY, so we set it inline to each vendor's key.
run_frontier () {  # $1=model  $2=base_url  $3=api_key
  if [ -z "$3" ]; then echo "(skipping frontier $1: key not set)"; return; fi
  OPENAI_API_KEY="$3" python rung0-prompting/prompt.py --mode rag --k 5 \
    --model "$1" --base-url "$2" --label "frontier-$1"
}
run_frontier "${FRONTIER_OPENAI_MODEL:-gpt-5.6-sol}"     "https://api.openai.com/v1"                                "${OPENAI_API_KEY:-}"
run_frontier "${FRONTIER_ANTHROPIC_MODEL:-claude-opus-4-8}" "https://api.anthropic.com/v1/"                        "${ANTHROPIC_API_KEY:-}"
run_frontier "${FRONTIER_GEMINI_MODEL:-gemini-3.5-flash}"   "https://generativelanguage.googleapis.com/v1beta/openai/" "${GEMINI_API_KEY:-}"

echo "Rung 0 complete. results/results.csv now has the base-Qwen baseline (zero/few/rag) plus a"
echo "frontier row for each provider key that was set. Fine-tuning gains (rungs 2a-2e) are read"
echo "against the base-Qwen rows; the frontier rows are the ceiling to chase, not the baseline."
