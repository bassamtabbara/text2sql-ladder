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

# --- Frontier REFERENCE line (optional): the strongest model you can rent, few-shot+RAG. This is
#     "rent the best and prompt it well" -- the ceiling the small owned model chases, NOT the
#     fine-tuning baseline. Kept on OpenAI so it matches the rung-1 vendor (one vendor, apples to
#     apples). By default we auto-pick the NEWEST flagship gpt-5* the account has; override with
#     FRONTIER_MODEL=... for a specific id.
if [ -n "${OPENAI_API_KEY:-}" ]; then
  if [ -z "${FRONTIER_MODEL:-}" ]; then
    FRONTIER_MODEL="$(python - <<'PY'
import os, json, urllib.request
req = urllib.request.Request("https://api.openai.com/v1/models",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
data = json.load(urllib.request.urlopen(req))["data"]
# flagship gpt-5* only: drop mini/nano and non-text (audio/realtime/image/chat) variants
cands = [m for m in data if m["id"].startswith("gpt-5")
         and not any(x in m["id"] for x in ("mini", "nano", "chat", "audio", "realtime", "image"))]
cands.sort(key=lambda m: m.get("created", 0), reverse=True)   # newest first
print(cands[0]["id"] if cands else "gpt-5")
PY
)"
  fi
  echo "Frontier ceiling model: $FRONTIER_MODEL"
  python rung0-prompting/prompt.py --mode rag --k 5 \
    --model "$FRONTIER_MODEL" --base-url https://api.openai.com/v1 \
    --label "frontier-$FRONTIER_MODEL"
else
  echo "(skipping frontier reference: set OPENAI_API_KEY to record the frontier ceiling)"
fi

echo "Rung 0 complete. results/results.csv now has the base-Qwen baseline (zero/few/rag) plus,"
echo "if a key was set, the Opus 4.8 reference line. Remember: fine-tuning gains are read against"
echo "the base-Qwen rows; the frontier row is the ceiling to chase, not the baseline."
