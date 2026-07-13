# Setup

## 1. Provision a GPU (Nebius)

The whole ladder runs on **a single H100 80GB**. Qwen2.5-3B full fine-tuning and GRPO both fit,
and LoRA has room to spare. Rough total cost for the full arc is 15-25 GPU-hours.

1. In the Nebius console, create a GPU instance: 1x H100 80GB, a recent CUDA image (12.x), and at
   least 200 GB disk (checkpoints add up).
2. SSH in. Confirm the GPU:
   ```bash
   nvidia-smi
   ```
3. (Optional) Nebius also has a managed inference/fine-tuning surface. We deliberately do the work
   on a raw instance so the serving rungs are honest about what running a model actually takes.

## 2. Environment

Use **Python 3.11**. Some pins declare `Requires-Python <=3.12`, and in PEP 440 ordering `3.12.3`
is *greater* than `3.12`, so on a 3.12.x VM those releases get excluded and the resolve fails (pip
misreports it, often against `openai`). `uv` gets a standalone 3.11 without touching system Python:

```bash
git clone https://github.com/bassamtabbara/text2sql-ladder
cd text2sql-ladder

# recommended: uv + Python 3.11
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r setup/requirements.txt

# (plain venv also works IF `python3.11` is already on the box:
#   python3.11 -m venv .venv && source .venv/bin/activate && pip install -r setup/requirements.txt )

export T2S_DATA_ROOT=data/spider
```

## 3. Data

```bash
bash setup/download_data.sh          # fetches Spider, builds the frozen dev subset
```

This downloads Spider (never redistributed here), runs a sanity check, and generates the frozen
300-example dev split every rung is scored on. The split is built deterministically (`seed=0`), so
it regenerates identically for anyone who runs this. Committing `common/dev_set.json` is optional
(convenient for readers, not required for your own runs). BIRD is optional; grab it from
https://bird-bench.github.io/ if you want a harder held-out slice to expose where LoRA plateaus,
and point `T2S_DATA_ROOT` at it for a second eval pass.

## 4. Convention for every rung

- Training writes weights/adapters to `outputs/` or `checkpoints/` (git-ignored). Publish the good
  ones to GitHub Releases.
- Serving uses vLLM's OpenAI-compatible server, so `common/eval.py` can hit any model unchanged.
- `run.sh` in each rung trains (if needed), serves, evaluates, and appends the number to
  `results/results.csv`.

## Environment variables

| Var | When you actually need it |
|-----|---------------------------|
| `T2S_DATA_ROOT` | The dataset folder the code reads (default `data/spider`). Set this once, as in step 2. |
| `OPENAI_API_KEY` | For rung 0's frontier reference line and for rung 1 (the vendor fine-tune). `export` it before those runs. Leave it unset for local Qwen runs; vLLM ignores the key (the client defaults to `EMPTY`). |
| `FRONTIER_MODEL` | Optional. Which OpenAI model is the rung-0 ceiling (default `gpt-4o`; e.g. set `gpt-4.1`). |
| `T2S_BASE_URL` | Optional. A fallback eval endpoint used only if a script is invoked without `--base-url`. The rung scripts always pass `--base-url` (local vLLM at `http://localhost:8000/v1`), so you normally never set this. |

So for rung 0: set `T2S_DATA_ROOT` (done in step 2) and, if you want the frontier line,
`OPENAI_API_KEY`. Nothing else.
