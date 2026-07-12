# text2sql-ladder

A hands-on walk down the model-customization ladder, from renting a frontier API to owning
inference, using **one use case (text-to-SQL)**, **one base model (`Qwen/Qwen2.5-3B-Instruct`)**,
and **one automatic metric (execution accuracy)** carried unchanged through every rung.

Every rung is a small, self-contained experiment that measurably buys you something the rung above
couldn't, or shows you where it plateaus. The point isn't a state-of-the-art SQL model. The point
is to *feel* the tradeoffs a team faces as it climbs, and to see why the heavier rungs stop fitting
inside a managed per-token API.

This repo is the companion code for a blog series (links below). Clone it, run any rung, and
reproduce the number yourself.

## The ladder

| Rung | Technique | What it buys | Artifact | Where it can be served |
|------|-----------|--------------|----------|------------------------|
| 0 | Prompt + few-shot + RAG | behavior, no weight change | none (rented) | any per-token API |
| 1 | Vendor fine-tune | small lift, rented customization | weights you can't download | only that vendor |
| 2a | QLoRA | style/format/valid-SQL, cheap and reversible | ~tens-of-MB adapter | multi-LoRA on a catalog base |
| 2b | Full fine-tune | higher ceiling, harder queries | full checkpoint | dedicated GPU (BYOC) |
| 2c | Continued pretraining | domain shift (usually skippable) | new base checkpoint | dedicated GPU (BYOC) |
| 2d | DPO | preference shaping, fewer repeat mistakes | checkpoint | dedicated GPU (BYOC) |
| 2e | GRPO (RL) | trained against a real reward | checkpoint | dedicated GPU (BYOC) |
| S1-S5 | custom serving | constrained/spec decoding, verifier pipeline | pipeline + weights | your container only (BYOC) |

## Results (filled in as each rung is run)

Numbers come from real runs on a single H100. Empty until measured. `eval.py` is frozen and shared,
so these are directly comparable.

| Rung | Technique | EX % | Valid-SQL % | p50 latency | Notes |
|------|-----------|:----:|:-----------:|:-----------:|-------|
| 0 | _frontier reference (Opus 4.8, few-shot+RAG)_ | | | | ceiling to chase, not the baseline |
| 0 | zero-shot (base Qwen) | | | | fine-tuning baseline |
| 0 | few-shot (base Qwen) | | | | fine-tuning baseline |
| 0 | few-shot + RAG (base Qwen) | | | | fine-tuning baseline |
| 1 | vendor fine-tune | | | | |
| 2a | QLoRA | | | | |
| 2b | full fine-tune | | | | |
| 2c | continued pretrain | | | | |
| 2d | DPO | | | | |
| 2e | GRPO | | | | |
| S3 | + constrained decode | | | | |
| S4 | + speculative decode | | | | |
| S5 | + verifier pipeline | | | | |

The canonical, machine-readable version lives in [`results/results.csv`](results/results.csv).

## Layout

```
setup/      Nebius provisioning, environment, dataset download
common/     FROZEN: eval.py (execution accuracy), sql_executor.py, data + model client
rung0..2e/  one directory per rung: script(s), config, run.sh, README (the lesson + the number)
serving/    s1..s5: how each artifact is served, and where per-token economics break
results/    run configs, logs, and the source-of-truth results table
```

## Quick start

```bash
# 1. Provision + env (see setup/README.md for the Nebius walkthrough)
python -m venv .venv && source .venv/bin/activate
pip install -r setup/requirements.txt

# 2. Get the data (respects Spider/BIRD licenses; downloads, does not redistribute)
bash setup/download_data.sh

# 3. Establish the baseline, then climb
bash rung0-prompting/run.sh
bash rung2a-qlora/run.sh
# ... etc
```

Every `run.sh` ends by appending its number to `results/results.csv` via the shared `eval.py`.

## Reproducibility

- Library versions are pinned in `setup/requirements.txt`; seeds are fixed in every training script.
- The exact dev-set split is committed in `common/dev_set.json` (question ids only), so everyone
  evaluates on the same 300 examples.
- Trained weights and adapters are **not** committed. They are published to GitHub Releases; each
  rung README links its artifact.
- Datasets are **not** redistributed. `setup/download_data.sh` fetches Spider/BIRD from source.

## Blog series

1. When should you leave the frontier API? (rungs 0-1) — _link TBD_
2. Your first owned model (rung 2a) — _link TBD_
3. When LoRA isn't enough (rungs 2b-2d) — _link TBD_
4. When fine-tuning becomes a system you operate (rung 2e) — _link TBD_
5. Owning inference: why the model isn't the whole product (serving) — _link TBD_

## License

Code is Apache-2.0 (see [LICENSE](LICENSE)), matching the Qwen2.5 base model license. Datasets and
model weights carry their own upstream licenses.
