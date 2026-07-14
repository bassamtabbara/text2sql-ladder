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

## Results

Real runs on a single H100, Spider dev (300-example frozen split), via the shared frozen `eval.py`
(directly comparable). Rung-0 frontier rows show each model's **best** prompting mode.

| Rung | Technique | EX % | Valid-SQL % | p50 (ms) | Notes |
|------|-----------|:----:|:-----------:|:--------:|-------|
| 0 | zero-shot (base Qwen-3B) | 60.3 | 85.7 | 236 | fine-tuning baseline |
| 0 | few-shot (base Qwen-3B) | 63.3 | 87.3 | 249 | |
| 0 | few-shot + RAG (base Qwen-3B) | 66.3 | 85.0 | 405 | best prompting on the owned model |
| 0 | _frontier gemini-3.5-flash_ | 72.0 | 83.0 | 2031 | RAG; AI Studio (Vertex serves it at 83.7, see rung 1) |
| 0 | _frontier gpt-5.6-sol_ | 83.7 | 100 | 3714 | zero-shot (RAG didn't help) |
| 0 | _frontier claude-opus-4-8_ | 94.3 | 99.7 | 1992 | few-shot |
| 1 | base-gemini (zero-shot) | 83.7 | 99.3 | 883 | untuned, on Vertex |
| 1 | vendor-ft-gemini (Vertex SFT) | 88.0 | 99.3 | 1013 | +4.3 lift; weights stay in Vertex |
| 2a | **QLoRA** (rank 32) | **73.3** | 93.7 | 412 | **beats prompting (66.3) — owning wins** |
| 2b | full fine-tune | 70.3 | 92.0 | 199 | underperforms QLoRA (regularization/forgetting) |
| 2c | continued-PT → QLoRA | 72.0 | 92.0 | 417 | no gain (tiny in-domain corpus) |
| 2d | DPO (on QLoRA-merged) | 73.7 | 93.3 | 187 | flat (already-good model, self-sampled prefs) |
| 2e | GRPO (execution reward) | 74.0 | 93.0 | 187 | small gain; RL is the heaviest rung |
| S3 | + constrained decode | | | | _serving rungs: TBD_ |
| S4 | + speculative decode | | | | _TBD_ |
| S5 | + verifier pipeline | | | | _TBD_ |

Headline: prompting the owned 3B plateaus at 66.3; a lightweight QLoRA adapter clears it (73.3);
heavier training (full-FT, continued-PT, DPO, RL) all cluster ~70–74 on near-solved Spider — the
heavier methods earn their keep on harder tasks, not here. The full run-by-run log (including
superseded attempts, e.g. the first rank-16 QLoRA that *lost* to RAG) lives in
[`results/results.csv`](results/results.csv).

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
# Use Python 3.11 (uv fetches a standalone one; 3.12.x VMs fail the pinned resolve).
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r setup/requirements.txt

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
- The dev-set split (`common/dev_set.json`, question ids only) is built deterministically at
  `seed=0`, so everyone evaluates on the same 300 examples whether it is committed or regenerated by
  the setup step.
- Trained weights and adapters are **not** committed. They are published to GitHub Releases; each
  rung README links its artifact.
- Datasets are **not** redistributed. `setup/download_data.sh` fetches Spider/BIRD from source.

## Blog series

1. When should you leave the frontier API? (rungs 0-1) — _link TBD_
2. Your first owned model (rung 2a) — _link TBD_
3. When LoRA isn't enough (rungs 2b-2d) — _link TBD_
4. When fine-tuning becomes a system you operate (rung 2e) — _link TBD_
5. Owning inference: why the model isn't the whole product (serving) — _link TBD_

## Roadmap

- **BIRD track (future).** Spider is the current benchmark. BIRD (harder, real-world databases,
  needs value + domain-knowledge grounding) is planned as a separate held-out track, paired with a
  **frontier-class open-weights model (GLM 5.2)** rather than the 3B — because on hard, realistic
  queries the owned model has to be frontier-class to compete, and the point is that such models are
  *open* and therefore ownable/servable yourself. That track also exercises multi-GPU serving, so it
  doubles as the scaled-up version of the serving rungs.

## License

Code is Apache-2.0 (see [LICENSE](LICENSE)), matching the Qwen2.5 base model license. Datasets and
model weights carry their own upstream licenses.
