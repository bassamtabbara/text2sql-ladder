# Rung 2b - Full fine-tune (the higher ceiling, heavier artifact)

**First principles.** Unfreeze every parameter and keep training on your data. Highest quality
ceiling, learns the harder tasks LoRA can't, no adapter indirection. The costs: enough GPU memory to
train the whole model, and catastrophic forgetting, which you mitigate with a low learning rate and
by mixing in some general data.

**What you do.** Full-parameter SFT on the same Spider subset (`--lr 1e-5`, optional
`--mix-general`), producing a complete new checkpoint the size of the base. Then serve it as a
dedicated model in vLLM and evaluate.

**What to watch.** EX on the hard BIRD slice closes measurably versus LoRA. Probe forgetting by
asking a couple of non-SQL questions and seeing whether general ability degraded. Put the LoRA and
full-FT numbers side by side: this is the "LoRA recovers ~90-95% of full fine-tuning" intuition made
concrete in your own table.

**Customer lesson.** You now hold weights nobody else has, and a full checkpoint has no catalog base
to attach to, so multi-tenant per-token economics stop applying. This is the first artifact that
forces dedicated serving, which is where bring-your-own-container begins (serving/s2).

**Artifact.** A full checkpoint (~base-model size). Published to Releases, not committed.

Run:
```bash
bash rung2b-full-ft/run.sh
```
