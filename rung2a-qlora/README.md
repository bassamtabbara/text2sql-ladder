# Rung 2a - Open-weight QLoRA (the pivotal, cheap, additive move)

**First principles.** Freeze all of the base model's weights and learn a tiny pair of low-rank
matrices that represent the difference between the base and your tuned version. Trainable parameters
drop to roughly 0.1-1% of the total. The output is a small adapter file that sits on top of the
base. QLoRA quantizes the frozen base to 4-bit so it trains on modest hardware.

**What you do.** Train a rank-16 adapter on the 2k-example Spider subset (~30-60 min on the H100),
then serve the base with the adapter loaded on top in vLLM and evaluate.

**What to watch.** EX jumps past the rung-0 RAG plateau, and valid-SQL rate and dialect/format
adherence especially improve. LoRA is excellent at style, format, and bounded structure. On BIRD's
harder joins and nested queries you will still see a residual gap. That gap is the point, and it is
what rung 2b goes after.

**Customer lesson.** This is where "custom model" begins, and it is cheap and reversible. It is also
the boundary of tier-1 serving economics: a tiny adapter on a catalog base can be hot-swapped
per-tenant (serving/s1), which is why platforms can serve it cheaply per-token.

**Artifact.** A tens-of-MB adapter. Published to Releases, not committed.

Run:
```bash
bash rung2a-qlora/run.sh
```
