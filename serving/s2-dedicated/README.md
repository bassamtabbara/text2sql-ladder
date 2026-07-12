# S2 - Dedicated full checkpoint (tier-1 breaks)

Serve the rung-2b full checkpoint in vLLM. Note what changed from S1: there is no catalog base to
share, so this checkpoint needs its own dedicated GPU memory and its own instance.

**Lesson.** This is the exact moment multi-tenant per-token economics stop applying. You have
graduated to bring-your-own-container on dedicated GPUs. Everything from here down (S3-S5) is value
that lives in *your* container, not in a checkpoint a per-token API could host for you.

Run:
```bash
bash serving/s2-dedicated/serve.sh
```
