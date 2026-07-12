# S1 - Multi-LoRA serving (why tier-1 economics exist)

Launch vLLM with the base Qwen loaded **once** and multiple rung-2a adapters hot-swapped on top
(`--enable-lora --lora-modules a=... b=...`). Requests name the adapter they want; the base weights
are shared across all of them.

**Feel it.** One copy of the base sits in GPU memory and many tenants' tiny deltas ride on top. This
is precisely why Fireworks and Together can serve LoRA customers cheaply per-token: you did not
really hand them a new model, just a small delta on a catalog base they already had loaded.

**Lesson.** Ramp-style bounded tasks live here happily and economically. The moment you need a full
custom checkpoint instead of an adapter (rung 2b onward), this arrangement breaks. That break is S2.

Run:
```bash
bash serving/s1-multi-lora/serve.sh
```
