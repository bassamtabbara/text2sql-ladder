# Rung 0 - Frontier API, prompt only (the baseline)

**First principles.** You change the model's *behavior* without changing any *weights*. The only
lever is the context window: instructions, a few worked examples, and retrieval (RAG) to hand the
model the right facts (here, the exact table and column names). This is renting a model and steering
it from the outside.

**What you do.** Serve the untouched `Qwen2.5-3B-Instruct` with vLLM, then run the same eval three
ways: zero-shot, few-shot (a few fixed question->SQL pairs), and RAG (retrieve the nearest training
questions to each dev question and prepend those). `run.sh` does all three.

**What to watch.** EX climbs from zero-shot to few-shot to RAG, then flattens. RAG helps most with
the *facts* problem: the base model guesses plausible-but-wrong column names, and showing it real
examples from the target database fixes many of those. Note the exact point where more prompting
stops helping. That plateau is the whole reason the ladder exists.

**Customer lesson.** Most workloads should stay here. You only climb down when prompting and RAG
have visibly plateaued on a task that matters enough to justify owning a model. Feeling that plateau
yourself is the point of this rung.

**Artifact.** None. You produced nothing you own; it can run on any per-token API.

Run:
```bash
bash rung0-prompting/run.sh
```
