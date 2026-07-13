# Rung 0 - Frontier API, prompt only (the baseline)

**First principles.** You change the model's *behavior* without changing any *weights*. The only
lever is the context window: instructions, a few worked examples, and retrieval (RAG) to hand the
model the right facts (here, the exact table and column names). This is renting a model and steering
it from the outside.

**What you do.** Serve the untouched `Qwen2.5-3B-Instruct` with vLLM, then run the same eval three
ways: zero-shot, few-shot (a few fixed question->SQL pairs), and RAG (retrieve the nearest training
questions to each dev question and prepend those). `run.sh` does all three, and, if
`OPENAI_API_KEY` is set, also records a frontier reference line (a strong OpenAI model, few-shot+RAG;
default `gpt-5`, override with `FRONTIER_MODEL`).

**Two baselines, on purpose.** Rung 0 records two different kinds of number:

- **Base Qwen2.5-3B (zero/few/rag)** is the "before" for the fine-tuning arc. Rungs 2a-2e are
  measured against *this* line, so the comparison is apples-to-apples. If you instead measured the
  fine-tuned 3B against a frontier model, a "gain" would just be a model-size and family difference,
  not evidence that fine-tuning worked.
- **Frontier reference (a strong OpenAI model, few-shot+RAG)** is "rent the best model and prompt it
  well" -- the number a real customer starts from and the ceiling the small owned model chases. It
  is a reference, not the fine-tuning baseline. The compelling story is not "our 3B beats the
  frontier" (it likely won't on raw EX); it is how close a cheap model you own can get on *this*
  task, and what owning it then lets you do (serve it cheaply, add constrained decoding, add a
  verifier). Kept on OpenAI so it matches rung 1's vendor: one vendor, apples to apples.

The frontier line is just a labeled invocation of the same frozen `eval.py`, no new code.

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
