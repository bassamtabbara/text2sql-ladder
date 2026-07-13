# Rung 0 - Frontier API, prompt only (the baseline)

**Concepts (from first principles).**
- *Prompting* changes the model's behavior without changing any weights: the only lever is what you
  put in the context window.
- *Zero-shot*: give the model just the instruction, the schema, and the question, and ask for SQL.
  No examples.
- *Few-shot*: prepend a handful of worked question->SQL examples so the model can pattern-match the
  format and dialect. The examples are the same for every question.
- *RAG (retrieval-augmented generation)*: instead of fixed examples, retrieve the most similar
  training questions (plus the relevant schema) for *each* question and paste those in, so the model
  sees the right facts (correct table/column names) at answer time.
- *Execution accuracy (EX)*: the main metric -- run the predicted SQL and the gold SQL against the
  database and check they return the same rows. Right answer, not right string.
- *Valid-SQL rate*: the fraction of predictions that execute at all (no error), separate from
  whether they're correct.

**First principles.** You change the model's *behavior* without changing any *weights*. The only
lever is the context window: instructions, a few worked examples, and retrieval (RAG) to hand the
model the right facts (here, the exact table and column names). This is renting a model and steering
it from the outside.

**What you do.** Serve the untouched `Qwen2.5-3B-Instruct` with vLLM, then run the same eval three
ways: zero-shot, few-shot (a few fixed question->SQL pairs), and RAG (retrieve the nearest training
questions to each dev question and prepend those). `run.sh` does all three against the base model,
then records a **frontier panel**: the same few-shot+RAG eval against each rented frontier model
whose key is set -- `gpt-5.6-sol` (OpenAI), `claude-opus-4-8` (Anthropic), and `gemini-3.5-flash`
(Google). Override ids with `FRONTIER_OPENAI_MODEL` / `FRONTIER_ANTHROPIC_MODEL` /
`FRONTIER_GEMINI_MODEL`.

**Two baselines, on purpose.** Rung 0 records two different kinds of number:

- **Base Qwen2.5-3B (zero/few/rag)** is the "before" for the fine-tuning arc. Rungs 2a-2e are
  measured against *this* line, so the comparison is apples-to-apples. If you instead measured the
  fine-tuned 3B against a frontier model, a "gain" would just be a model-size and family difference,
  not evidence that fine-tuning worked.
- **Frontier panel (rented models, few-shot+RAG)** is "rent the best and prompt it well" -- the
  ceiling the small owned model chases, run across several vendors so it's clear the ceiling isn't a
  single number. It is a reference, not the fine-tuning baseline. The compelling story is not "our
  3B beats the frontier" (it won't on raw EX); it is how close a cheap model you own can get on
  *this* task, and what owning it then lets you do (serve it cheaply, add constrained decoding, add
  a verifier). Gemini Flash appears here untuned and again in rung 1 tuned, so that pair is a clean
  within-model view of what fine-tuning adds.

Each frontier row is just a labeled invocation of the same frozen `eval.py`, no new code.

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
