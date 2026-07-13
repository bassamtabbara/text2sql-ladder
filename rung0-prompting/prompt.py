"""Rung 0: change behavior with the context window only, no weights touched.

Three sub-experiments against the *same* base model served by vLLM:
  zero-shot   - just the schema + question
  few-shot    - prepend a few fixed question->SQL examples
  rag         - retrieve the nearest training examples to each question and prepend those

Run each and watch EX climb, then flatten. That plateau is the reason to consider climbing.
"""

from __future__ import annotations

import argparse
import functools

from common.data import Example, load_split
from common.eval import record_result, run_eval
from common.model_client import ChatClient


def fixed_few_shot(k: int) -> list[Example]:
    """A small, fixed set of shots reused for every question (cheap, no retrieval)."""
    train = load_split("train")
    return train[:k]


class RagRetriever:
    """Embed training questions once; retrieve the k nearest for each dev question."""

    def __init__(self, k: int = 5):
        from sentence_transformers import SentenceTransformer

        self.k = k
        self.train = load_split("train")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.emb = self.model.encode(
            [e.question for e in self.train], convert_to_numpy=True, normalize_embeddings=True
        )

    def __call__(self, ex: Example) -> list[Example]:
        import numpy as np

        q = self.model.encode([ex.question], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.emb @ q
        top = np.argsort(-sims)[: self.k]
        return [self.train[i] for i in top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero-shot", "few-shot", "rag"], required=True)
    ap.add_argument("--k", type=int, default=5, help="number of shots for few-shot / rag")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--label", default=None,
                    help="override the technique label (e.g. frontier-opus4.8); defaults to --mode")
    ap.add_argument("--rung", default="0",
                    help="rung id to record under (e.g. 2a to eval a fine-tuned model with RAG)")
    args = ap.parse_args()

    if args.mode == "zero-shot":
        few_shot_fn = None
    elif args.mode == "few-shot":
        shots = fixed_few_shot(args.k)
        few_shot_fn = lambda _ex: shots  # noqa: E731 - deliberately identical shots per question
    else:  # rag
        few_shot_fn = RagRetriever(k=args.k)

    from common.data import load_dev_subset

    # Greedy (temperature=0) for models that support it, so numbers are deterministic. Reasoning
    # models (claude-*, OpenAI o-series, gpt-5) reject temperature, so omit it and take the default.
    no_temp = args.model.startswith(("claude", "o1", "o3", "o4", "gpt-5"))
    temperature = None if no_temp else 0.0
    client = ChatClient(model=args.model, base_url=args.base_url, temperature=temperature)
    metrics = run_eval(client, load_dev_subset(), few_shot_fn=few_shot_fn)
    technique = args.label or args.mode
    record_result(args.rung, technique, metrics, args.model,
                  notes=f"k={args.k}" if args.mode != "zero-shot" else "")


if __name__ == "__main__":
    main()
