"""S3: evaluate with grammar-constrained decoding.

Same frozen eval, but every request carries a GBNF grammar so vLLM can only emit strings the
grammar accepts. This is serving logic, not a model change: the checkpoint is identical to the one
S2 served. Watch valid-SQL rate jump toward 100%, and EX often rise for free.
"""

from __future__ import annotations

import argparse

from common.data import load_dev_subset
from common.eval import record_result, run_eval
from common.model_client import ChatClient


class ConstrainedClient(ChatClient):
    """ChatClient that attaches a GBNF grammar to each request via vLLM's extra_body."""

    def __init__(self, *a, grammar: str, **kw):
        super().__init__(*a, **kw)
        self._grammar = grammar

    def complete(self, messages, **overrides):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=overrides.get("temperature", self.temperature),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            extra_body={"guided_grammar": self._grammar},
        )
        return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dedicated")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--grammar", default="serving/s3-constrained/sqlite_select.gbnf")
    args = ap.parse_args()

    with open(args.grammar) as f:
        grammar = f.read()

    client = ConstrainedClient(model=args.model, base_url=args.base_url, grammar=grammar)
    metrics = run_eval(client, load_dev_subset())
    record_result("S3", "constrained-decode", metrics, args.model,
                  notes="GBNF-guided decoding, same weights as S2")


if __name__ == "__main__":
    main()
