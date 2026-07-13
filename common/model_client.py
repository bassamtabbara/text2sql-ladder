"""One prompt format and one client interface, shared by every rung.

We standardize on an **OpenAI-compatible chat endpoint** for evaluation. That covers all three
serving situations without changing the eval code:
  - a frontier API (rung 0 baseline, rung 1 vendor fine-tune): point at the vendor's base_url
  - any open-weight model we train: serve it with `vllm serve ...` (exposes an OpenAI endpoint)
  - the verifier container (S5): it also speaks the OpenAI chat protocol

Keeping the prompt identical everywhere matters as much as keeping the metric identical. If the
schema serialization or the instruction changes between rungs, the numbers stop being comparable.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import re

from common.data import Example

SYSTEM_PROMPT = (
    "You are a text-to-SQL engine for SQLite. Given a database schema and a question, output a "
    "single valid SQLite query that answers it. Output only the SQL, no explanation, no markdown."
)


def build_messages(ex: Example, few_shot: list[Example] | None = None) -> list[dict]:
    """Assemble chat messages: system, optional few-shot pairs, then the target question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in few_shot or []:
        messages.append({"role": "user", "content": _user_turn(shot)})
        messages.append({"role": "assistant", "content": shot.gold_sql})
    messages.append({"role": "user", "content": _user_turn(ex)})
    return messages


def _user_turn(ex: Example) -> str:
    return f"Schema:\n{ex.schema}\n\nQuestion: {ex.question}\nSQL:"


_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of a model response.

    Handles ```sql fenced blocks, strips a trailing semicolon, and collapses whitespace. If the
    model emitted prose around the query, we take the fenced block; otherwise the whole thing.
    """
    m = _FENCE.search(text)
    sql = m.group(1) if m else text
    sql = sql.strip()
    # keep only the first statement if the model chained several
    if ";" in sql:
        sql = sql.split(";")[0]
    return " ".join(sql.split()).strip()


class ChatClient:
    """Thin wrapper over an OpenAI-compatible chat endpoint (works for OpenAI, vLLM, etc.).

    Frontier reference line (rung 0): Anthropic exposes an OpenAI-compatible endpoint, so Opus 4.8
    works here unchanged -- pass base_url="https://api.anthropic.com/v1/", model="claude-opus-4-8",
    and the Anthropic key as api_key (or via OPENAI_API_KEY).
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 512):
        from openai import OpenAI  # imported lazily so non-serving code doesn't need the dep

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # OpenAI reasoning models (o-series, gpt-5) use max_completion_tokens (not max_tokens),
        # reject temperature, and spend tokens on hidden reasoning, so they need a bigger budget.
        self._reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))
        self.client = OpenAI(
            base_url=base_url or os.environ.get("T2S_BASE_URL"),
            # vLLM ignores the key but the SDK requires a non-empty string
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    def complete(self, messages: list[dict], **overrides) -> str:
        kwargs = dict(model=self.model, messages=messages)
        max_out = overrides.get("max_tokens", self.max_tokens)
        if self._reasoning:
            # leave room for reasoning tokens on top of the short SQL output; keep effort low since
            # text-to-SQL doesn't need deep reasoning (faster + cheaper over 300 examples)
            kwargs["max_completion_tokens"] = max(max_out, 4096)
            kwargs["reasoning_effort"] = "low"
        else:
            kwargs["max_tokens"] = max_out
        # Reasoning models reject `temperature`; others get it only when set (None = model default).
        temp = overrides.get("temperature", self.temperature)
        if temp is not None and not self._reasoning:
            kwargs["temperature"] = temp
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete_batch(self, batch: list[list[dict]], workers: int = 8) -> list[str]:
        """Fan out requests concurrently. Order-preserving."""
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.complete, batch))
