"""The frozen evaluation harness. Write once, reuse unchanged everywhere.

Given any OpenAI-compatible endpoint, this measures the two numbers we carry across the ladder:
  - EX (execution accuracy): fraction of dev questions whose predicted SQL returns the gold rows
  - valid-SQL rate: fraction whose predicted SQL executes at all

plus median request latency. It appends one row to results/results.csv so every rung lands in the
same table.

Usage (the common case, evaluating a served model zero-shot):

    python -m common.eval --rung 2a --technique qlora \
        --model qwen2.5-3b-qlora --base-url http://localhost:8000/v1

Rungs that need in-context examples (rung 0 few-shot / RAG) import `run_eval` and pass a
`few_shot_fn`; see rung0-prompting/prompt.py.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

from common import sql_executor
from common.data import Example, load_dev_subset
from common.model_client import ChatClient, build_messages, extract_sql

FewShotFn = Callable[[Example], list[Example]]


def run_eval(
    client: ChatClient,
    examples: list[Example],
    few_shot_fn: FewShotFn | None = None,
    workers: int = 8,
    verbose: bool = True,
) -> dict:
    """Evaluate a client over examples. Returns {ex, valid_sql, n, p50_latency_ms, errors}."""

    def _one(ex: Example) -> dict:
        shots = few_shot_fn(ex) if few_shot_fn else None
        messages = build_messages(ex, few_shot=shots)
        t0 = time.perf_counter()
        raw = client.complete(messages)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred = extract_sql(raw)
        valid = sql_executor.is_valid_sql(pred, ex.db_path)
        match = sql_executor.results_match(pred, ex.gold_sql, ex.db_path) if valid else False
        return {"latency_ms": latency_ms, "valid": valid, "match": match,
                "pred": pred, "db": ex.db_id}

    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one, examples))

    n = len(rows)
    ex_acc = sum(r["match"] for r in rows) / n if n else 0.0
    valid_rate = sum(r["valid"] for r in rows) / n if n else 0.0
    p50 = statistics.median(r["latency_ms"] for r in rows) if n else 0.0
    errors = [r for r in rows if not r["match"]]

    if verbose:
        print(f"n={n}  EX={ex_acc:.1%}  valid-SQL={valid_rate:.1%}  p50={p50:.0f}ms")
    return {"ex": ex_acc, "valid_sql": valid_rate, "n": n,
            "p50_latency_ms": p50, "errors": errors}


def record_result(rung: str, technique: str, metrics: dict, model: str,
                  notes: str = "", csv_path: str = "results/results.csv") -> None:
    """Append one comparable row to the shared results table."""
    import csv
    import datetime
    import os

    header = ["rung", "technique", "ex", "valid_sql", "n",
              "p50_latency_ms", "model", "notes", "run_date"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            rung, technique,
            f"{metrics['ex']:.4f}", f"{metrics['valid_sql']:.4f}", metrics["n"],
            f"{metrics['p50_latency_ms']:.0f}", model, notes,
            datetime.date.today().isoformat(),
        ])
    print(f"recorded rung {rung} ({technique}) -> {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a served text-to-SQL model (zero-shot).")
    ap.add_argument("--rung", required=True, help="ladder rung id, e.g. 2a")
    ap.add_argument("--technique", required=True, help="short label, e.g. qlora")
    ap.add_argument("--model", required=True, help="model name the endpoint expects")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base url (vLLM: .../v1)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N (debug)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--no-record", action="store_true", help="print but don't write results.csv")
    args = ap.parse_args()

    examples = load_dev_subset()
    if args.limit:
        examples = examples[: args.limit]

    client = ChatClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    metrics = run_eval(client, examples)
    if not args.no_record:
        record_result(args.rung, args.technique, metrics, args.model, args.notes)


if __name__ == "__main__":
    main()
