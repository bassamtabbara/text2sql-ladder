"""S5 eval: score the verifier pipeline end to end.

Unlike the other rungs, the prediction path includes executing SQL and repairing it, which needs
database access. So this uses the frozen sql_executor + record_result for scoring, but drives the
pipeline (generate -> execute -> repair) instead of a single model call. That difference IS the
rung: the served unit is the pipeline, not just the weights.
"""

from __future__ import annotations

import argparse
import statistics
import time

from common import sql_executor
from common.data import load_dev_subset
from common.eval import record_result
from common.model_client import ChatClient
from pipeline import generate_with_repair  # sibling module (run this script from its own dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dedicated")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-repairs", type=int, default=2)
    args = ap.parse_args()

    client = ChatClient(model=args.model, base_url=args.base_url)
    examples = load_dev_subset()

    matches, valids, latencies = 0, 0, []
    for ex in examples:
        t0 = time.perf_counter()
        sql = generate_with_repair(client, ex.schema, ex.question, ex.db_path, args.max_repairs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        valid = sql_executor.is_valid_sql(sql, ex.db_path)
        valids += valid
        matches += sql_executor.results_match(sql, ex.gold_sql, ex.db_path) if valid else 0

    n = len(examples)
    metrics = {"ex": matches / n, "valid_sql": valids / n, "n": n,
               "p50_latency_ms": statistics.median(latencies)}
    print(f"n={n}  EX={metrics['ex']:.1%}  valid-SQL={metrics['valid_sql']:.1%}  "
          f"p50={metrics['p50_latency_ms']:.0f}ms")
    record_result("S5", "verifier-pipeline", metrics, args.model,
                  notes=f"generate+execute+repair (max_repairs={args.max_repairs})")


if __name__ == "__main__":
    main()
