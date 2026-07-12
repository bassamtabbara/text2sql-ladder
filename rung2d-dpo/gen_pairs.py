"""Rung 2d, step 1: generate preference pairs for free using the SQL executor.

Sample several candidate queries per training question from the current model, then let the
executor label them: a query that returns the gold rows is "chosen", one that doesn't is
"rejected". When several are correct, prefer the shorter one (a cheap simplicity signal). This is
the whole trick: the same executor that scores the eval also manufactures the preference data.
"""

from __future__ import annotations

import argparse
import json

from common import sql_executor
from common.data import load_split
from common.model_client import ChatClient, build_messages, extract_sql


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="full-ft", help="served model name to sample from")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--n-questions", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=6, help="candidates per question")
    ap.add_argument("--out", default="rung2d-dpo/pairs.jsonl")
    args = ap.parse_args()

    client = ChatClient(model=args.model, base_url=args.base_url, temperature=0.9)
    train = load_split("train")[: args.n_questions]

    n_pairs = 0
    with open(args.out, "w") as f:
        for ex in train:
            messages = build_messages(ex)
            cands = [extract_sql(client.complete(messages)) for _ in range(args.samples)]
            correct, wrong = [], []
            for sql in cands:
                if sql_executor.results_match(sql, ex.gold_sql, ex.db_path):
                    correct.append(sql)
                elif sql_executor.is_valid_sql(sql, ex.db_path) is not None:
                    wrong.append(sql)
            if not correct or not wrong:
                continue  # need both a winner and a loser to form a pair
            chosen = min(correct, key=len)      # prefer the simplest correct query
            rejected = wrong[0]
            prompt = messages  # DPOTrainer accepts chat-format prompt
            f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}) + "\n")
            n_pairs += 1

    print(f"wrote {n_pairs} preference pairs to {args.out}")


if __name__ == "__main__":
    main()
