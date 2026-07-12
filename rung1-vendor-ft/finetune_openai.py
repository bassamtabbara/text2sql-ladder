"""Rung 1: fine-tune a hosted frontier model through the vendor (OpenAI here).

The point of this rung is not the score. It is the *feeling* of the constraints: you upload data,
the vendor adjusts weights you never see, and the result runs only on their API. This rung is the
argument for open weights, which is why the ladder continues to rung 2.

Steps: build a small JSONL from Spider train, create a fine-tune job, wait, then evaluate the
resulting model with the same frozen eval. Requires OPENAI_API_KEY.

Note (2026): OpenAI is winding new fine-tuning down; after May 2026 new orgs can't create jobs.
That deadline is itself part of the lesson, so capture it in the post.
"""

from __future__ import annotations

import argparse
import json
import time

from common.data import load_split
from common.eval import record_result, run_eval
from common.model_client import ChatClient, SYSTEM_PROMPT, _user_turn


def build_jsonl(n: int, path: str) -> str:
    train = load_split("train")[:n]
    with open(path, "w") as f:
        for ex in train:
            rec = {"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_turn(ex)},
                {"role": "assistant", "content": ex.gold_sql},
            ]}
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {n} training examples to {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="train examples (kept small on purpose)")
    ap.add_argument("--base-model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--jsonl", default="rung1-vendor-ft/train.jsonl")
    ap.add_argument("--ft-model", default=None,
                    help="skip training and evaluate an already-finished ft:... model id")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI()

    ft_model = args.ft_model
    if ft_model is None:
        path = build_jsonl(args.n, args.jsonl)
        up = client.files.create(file=open(path, "rb"), purpose="fine-tune")
        job = client.fine_tuning.jobs.create(training_file=up.id, model=args.base_model)
        print(f"created job {job.id}; polling...")
        while True:
            job = client.fine_tuning.jobs.retrieve(job.id)
            print(f"  status={job.status}")
            if job.status in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(30)
        if job.status != "succeeded":
            raise SystemExit(f"fine-tune {job.status}")
        ft_model = job.fine_tuned_model
        print(f"fine-tuned model: {ft_model}  (note: you cannot download these weights)")

    # Evaluate against OpenAI's endpoint (base_url=None -> api.openai.com).
    from common.data import load_dev_subset

    eval_client = ChatClient(model=ft_model, base_url=None)
    metrics = run_eval(eval_client, load_dev_subset())
    record_result("1", "vendor-ft", metrics, ft_model,
                  notes=f"base={args.base_model}, n={args.n}, weights not portable")


if __name__ == "__main__":
    main()
