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

    from common.data import load_dev_subset

    client = OpenAI()
    dev = load_dev_subset()

    # Fair within-model baseline: the SAME base model, prompted zero-shot, before any fine-tuning.
    # The lift from this row to the fine-tuned row is the honest "what vendor fine-tuning buys"
    # number -- same model, same vendor, same inference, so it isolates the effect of tuning.
    base_client = ChatClient(model=args.base_model, base_url=None)
    base_metrics = run_eval(base_client, dev)
    record_result("1", "base-mini", base_metrics, args.base_model, notes="zero-shot, no fine-tune")

    ft_model = args.ft_model
    if ft_model is None:
        try:
            path = build_jsonl(args.n, args.jsonl)
            up = client.files.create(file=open(path, "rb"), purpose="fine-tune")
            job = client.fine_tuning.jobs.create(training_file=up.id, model=args.base_model)
        except Exception as exc:  # noqa: BLE001
            # The rented-customization door. Some vendors have wound fine-tuning down (OpenAI stopped
            # letting new orgs create jobs after May 2026). This refusal IS rung 1's lesson, not a
            # bug: your ability to customize a hosted model can simply be revoked. The base-mini
            # baseline above is already recorded, so there is nothing more to run here.
            print("\n=== Rung 1: vendor fine-tuning unavailable ===")
            print(f"{type(exc).__name__}: {exc}")
            print("This refusal is the lesson. It is the argument for open weights (rung 2). See "
                  "rung1-vendor-ft/README.md for managed-FT alternatives if you want a lift number.")
            return
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

    # Evaluate the fine-tuned model against OpenAI's endpoint (base_url=None -> api.openai.com),
    # zero-shot -- same inference as the base-mini baseline above, so the delta is the FT lift.
    eval_client = ChatClient(model=ft_model, base_url=None)
    metrics = run_eval(eval_client, dev)
    record_result("1", "vendor-ft", metrics, ft_model,
                  notes=f"base={args.base_model}, n={args.n}, weights not portable")


if __name__ == "__main__":
    main()
