"""Rung 2d, step 2: DPO on the executor-labeled preference pairs.

DPO learns directly from (chosen, rejected) pairs, no separate reward model. Start from the 2b
checkpoint so we are shaping an already-competent model toward "better", not just "correct".
"""

from __future__ import annotations

import argparse
import json


def load_pairs(path: str):
    from datasets import Dataset

    prompts, chosen, rejected = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prompts.append(r["prompt"])
            chosen.append(r["chosen"])
            rejected.append(r["rejected"])
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/qlora-merged")
    ap.add_argument("--pairs", default="rung2d-dpo/pairs.jsonl")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        num_train_epochs=1,
        per_device_train_batch_size=2,   # DPO holds policy + reference model, so keep it small
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        max_length=4096,                 # big Spider schemas live in the prompt
        report_to="none",
    )
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=load_pairs(args.pairs),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved DPO checkpoint to {args.out}")


if __name__ == "__main__":
    main()
