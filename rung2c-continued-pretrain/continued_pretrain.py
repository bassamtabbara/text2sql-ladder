"""Rung 2c (optional): continued pretraining. Keep pre-training the base on in-domain text.

This is the heaviest, most knowledge-altering step, and for Spider it is probably marginal (SQL is
not far from Qwen's world). We run a small version mostly to feel that it is skippable for most
tasks, exactly as you would advise a customer. It is only worth it with hundreds of millions of
in-domain tokens and a domain genuinely far from the base's world.

Corpus here: raw schema text plus gold SQL, concatenated as plain documents (no chat template, no
question). We are shifting the distribution, not teaching the task yet.
"""

from __future__ import annotations

import argparse

from common.data import load_split


def build_corpus():
    from datasets import Dataset

    train = load_split("train")
    docs, seen_schemas = [], set()
    for ex in train:
        if ex.db_id not in seen_schemas:
            docs.append(ex.schema)          # each schema once
            seen_schemas.add(ex.db_id)
        docs.append(ex.gold_sql)            # every gold query as raw SQL text
    return Dataset.from_dict({"text": docs})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="checkpoints/continued-pt")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        logging_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=2048,
        # in current TRL, model-load kwargs live in the config, not SFTTrainer.__init__
        model_init_kwargs={"torch_dtype": torch.bfloat16},
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.base,
        args=cfg,
        train_dataset=build_corpus(),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved continued-pretrained base to {args.out}")
    print("Now re-run 2a/2b on top of this base to see whether the domain shift bought anything.")


if __name__ == "__main__":
    main()
