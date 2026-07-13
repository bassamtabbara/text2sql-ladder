"""Rung 2b: full fine-tune. Unfreeze every parameter and keep training.

Highest quality ceiling and it learns the harder queries LoRA can't, at the cost of a full new
checkpoint the size of the base and a real catastrophic-forgetting risk. We mitigate forgetting the
standard way: a low learning rate and an optional splash of general instruction data mixed in.
"""

from __future__ import annotations

import argparse

from common.data import load_split


def build_dataset(tokenizer, mix_general: int):
    from datasets import Dataset

    from common.model_client import SYSTEM_PROMPT, _user_turn

    train = load_split("train")

    def to_text_sql(ex):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_turn(ex)},
            {"role": "assistant", "content": ex.gold_sql},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    texts = [to_text_sql(e) for e in train]

    # Optional: mix in some general instruction data to blunt catastrophic forgetting. Off by
    # default; flip on to probe how much forgetting the low LR alone is already preventing.
    if mix_general > 0:
        from datasets import load_dataset

        gen = load_dataset("HuggingFaceH4/no_robots", split="train").shuffle(seed=0)
        for row in gen.select(range(min(mix_general, len(gen)))):
            texts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False))

    return Dataset.from_dict({"text": texts})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="checkpoints/full-ft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5, help="low, to limit forgetting")
    ap.add_argument("--mix-general", type=int, default=0, help="# general examples to mix in")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(tokenizer, args.mix_general)

    # completion-only loss: train on the SQL, not the prompt (Qwen2.5's assistant turn marker)
    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,   # small micro-batch: full-FT + the seq-4096 logits spike
        gradient_accumulation_steps=8,   # effective batch stays 16
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=4096,   # avoid truncating large Spider schemas (would drop the SQL)
        save_strategy="epoch",
        # in current TRL, model-load kwargs live in the config, not SFTTrainer.__init__
        model_init_kwargs={"torch_dtype": torch.bfloat16},
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.base,
        args=cfg,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved full checkpoint to {args.out}")


if __name__ == "__main__":
    main()
