"""Rung 2a: QLoRA. Freeze the base, learn a tiny low-rank adapter (~0.1-1% of params).

Output is a small adapter directory that sits on top of the base model. On an H100 this is a
30-60 minute run. The adapter is what makes tier-1 multi-tenant serving possible (see serving/s1).
"""

from __future__ import annotations

import argparse

from common.data import load_split


def build_dataset(tokenizer):
    from datasets import Dataset

    from common.model_client import SYSTEM_PROMPT, _user_turn

    train = load_split("train")

    def to_text(ex):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_turn(ex)},
            {"role": "assistant", "content": ex.gold_sql},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    return Dataset.from_dict({"text": [to_text(e) for e in train]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="outputs/qlora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(tokenizer)

    # Completion-only loss: mask everything before the assistant turn so the model learns to
    # GENERATE the SQL, not reproduce the schema/question. Qwen2.5's assistant turn opens with this.
    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=2048,
        # in current TRL, model-load kwargs live in the config, not SFTTrainer.__init__
        model_init_kwargs={"quantization_config": bnb, "torch_dtype": torch.bfloat16},
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.base,
        args=cfg,
        train_dataset=dataset,
        peft_config=lora,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
