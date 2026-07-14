"""Rung 2e: GRPO. Train against a real reward in a real environment.

The reward is the SQL executor: the model generates candidate queries (rollouts), each is executed
against SQLite and scored (1.0 for the right rows, 0.1 for valid-but-wrong, 0.0 for broken), and the
trainer updates weights to favor higher-reward rollouts. This is where "fine-tuning" stops being a
job you submit and becomes a generate-score-update loop you keep in sync.

Start from the 2d (or 2b) checkpoint. Watch mean reward climb over steps.
"""

from __future__ import annotations

import argparse

from common import sql_executor
from common.data import load_split
from common.model_client import build_messages, extract_sql


def build_dataset():
    from datasets import Dataset

    train = load_split("train")
    return Dataset.from_dict({
        "prompt": [build_messages(e) for e in train],   # chat-format prompt
        "gold_sql": [e.gold_sql for e in train],
        "db_path": [e.db_path for e in train],
    })


def sql_reward(completions, gold_sql, db_path, **kwargs):
    """GRPO reward function. Receives batched completions + the extra dataset columns as lists."""
    rewards = []
    for comp, gold, db in zip(completions, gold_sql, db_path):
        # chat completions arrive as a list of message dicts; take the assistant text
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        rewards.append(sql_executor.reward(extract_sql(text), gold, db))
    return rewards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/dpo")
    ap.add_argument("--out", default="checkpoints/grpo")
    ap.add_argument("--num-generations", type=int, default=4, help="rollouts per prompt (smaller = faster without vLLM)")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=3072,   # big Spider schemas live in the prompt (1536 truncated them)
        max_completion_length=384,   # SQL is short; smaller cap speeds HF-gen rollouts
        max_steps=args.steps,
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        # use_vllm=False: colocating a vLLM rollout server + the trainer on ONE gpu is the usual
        # single-GPU OOM. HF generation is slower but robust; flip on with a dedicated rollout GPU.
        use_vllm=False,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=build_dataset(),
        reward_funcs=sql_reward,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved GRPO checkpoint to {args.out}")


if __name__ == "__main__":
    main()
