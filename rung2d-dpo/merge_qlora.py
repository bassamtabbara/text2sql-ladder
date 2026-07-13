"""Merge the rung-2a QLoRA adapter into a standalone checkpoint.

DPO (2d) and RL (2e) should build on our best SFT model. On this task that's the QLoRA adapter
(73.3), not the full-FT checkpoint (70.3). Serving/continuing from an adapter is awkward, so we
merge it into a plain full checkpoint first (load the base in bf16, fold in the low-rank delta).
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/qlora")
    ap.add_argument("--out", default="checkpoints/qlora-merged")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"merged {args.adapter} onto {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
