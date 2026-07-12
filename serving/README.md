# Serving: why the heavy rungs force bring-your-own-container

Once you own a checkpoint, *where* you can run it stops being a footnote. These five steps walk the
boundary Modelplane sits on, using the same frozen eval (now also watching latency):

| Step | What it shows | Weight change? |
|------|---------------|:--------------:|
| [S1](s1-multi-lora/) | multi-LoRA: one base, many adapters, cheap per-token economics | no |
| [S2](s2-dedicated/) | a full checkpoint needs its own instance; tier-1 economics break | no |
| [S3](s3-constrained/) | grammar-constrained decoding drives valid-SQL up | no |
| [S4](s4-speculative/) | speculative decoding cuts latency ~2-3x | no |
| [S5](s5-verifier/) | generate->execute->repair pipeline, packaged as a container | no |

The through-line: S3-S5 add real value that lives in the *container*, not the checkpoint. A managed
per-token API can host a checkpoint; it cannot run your grammar, your draft model, or your
execute-and-repair loop. When your product is the pipeline plus the weights, you need to control the
container. That is tier 2, and it is what these steps let you feel.
