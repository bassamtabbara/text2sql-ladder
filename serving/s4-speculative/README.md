# S4 - Speculative decoding (custom serving for speed)

Add speculation to the same checkpoint and measure tokens/sec before and after. This script uses
n-gram self-speculation (no extra model) to make the point cheaply. The production version trains a
small draft model or extra multi-token-prediction heads, which is exactly what Cursor did to get
2-3x faster inference.

**What to watch.** EX and valid-SQL are unchanged (same weights). p50 latency drops, typically
2-3x. Compare the p50 recorded here against S2's p50.

**Lesson.** This is serving logic a managed per-token API will not run for you. Your latency, and
the draft model behind it, are part of the product, and they only exist because you control the
serving path.

Run:
```bash
bash serving/s4-speculative/serve.sh
```
