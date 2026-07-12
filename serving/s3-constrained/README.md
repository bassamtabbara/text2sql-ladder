# S3 - Constrained / grammar decoding (serving logic a per-token API won't run)

Turn on vLLM guided decoding with a SQL grammar (`sqlite_select.gbnf`) so the model can only emit
strings the grammar accepts. The checkpoint is byte-for-byte the same one S2 served; only the
decoding changed.

**What to watch.** Valid-SQL rate jumps toward 100%, and EX often rises for free, with no weight
change. The grammar guarantees the query parses; the model still has to pick the right columns.

**Lesson.** This is value that lives in the container, not the checkpoint. A managed per-token API
gives you a text stream; it will not run your grammar in the decode loop. That is a concrete reason
the serving path, not just the weights, has to be yours.

Run:
```bash
bash serving/s3-constrained/serve.sh
```
