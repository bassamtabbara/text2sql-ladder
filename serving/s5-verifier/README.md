# S5 - The verifier pipeline as one served unit (the definition of tier 2)

Wrap the model in a loop: generate a query, **execute it** against the database, and if it errors,
feed the error back to the model and let it repair, up to a few times. Package the model runtime,
this logic, and database access into a container (see `Dockerfile`) and serve it. `serve.sh` scores
the whole pipeline with the frozen executor.

**What to watch.** EX rises again from the self-repair loop (many first-try failures are one fix
away). And notice what the artifact has become: the product is the pipeline plus the custom weights
together, not a checkpoint you could upload to a per-token API.

**Lesson.** This is tier 2 in one sentence: you control the container, not just the checkpoint. It
is the reason customers climb here, often not for cost but because their model or serving path is no
longer something a managed API can express. It is also the surface Modelplane is built to run.

**Note on the eval.** This rung uses its own `eval_verifier.py` rather than the shared endpoint
eval, because the prediction path itself executes SQL and needs database access. It still scores
with the same frozen `sql_executor`, so the EX number stays comparable to every other rung.

Run:
```bash
bash serving/s5-verifier/serve.sh
# or build the container:
# docker build -f serving/s5-verifier/Dockerfile -t text2sql-verifier .
```
