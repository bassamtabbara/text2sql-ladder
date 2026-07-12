"""S5: the verifier pipeline. Model + execution + repair as one served unit.

The product here is not the checkpoint alone. It is: generate a query, EXECUTE it against the
database, and if it errors, feed the error back to the model and let it repair, up to a few times.
That loop needs the model, a runtime, and database access in the request path, which is why it
lives in a container you control rather than behind a per-token API.

This module is both:
  - a library (`generate_with_repair`) used by eval_verifier.py to score the pipeline, and
  - a FastAPI app (`app`) that packages the same logic as a deployable endpoint (see Dockerfile).
"""

from __future__ import annotations

import os

from common import sql_executor
from common.model_client import ChatClient, extract_sql

REPAIR_SYSTEM = (
    "You are a SQLite expert. The previous query failed to execute. Given the schema, the question, "
    "the failed query, and the database error, output a corrected single SQLite query. Output only "
    "the SQL."
)


def generate_with_repair(client: ChatClient, schema: str, question: str, db_path: str,
                         max_repairs: int = 2) -> str:
    """Generate SQL, execute it, and repair on error up to max_repairs times."""
    from common.model_client import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"},
    ]
    sql = extract_sql(client.complete(messages))

    for _ in range(max_repairs):
        res = sql_executor.execute(sql, db_path)
        if res.ok:
            return sql  # runs; hand it back (correctness is still judged by the eval)
        repair_msgs = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": (
                f"Schema:\n{schema}\n\nQuestion: {question}\n"
                f"Failed query: {sql}\nError: {res.error}\nCorrected SQL:")},
        ]
        sql = extract_sql(client.complete(repair_msgs))
    return sql


# ---- Deployable container endpoint -------------------------------------------------------------
# A minimal HTTP surface so the pipeline can be built into an image and run on dedicated GPUs.
# Databases are mounted into the container; the request names the db_id to execute against.
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="text2sql-verifier")

    class Req(BaseModel):
        schema_text: str
        question: str
        db_id: str

    def _client() -> ChatClient:
        # points at the model server co-located in the container / pod
        return ChatClient(model=os.environ.get("MODEL_NAME", "dedicated"),
                          base_url=os.environ.get("MODEL_URL", "http://localhost:8001/v1"))

    def _db_path(db_id: str) -> str:
        root = os.environ.get("T2S_DATA_ROOT", "data/spider")
        return os.path.join(root, "database", db_id, f"{db_id}.sqlite")

    @app.post("/generate")
    def generate(req: Req) -> dict:
        sql = generate_with_repair(_client(), req.schema_text, req.question, _db_path(req.db_id))
        return {"sql": sql}
except ImportError:
    # fastapi/pydantic not installed in a training-only env; the library path still works.
    app = None
