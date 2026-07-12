"""Dataset loading and schema serialization for Spider (and BIRD, same shape).

Produces a flat list of `Example` records that every rung consumes identically: the question, the
database it targets, the path to that database's SQLite file, the gold SQL, and a plain-text
serialization of the schema to drop into the prompt.

Expected on-disk layout after `setup/download_data.sh` (standard Spider release):

    data/spider/
      train_spider.json
      dev.json
      tables.json
      database/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

DATA_ROOT = os.environ.get("T2S_DATA_ROOT", "data/spider")


@dataclass
class Example:
    question: str
    db_id: str
    db_path: str
    gold_sql: str
    schema: str  # human/model-readable schema string


@lru_cache(maxsize=1)
def _tables_by_db(tables_path: str) -> dict:
    with open(tables_path) as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def serialize_schema(db_id: str, tables_path: str) -> str:
    """Render a db's schema as CREATE-TABLE-like text: readable to a model, cheap on tokens."""
    t = _tables_by_db(tables_path)[db_id]
    table_names = t["table_names_original"]
    cols = t["column_names_original"]  # list of [table_idx, col_name]
    col_types = t["column_types"]
    pks = set(t.get("primary_keys", []))

    per_table: dict[int, list[str]] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(cols):
        if tbl_idx == -1:
            continue  # the synthetic '*' column
        marker = " PRIMARY KEY" if col_idx in pks else ""
        per_table[tbl_idx].append(f"  {col_name} {col_types[col_idx]}{marker}")

    lines = []
    for i, name in enumerate(table_names):
        body = ",\n".join(per_table[i])
        lines.append(f"CREATE TABLE {name} (\n{body}\n);")

    # foreign keys, rendered as comments so the model sees join paths without SQL-dialect noise
    fks = t.get("foreign_keys", [])
    if fks:
        lines.append("-- foreign keys:")
        for a, b in fks:
            ta, ca = cols[a]
            tb, cb = cols[b]
            lines.append(f"--   {table_names[ta]}.{ca} -> {table_names[tb]}.{cb}")
    return "\n".join(lines)


def _db_path(db_id: str) -> str:
    return os.path.join(DATA_ROOT, "database", db_id, f"{db_id}.sqlite")


def load_split(split: str) -> list[Example]:
    """Load 'train' or 'dev'. Returns Example records with absolute-ish db paths."""
    fname = {"train": "train_spider.json", "dev": "dev.json"}[split]
    path = os.path.join(DATA_ROOT, fname)
    tables_path = os.path.join(DATA_ROOT, "tables.json")
    with open(path) as f:
        raw = json.load(f)
    out = []
    for r in raw:
        db_id = r["db_id"]
        out.append(
            Example(
                question=r["question"].strip(),
                db_id=db_id,
                db_path=_db_path(db_id),
                gold_sql=r["query"].strip(),
                schema=serialize_schema(db_id, tables_path),
            )
        )
    return out


def load_dev_subset(manifest_path: str = "common/dev_set.json") -> list[Example]:
    """Load the FROZEN dev subset used for every rung's number.

    The manifest is a list of integer indices into the full dev split. Freezing it (and committing
    it) is what keeps every rung's EX directly comparable.
    """
    dev = load_split("dev")
    with open(manifest_path) as f:
        idxs = json.load(f)["indices"]
    return [dev[i] for i in idxs]


def build_dev_manifest(n: int = 300, seed: int = 0, out: str = "common/dev_set.json") -> None:
    """Create the frozen dev manifest once. Run this a single time, then commit the output."""
    import random

    dev = load_split("dev")
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(dev)), min(n, len(dev))))
    with open(out, "w") as f:
        json.dump({"seed": seed, "n": len(idxs), "indices": idxs}, f, indent=2)
    print(f"wrote {out} with {len(idxs)} indices (seed={seed})")


if __name__ == "__main__":
    # Convenience: `python -m common.data build` to (re)generate the frozen dev manifest.
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_dev_manifest()
    else:
        print(f"DATA_ROOT={DATA_ROOT}")
        print("Pass 'build' to regenerate the frozen dev manifest (do this once, then commit it).")
