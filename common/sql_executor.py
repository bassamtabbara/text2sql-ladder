"""Execute SQL against a SQLite database and compare result sets.

This is the heart of the whole ladder. Execution accuracy (does the predicted query return the
same rows as the gold query?) is the single metric we carry across every rung, and it is also the
reward signal for the RL rung. So it lives here, once, and everything imports it.

Design choices worth knowing:
- Comparison is by *denotation* (the returned rows), not by SQL string. Two different queries that
  return the same rows count as correct. This is the standard text-to-SQL notion of correctness.
- Row order is ignored unless the gold query contains ORDER BY, in which case order matters.
- Column order is ignored (we compare each row as a sorted multiset of cell values), because models
  legitimately reorder SELECT columns. This is the common, slightly lenient convention.
- Every query runs under a timeout so a pathological prediction can't hang a training run.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool                 # did the query execute without error?
    rows: list | None        # returned rows (list of tuples) if ok, else None
    error: str | None        # error message if not ok


def _normalize_cell(value):
    """Make cells comparable across trivial type differences (1 vs 1.0, ' x ' vs 'x')."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_rows(rows, order_matters: bool):
    """Normalize a result set for comparison.

    Each row's cells are sorted (column-order-insensitive). The set of rows is sorted too unless
    order_matters, in which case we keep the returned order.
    """
    def key(cell):
        # sort mixed types deterministically by (type-name, str-value)
        return (type(cell).__name__, str(cell))

    norm = [tuple(sorted((_normalize_cell(c) for c in row), key=key)) for row in rows]
    if not order_matters:
        norm = sorted(norm, key=lambda r: tuple(key(c) for c in r))
    return norm


def execute(sql: str, db_path: str, timeout_s: float = 15.0) -> ExecResult:
    """Run one SQL statement against db_path, returning rows or an error, under a hard timeout."""
    conn = sqlite3.connect(db_path)
    # tolerate the odd non-UTF8 byte string found in some Spider DBs
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

    timed_out = threading.Event()

    def _interrupt():
        timed_out.set()
        conn.interrupt()

    timer = threading.Timer(timeout_s, _interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return ExecResult(ok=True, rows=rows, error=None)
    except Exception as exc:  # noqa: BLE001 - any failure means "invalid prediction"
        msg = "timeout" if timed_out.is_set() else f"{type(exc).__name__}: {exc}"
        return ExecResult(ok=False, rows=None, error=msg)
    finally:
        timer.cancel()
        conn.close()


def results_match(pred_sql: str, gold_sql: str, db_path: str, timeout_s: float = 15.0) -> bool:
    """True iff the predicted query executes and returns the same rows as the gold query."""
    gold = execute(gold_sql, db_path, timeout_s)
    if not gold.ok:
        # A broken gold query is a data problem, not a model win. Treat as non-match and let the
        # caller surface it (eval.py logs these separately).
        return False
    pred = execute(pred_sql, db_path, timeout_s)
    if not pred.ok:
        return False
    order_matters = "order by" in gold_sql.lower()
    return _normalize_rows(pred.rows, order_matters) == _normalize_rows(gold.rows, order_matters)


def is_valid_sql(pred_sql: str, db_path: str, timeout_s: float = 15.0) -> bool:
    """True iff the predicted query executes at all (parses and runs), regardless of correctness."""
    return execute(pred_sql, db_path, timeout_s).ok


def reward(pred_sql: str, gold_sql: str, db_path: str, timeout_s: float = 15.0) -> float:
    """Reward for the RL rung (2e).

    1.0  -> executes and matches the gold result set
    0.1  -> executes but returns the wrong rows (partial credit for producing valid SQL)
    0.0  -> does not execute (syntax error, bad column, timeout, ...)

    The partial-credit tier gives GRPO a smoother signal than a bare 0/1 and nudges the model
    toward at-least-valid SQL early in training.
    """
    if not execute(pred_sql, db_path, timeout_s).ok:
        return 0.0
    return 1.0 if results_match(pred_sql, gold_sql, db_path, timeout_s) else 0.1


if __name__ == "__main__":
    # tiny self-test against an in-memory-style temp DB
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    c = sqlite3.connect(path)
    c.executescript(
        "CREATE TABLE t(id INTEGER, name TEXT);"
        "INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c');"
    )
    c.commit()
    c.close()

    assert results_match("SELECT name, id FROM t", "SELECT id, name FROM t", path)  # col order
    assert results_match("SELECT id FROM t ORDER BY id DESC",
                         "SELECT id FROM t ORDER BY id DESC", path)
    assert not results_match("SELECT id FROM t ORDER BY id",
                             "SELECT id FROM t ORDER BY id DESC", path)  # order matters
    assert not is_valid_sql("SELECT nope FROM t", path)
    assert reward("SELECT id, name FROM t", "SELECT id, name FROM t", path) == 1.0
    assert reward("SELECT id FROM t", "SELECT id, name FROM t", path) == 0.1
    assert reward("SELECT bad", "SELECT id FROM t", path) == 0.0
    os.unlink(path)
    print("sql_executor self-test passed")
