"""Read-only query execution.

The SQLite file is opened in read-only mode and with `query_only` switched on,
so even a query that slipped past the validator could not change data.
A time limit stops runaway queries.

To use SQL Server, PostgreSQL or Oracle instead, replace `run_query` with a
SQLAlchemy engine that logs in with a read-only database account.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd


class QueryTimeout(Exception):
    pass


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run: python -m sqlassist.data.generate"
        )
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    return con


def run_query(db_path: str | Path, sql: str, timeout_s: float = 10.0) -> pd.DataFrame:
    con = connect_readonly(db_path)
    deadline = time.monotonic() + timeout_s

    def _guard() -> int:
        # returning non-zero aborts the running statement
        return 1 if time.monotonic() > deadline else 0

    con.set_progress_handler(_guard, 10_000)
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description or []]
        return pd.DataFrame(cur.fetchall(), columns=columns)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise QueryTimeout(f"Query took longer than {timeout_s:.0f} seconds") from e
        raise
    finally:
        con.close()
