import sqlite3

import pytest

from sqlassist.db import QueryTimeout, run_query


def test_reads_work(db_path):
    df = run_query(db_path, "SELECT COUNT(*) AS n FROM vw_order_lines")
    assert df.iloc[0, 0] > 0


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE orders SET status = 'x'",
    "DROP TABLE orders",
    "CREATE TABLE x (a INT)",
])
def test_connection_is_read_only(db_path, sql):
    # Even if the validator were bypassed, the connection refuses writes.
    with pytest.raises(sqlite3.OperationalError):
        run_query(db_path, sql)
    assert run_query(db_path, "SELECT COUNT(*) FROM orders").iloc[0, 0] == 3000


def test_runaway_query_times_out(db_path):
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(QueryTimeout):
        run_query(db_path, endless, timeout_s=0.5)


def test_missing_database_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate"):
        run_query(tmp_path / "nope.db", "SELECT 1")
