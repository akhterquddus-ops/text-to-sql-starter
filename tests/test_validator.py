from pathlib import Path

import pytest
import yaml

from sqlassist.validator import validate_sql

ROOT = Path(__file__).resolve().parent.parent
ATTACKS = yaml.safe_load((ROOT / "eval" / "attacks.yaml").read_text())["sql_attacks"]
VIEWS = {"vw_order_lines", "vw_sales_by_month_region", "vw_customer_summary"}


@pytest.mark.parametrize("attack", ATTACKS, ids=[a["id"] for a in ATTACKS])
def test_every_attack_is_blocked(attack):
    result = validate_sql(attack["sql"], VIEWS)
    assert not result.ok, f"{attack['id']} was not blocked"
    assert result.reasons


@pytest.mark.parametrize("sql", [
    "SELECT region_name, SUM(net_revenue) AS rev FROM vw_sales_by_month_region GROUP BY region_name",
    "WITH t AS (SELECT * FROM vw_order_lines WHERE status = 'Delivered') SELECT COUNT(*) FROM t",
    "SELECT segment FROM vw_customer_summary UNION SELECT segment FROM vw_order_lines",
    "select * from VW_ORDER_LINES limit 5",
])
def test_safe_queries_pass(sql):
    assert validate_sql(sql, VIEWS).ok


def test_limit_is_added_when_missing():
    r = validate_sql("SELECT * FROM vw_order_lines", VIEWS, row_limit=50)
    assert r.ok and "LIMIT 50" in r.sql


def test_existing_limit_is_kept():
    r = validate_sql("SELECT * FROM vw_order_lines LIMIT 5", VIEWS)
    assert r.ok and "LIMIT 5" in r.sql and "LIMIT 1000" not in r.sql


def test_comments_are_removed_from_output():
    r = validate_sql("SELECT 1 FROM vw_order_lines -- ; DROP TABLE orders", VIEWS)
    assert r.ok and "DROP" not in r.sql


def test_cte_named_like_a_raw_table_is_fine_when_built_from_views():
    # 'orders' here is the CTE, defined from an approved view.
    r = validate_sql("WITH orders AS (SELECT * FROM vw_order_lines) SELECT COUNT(*) FROM orders", VIEWS)
    assert r.ok


def test_cte_cannot_be_used_to_smuggle_in_a_raw_table():
    # Inside its own body, 'customers' means the real table.
    r = validate_sql("WITH customers AS (SELECT * FROM customers) SELECT * FROM customers", VIEWS)
    assert not r.ok and "customers" in r.reasons[0]


def test_ctes_can_build_on_each_other():
    sql = "WITH a AS (SELECT * FROM vw_order_lines), b AS (SELECT * FROM a) SELECT COUNT(*) FROM b"
    assert validate_sql(sql, VIEWS).ok


def test_garbage_is_rejected():
    assert not validate_sql("this is not sql", VIEWS).ok
    assert not validate_sql("", VIEWS).ok


def test_approved_view_with_system_like_name_is_allowed():
    # "user_" is an Oracle catalog prefix, but an approved view may still use it.
    assert validate_sql("SELECT * FROM user_activity", {"user_activity"}).ok
    assert not validate_sql("SELECT * FROM user_tables", {"user_activity"}).ok


@pytest.mark.parametrize("dialect,keyword", [("tsql", "TOP"), ("postgres", "LIMIT"), ("oracle", "FETCH")])
def test_row_limit_uses_the_target_dialect(dialect, keyword):
    r = validate_sql("SELECT region_name FROM vw_order_lines", VIEWS, dialect=dialect, row_limit=10)
    assert r.ok and keyword in r.sql.upper()
