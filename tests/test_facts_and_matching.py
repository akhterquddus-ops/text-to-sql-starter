import pandas as pd

from sqlassist.evaluate import results_match
from sqlassist.facts import compute_facts, summarize


def test_summary_numbers_come_from_the_table():
    df = pd.DataFrame({"region_name": ["South", "North", "West"], "net_revenue": [300.0, 500.0, 100.0]})
    f = compute_facts(df)
    assert f.top == ("North", 500.0)
    assert f.bottom == ("West", 100.0)
    assert f.total == 900.0
    text = summarize(f)
    assert "North (500)" in text and "West (100)" in text and "900" in text


def test_rates_are_not_added_up():
    df = pd.DataFrame({"channel": ["Website", "App"], "cancellation_rate_pct": [7.1, 6.4]})
    f = compute_facts(df)
    assert f.total is None
    assert "Total" not in summarize(f)


def test_single_value_summary():
    assert summarize(compute_facts(pd.DataFrame({"delivered_orders": [1349]}))) == "Delivered orders: 1,349."


def test_empty_result():
    assert summarize(compute_facts(pd.DataFrame({"a": []}))) == "The query returned no rows."


def test_match_ignores_row_order_and_column_names():
    exp = pd.DataFrame({"region": ["A", "B"], "rev": [1.0, 2.0]})
    act = pd.DataFrame({"x": ["B", "A"], "y": [2.004, 1.0]})
    assert results_match(exp, act)


def test_match_ignores_column_order_and_extra_columns():
    exp = pd.DataFrame({"region": ["A", "B"], "rev": [1.0, 2.0]})
    act = pd.DataFrame({"rev": [2.0, 1.0], "orders": [9, 8], "region": ["B", "A"]})
    assert results_match(exp, act)


def test_mismatch_is_detected():
    exp = pd.DataFrame({"region": ["A", "B"], "rev": [1.0, 2.0]})
    assert not results_match(exp, pd.DataFrame({"region": ["A", "B"], "rev": [1.0, 3.0]}))
    assert not results_match(exp, pd.DataFrame({"region": ["A"], "rev": [1.0]}))
    # same values in each column but paired differently across rows
    assert not results_match(exp, pd.DataFrame({"region": ["A", "B"], "rev": [2.0, 1.0]}))
