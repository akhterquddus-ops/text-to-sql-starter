from pathlib import Path

import yaml

from eval.run import run_accuracy
from sqlassist.llm import ScriptedProvider, extract_sql
from sqlassist.pipeline import Assistant

ROOT = Path(__file__).resolve().parent.parent
GOOD = ("```sql\nSELECT region_name, ROUND(SUM(net_revenue), 2) AS net_revenue "
        "FROM vw_sales_by_month_region WHERE order_year = 2025 GROUP BY region_name\n```")


def test_extract_sql_from_code_block_and_plain_text():
    assert extract_sql("Here you go:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_answer_on_first_try(semantic, db_path):
    a = Assistant(ScriptedProvider([GOOD]), semantic, db_path)
    ans = a.ask("Net revenue by region in 2025?")
    assert ans.ok and len(ans.attempts) == 1
    assert len(ans.data) == 5
    assert "Highest net revenue" in ans.summary


def test_self_corrects_after_validator_and_database_errors(semantic, db_path):
    provider = ScriptedProvider([
        "```sql\nSELECT * FROM orders\n```",                                   # raw table -> rejected
        "```sql\nSELECT region_nam FROM vw_sales_by_month_region\n```",       # typo -> DB error
        GOOD,
    ])
    ans = Assistant(provider, semantic, db_path).ask("Net revenue by region in 2025?")
    assert ans.ok
    assert len(ans.attempts) == 3
    assert "not approved" in ans.attempts[0].error
    assert "no such column" in ans.attempts[1].error
    # the error was fed back to the model
    assert "rejected" in provider.calls[1][-1]["content"]


def test_gives_up_safely(semantic, db_path):
    provider = ScriptedProvider(["```sql\nDELETE FROM orders\n```"] * 3)
    ans = Assistant(provider, semantic, db_path, max_attempts=3).ask("Delete cancelled orders")
    assert not ans.ok
    assert ans.data is None
    assert all(a.sql is None for a in ans.attempts)  # nothing reached the database


def test_model_failure_is_reported(semantic, db_path):
    ans = Assistant(ScriptedProvider([]), semantic, db_path).ask("anything")
    assert not ans.ok and "Model call failed" in ans.error


def test_eval_harness_scores_reference_sql_as_correct(semantic, db_path):
    """Feed the gold SQL back as the 'model' answer: every question must score correct.
    This tests the harness itself, not a model."""
    gold = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text())["questions"]
    provider = ScriptedProvider([f"```sql\n{q['sql']}\n```" for q in gold])
    result = run_accuracy(Assistant(provider, semantic, db_path), db_path)
    assert result["correct"] == result["total"] == len(gold)


def test_semantic_layer_exposes_only_views(semantic):
    assert semantic.allowed_tables == {"vw_order_lines", "vw_sales_by_month_region", "vw_customer_summary"}
    prompt = semantic.system_prompt()
    assert "vw_order_lines" in prompt and "Business rules" in prompt
