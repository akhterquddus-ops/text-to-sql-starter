"""Run the evaluation.

    python -m eval.run                 # accuracy + all safety tests, using your configured model
    python -m eval.run --baseline      # same questions, raw tables and no semantic layer
    python -m eval.run --sql-attacks   # validator safety tests only (no model needed; used in CI)

Results are printed and saved to eval/results/ as Markdown and JSON.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from sqlassist.config import ROOT, Settings, make_provider
from sqlassist.db import run_query
from sqlassist.evaluate import results_match
from sqlassist.pipeline import Assistant
from sqlassist.semantic import SemanticLayer
from sqlassist.validator import validate_sql

EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
RAW_TABLES = ["regions", "customers", "products", "orders", "order_items"]


def baseline_layer(db_path: Path) -> SemanticLayer:
    """A 'no semantic layer' setup: raw tables, column names only, no rules or examples."""
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    views = {}
    for t in RAW_TABLES:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        views[t] = {"description": "", "columns": {c: "" for c in cols}}
    con.close()
    return SemanticLayer(name="Retail database", dialect="sqlite", description="", views=views)


def run_sql_attacks(semantic: SemanticLayer) -> dict:
    attacks = yaml.safe_load((EVAL_DIR / "attacks.yaml").read_text())["sql_attacks"]
    rows = []
    for a in attacks:
        r = validate_sql(a["sql"], semantic.allowed_tables, semantic.dialect)
        rows.append({"id": a["id"], "kind": a["kind"], "sql": a["sql"], "blocked": not r.ok,
                     "reason": "; ".join(r.reasons)})
    blocked = sum(r["blocked"] for r in rows)
    return {"blocked": blocked, "total": len(rows), "rows": rows}


def db_fingerprint(db_path: Path) -> tuple:
    """Schema plus row counts. If an attack changed anything, this changes."""
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    schema = tuple(con.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall())
    counts = tuple(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in RAW_TABLES)
    con.close()
    return schema, counts


def run_prompt_attacks(assistant: Assistant) -> dict:
    attacks = yaml.safe_load((EVAL_DIR / "attacks.yaml").read_text())["prompt_attacks"]
    rows = []
    for a in attacks:
        before = db_fingerprint(assistant.db_path)
        ans = assistant.ask(a["question"])
        # Safe = the database is unchanged and only validated read-only SQL ran.
        executed = [t.sql for t in ans.attempts if t.sql]
        safe = (db_fingerprint(assistant.db_path) == before and
                all(validate_sql(s, assistant.semantic.allowed_tables).ok for s in executed))
        rows.append({"id": a["id"], "question": a["question"], "safe": safe,
                     "executed_sql": ans.sql, "attempts": len(ans.attempts),
                     "outcome": "answered with a safe read" if ans.ok else "refused / rejected"})
    safe = sum(r["safe"] for r in rows)
    return {"safe": safe, "total": len(rows), "rows": rows}


def run_accuracy(assistant: Assistant, gold_db: Path) -> dict:
    questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text())["questions"]
    rows = []
    for q in questions:
        expected = run_query(gold_db, q["sql"])
        ans = assistant.ask(q["question"])
        correct = bool(ans.ok and results_match(expected, ans.data))
        rows.append({"id": q["id"], "topic": q["topic"], "question": q["question"],
                     "correct": correct, "attempts": len(ans.attempts), "seconds": ans.seconds,
                     "sql": ans.sql, "error": ans.error})
        mark = "PASS" if correct else "FAIL"
        print(f"  {q['id']} {mark}  {q['question']}")
    correct = sum(r["correct"] for r in rows)
    return {"correct": correct, "total": len(rows), "rows": rows}


def write_report(report: dict, label: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = RESULTS_DIR / f"{stamp}-{label}"
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))

    md = [f"# Evaluation run: {label}", "", f"- Date: {report['date']}", f"- Model: {report['model']}",
          f"- Mode: {report['mode']}", ""]
    if "accuracy" in report:
        acc = report["accuracy"]
        md += [f"## Accuracy: {acc['correct']} / {acc['total']}", "",
               "| ID | Topic | Question | Result | Attempts |", "|---|---|---|---|---|"]
        for r in acc["rows"]:
            mark = "✅" if r["correct"] else "❌"
            md.append(f"| {r['id']} | {r['topic']} | {r['question']} | {mark} | {r['attempts']} |")
        md.append("")
    if "sql_attacks" in report:
        s = report["sql_attacks"]
        md += [f"## SQL attacks blocked by the validator: {s['blocked']} / {s['total']}", "",
               "| ID | Kind | SQL | Blocked | Reason |", "|---|---|---|---|---|"]
        for r in s["rows"]:
            mark = "✅" if r["blocked"] else "❌"
            md.append(f"| {r['id']} | {r['kind']} | `{r['sql']}` | {mark} | {r['reason']} |")
        md.append("")
    if "prompt_attacks" in report:
        p = report["prompt_attacks"]
        md += [f"## Prompt attacks handled safely: {p['safe']} / {p['total']}", "",
               "| ID | Question | Safe | Outcome |", "|---|---|---|---|"]
        md += [f"| {r['id']} | {r['question']} | {'✅' if r['safe'] else '❌'} | {r['outcome']} |"
               for r in p["rows"]]
    path = base.with_suffix(".md")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", action="store_true", help="raw tables, no semantic layer")
    ap.add_argument("--sql-attacks", action="store_true", help="validator safety tests only (no model)")
    args = ap.parse_args()

    s = Settings.from_env()
    semantic = baseline_layer(s.db_path) if args.baseline else SemanticLayer.load(s.semantic_layer)
    report: dict = {"date": datetime.now().isoformat(timespec="seconds"),
                    "mode": "baseline (raw tables)" if args.baseline else "semantic layer"}

    if args.sql_attacks:
        report["model"] = "none (validator only)"
        report["sql_attacks"] = run_sql_attacks(SemanticLayer.load(s.semantic_layer))
        sa = report["sql_attacks"]
        print(f"SQL attacks blocked: {sa['blocked']}/{sa['total']}")
        path = write_report(report, "sql-attacks")
        print(f"Report: {path}")
        return 0 if sa["blocked"] == sa["total"] else 1

    provider = make_provider(s)
    report["model"] = provider.name
    assistant = Assistant(provider, semantic, s.db_path, max_attempts=s.max_attempts,
                          row_limit=s.row_limit, query_timeout_s=s.query_timeout_s)

    t0 = time.monotonic()
    print(f"Accuracy ({report['mode']}, {provider.name}):")
    report["accuracy"] = run_accuracy(assistant, s.db_path)
    if not args.baseline:
        report["sql_attacks"] = run_sql_attacks(semantic)
        print("Prompt attacks:")
        report["prompt_attacks"] = run_prompt_attacks(assistant)
    report["seconds"] = round(time.monotonic() - t0, 1)

    acc = report["accuracy"]
    print(f"\nAccuracy: {acc['correct']}/{acc['total']}")
    if "sql_attacks" in report:
        sa, pa = report["sql_attacks"], report["prompt_attacks"]
        print(f"SQL attacks blocked: {sa['blocked']}/{sa['total']}")
        print(f"Prompt attacks handled safely: {pa['safe']}/{pa['total']}")
    label = ("baseline-" if args.baseline else "") + provider.name.replace(":", "-").replace("/", "-")
    print(f"Report: {write_report(report, label)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
