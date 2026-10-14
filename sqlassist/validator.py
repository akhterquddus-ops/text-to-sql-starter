"""SQL safety validator.

Every query the model writes passes through here before it reaches the
database. The validator parses the SQL into a syntax tree (with sqlglot) and
rejects anything that is not a single read-only query over approved views.

This is the second line of defence. The first is that the database connection
itself is opened read-only (see db.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

DEFAULT_ROW_LIMIT = 1000

# sqlglot logs a warning when it falls back to a generic Command; we reject those anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Statement and expression types that must never appear anywhere in the tree.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    t
    for t in (
        getattr(exp, name, None)
        for name in (
            "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter",
            "AlterTable", "TruncateTable", "Command", "Pragma", "Attach", "Detach",
            "Transaction", "Commit", "Rollback", "Grant", "Revoke", "Use", "Set",
            "Copy", "Into", "LoadData",
        )
    )
    if t is not None
)

# Functions that can read files, load code or stall the server.
FORBIDDEN_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "randomblob", "zeroblob", "sqlite_compileoption_get", "sqlite_compileoption_used",
    "sleep", "pg_sleep", "benchmark", "xp_cmdshell", "openrowset", "opendatasource",
}

# Catalog tables that would leak the schema beyond the semantic layer.
SYSTEM_PREFIXES = ("sqlite_", "information_schema", "sys.", "pg_", "all_", "dba_", "user_")


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    reasons: list[str] = field(default_factory=list)


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() or "").lower()


def _inside_own_cte(table: exp.Table, name: str) -> bool:
    """True when `table` sits inside the body of the CTE that has the same name.

    In `WITH customers AS (SELECT * FROM customers) ...` the inner `customers`
    is the real table, not the CTE, so it must be checked like any other table.
    """
    node = table.parent
    while node is not None:
        if isinstance(node, exp.CTE) and node.alias_or_name.lower() == name:
            return True
        node = node.parent
    return False


def validate_sql(
    sql: str,
    allowed_tables: set[str],
    dialect: str = "sqlite",
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> ValidationResult:
    """Check that `sql` is one safe SELECT over `allowed_tables`.

    Returns the (possibly rewritten) SQL with a LIMIT added when missing.
    """
    reasons: list[str] = []
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        return ValidationResult(False, sql, ["Empty query"])

    try:
        statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
    except ParseError as e:
        return ValidationResult(False, sql, [f"Could not parse SQL: {str(e).splitlines()[0]}"])

    if len(statements) != 1:
        return ValidationResult(False, sql, ["Only one statement is allowed"])

    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        reasons.append(f"Only SELECT queries are allowed (got {tree.key.upper()})")

    for node in tree.walk():
        if FORBIDDEN_NODES and isinstance(node, FORBIDDEN_NODES):
            reasons.append(f"Forbidden operation: {node.key.upper()}")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in FORBIDDEN_FUNCTIONS:
                reasons.append(f"Forbidden function: {name}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        full = ".".join(p for p in (table.db, table.name) if p).lower()
        if not name:
            reasons.append("Table functions are not allowed")
            continue
        if not table.db and name in allowed:
            continue
        if not table.db and name in cte_names and not _inside_own_cte(table, name):
            continue
        if full.startswith(SYSTEM_PREFIXES) or name.startswith(SYSTEM_PREFIXES):
            reasons.append(f"System catalog access is not allowed: {full}")
        else:
            reasons.append(f"Table or view not approved: {full}")

    if reasons:
        # de-duplicate while keeping order
        return ValidationResult(False, sql, list(dict.fromkeys(reasons)))

    # Cap the number of rows returned.
    if isinstance(tree, exp.Select) and tree.args.get("limit") is None:
        tree = tree.limit(row_limit)
    elif not isinstance(tree, exp.Select) and tree.args.get("limit") is None:
        tree = exp.select("*").from_(tree.subquery("q")).limit(row_limit)

    return ValidationResult(True, tree.sql(dialect=dialect, pretty=True, comments=False), [])
