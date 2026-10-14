"""Turn a result table into a short, accurate summary.

Language models are good at writing SQL and bad at reading numbers back out
of a table: they swap rankings and misquote totals. So the summary is built
here, in plain Python, from the actual result. Every number in it comes
straight from the DataFrame.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Facts:
    rows: int
    label_col: str | None = None
    value_col: str | None = None
    top: tuple[str, float] | None = None
    bottom: tuple[str, float] | None = None
    total: float | None = None
    single_value: tuple[str, object] | None = None
    notes: list[str] = field(default_factory=list)


def _fmt(x: object) -> str:
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, numbers.Integral):
        return f"{int(x):,}"
    if isinstance(x, numbers.Real):
        x = float(x)
        return f"{int(x):,}" if x.is_integer() else f"{x:,.2f}"
    return str(x)


def _pretty(col: str) -> str:
    return col.replace("_", " ")


def compute_facts(df: pd.DataFrame) -> Facts:
    facts = Facts(rows=len(df))
    if df.empty:
        return facts

    if df.shape == (1, 1):
        col = df.columns[0]
        facts.single_value = (col, df.iloc[0, 0])
        return facts

    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    # Id columns are numeric but are labels, not measures.
    ids = [c for c in numeric if c.lower().endswith("_id") or c.lower() in ("id", "year", "order_year")]
    measures = [c for c in numeric if c not in ids]
    labels = labels + ids

    if not labels or not measures:
        return facts

    label, value = labels[0], measures[-1]
    series = df[[label, value]].dropna()
    if series.empty:
        return facts

    facts.label_col, facts.value_col = label, value
    top = series.loc[series[value].idxmax()]
    bottom = series.loc[series[value].idxmin()]
    facts.top = (str(top[label]), float(top[value]))
    facts.bottom = (str(bottom[label]), float(bottom[value]))

    # Only add up columns that are additive (counts, money, units), not rates or averages.
    if not any(w in value.lower() for w in ("pct", "percent", "rate", "avg", "average", "margin", "price")):
        facts.total = float(series[value].sum())

    if series[label].duplicated().any():
        facts.notes.append(f"{label} repeats across rows, so the table is broken down further.")
    return facts


def summarize(facts: Facts) -> str:
    if facts.rows == 0:
        return "The query returned no rows."
    if facts.single_value:
        col, val = facts.single_value
        return f"{_pretty(col).capitalize()}: {_fmt(val)}."
    if facts.top is None:
        return f"The query returned {facts.rows:,} rows."

    v = _pretty(facts.value_col)
    parts = [f"{facts.rows:,} rows."]
    if facts.rows > 1:
        parts.append(f"Highest {v}: {facts.top[0]} ({_fmt(facts.top[1])}).")
        parts.append(f"Lowest {v}: {facts.bottom[0]} ({_fmt(facts.bottom[1])}).")
    else:
        parts.append(f"{facts.top[0]}: {_fmt(facts.top[1])} {v}.")
    if facts.total is not None and facts.rows > 1:
        parts.append(f"Total {v}: {_fmt(facts.total)}.")
    parts.extend(facts.notes)
    return " ".join(parts)
