"""Compare a model's result table with the reference result.

Two answers match when they contain the same rows, ignoring row order,
column order and column names. Numbers are compared to 2 decimal places.
If the model returns extra columns, the answer still matches as long as
every reference column can be found among them.
"""

from __future__ import annotations

from collections import Counter
from itertools import permutations

import pandas as pd


def _norm(v: object) -> object:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) or hasattr(v, "__float__") and not isinstance(v, str):
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            pass
    return str(v).strip()


def _rows(df: pd.DataFrame, cols: list) -> Counter:
    return Counter(tuple(_norm(v) for v in row) for row in df[cols].itertuples(index=False, name=None))


def _column_values(df: pd.DataFrame, col) -> Counter:
    return Counter(_norm(v) for v in df[col].tolist())


def results_match(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    if expected is None or actual is None:
        return False
    if len(expected) != len(actual):
        return False
    if len(expected) == 0:
        return True
    ecols, acols = list(expected.columns), list(actual.columns)
    if len(acols) < len(ecols):
        return False

    # Columns with duplicate names confuse pandas indexing; use positions.
    expected = expected.set_axis(range(len(ecols)), axis=1)
    actual = actual.set_axis(range(len(acols)), axis=1)
    ecols, acols = list(expected.columns), list(actual.columns)

    # For each reference column, which model columns hold the same values?
    candidates = []
    for e in ecols:
        ev = _column_values(expected, e)
        matches = [a for a in acols if _column_values(actual, a) == ev]
        if not matches:
            return False
        candidates.append(matches)

    target = _rows(expected, ecols)
    # Try the column pairings (small in practice) and check whole rows line up.
    if len(acols) <= 8:
        for perm in permutations(acols, len(ecols)):
            pairs_ok = all(p in c for p, c in zip(perm, candidates, strict=True))
            if pairs_ok and _rows(actual, list(perm)) == target:
                return True
        return False
    first = [c[0] for c in candidates]
    return _rows(actual, first) == target
