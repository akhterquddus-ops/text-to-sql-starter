"""Question in, answer out.

    question -> model writes SQL -> validator -> read-only database -> facts -> summary

If the validator rejects the SQL or the database returns an error, the error
is sent back to the model and it gets another try (up to `max_attempts`).
Every step is recorded so the user can see exactly how the answer was made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .db import QueryTimeout, run_query
from .facts import Facts, compute_facts, summarize
from .llm import LLMProvider, extract_sql
from .semantic import SemanticLayer
from .validator import DEFAULT_ROW_LIMIT, validate_sql


@dataclass
class Attempt:
    raw_sql: str
    sql: str | None
    error: str | None


@dataclass
class Answer:
    question: str
    ok: bool
    sql: str | None = None
    data: pd.DataFrame | None = None
    facts: Facts | None = None
    summary: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


class Assistant:
    def __init__(
        self,
        provider: LLMProvider,
        semantic: SemanticLayer,
        db_path: str | Path,
        max_attempts: int = 3,
        row_limit: int = DEFAULT_ROW_LIMIT,
        query_timeout_s: float = 10.0,
    ):
        self.provider = provider
        self.semantic = semantic
        self.db_path = Path(db_path)
        self.max_attempts = max_attempts
        self.row_limit = row_limit
        self.query_timeout_s = query_timeout_s
        self._system = semantic.system_prompt()

    def ask(self, question: str) -> Answer:
        started = time.monotonic()
        answer = Answer(question=question, ok=False)
        messages: list[dict] = [{"role": "user", "content": question}]

        for _ in range(self.max_attempts):
            try:
                reply = self.provider.complete(self._system, messages)
            except Exception as e:  # network / model errors are not retried
                answer.error = f"Model call failed: {e}"
                break

            raw = extract_sql(reply)
            check = validate_sql(raw, self.semantic.allowed_tables, self.semantic.dialect, self.row_limit)
            if not check.ok:
                err = "; ".join(check.reasons)
                answer.attempts.append(Attempt(raw, None, f"Rejected by validator: {err}"))
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That SQL was rejected: {err}. "
                                                f"Write one SELECT using only the approved views."},
                ]
                continue

            try:
                df = run_query(self.db_path, check.sql, timeout_s=self.query_timeout_s)
            except QueryTimeout as e:
                answer.attempts.append(Attempt(raw, check.sql, str(e)))
                answer.error = str(e)
                break
            except Exception as e:
                err = str(e).splitlines()[0]
                answer.attempts.append(Attempt(raw, check.sql, f"Database error: {err}"))
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"The database returned an error: {err}. Fix the SQL."},
                ]
                continue

            answer.attempts.append(Attempt(raw, check.sql, None))
            answer.ok = True
            answer.sql = check.sql
            answer.data = df
            answer.facts = compute_facts(df)
            answer.summary = summarize(answer.facts)
            answer.error = None
            break
        else:
            answer.error = answer.attempts[-1].error if answer.attempts else "No attempts made"

        answer.seconds = round(time.monotonic() - started, 2)
        return answer
