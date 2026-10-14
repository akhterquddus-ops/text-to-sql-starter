"""Command-line interface.

    python -m sqlassist.cli "What was net revenue by region in 2025?"
    python -m sqlassist.cli            # interactive
"""

from __future__ import annotations

import sys

from .config import Settings, make_provider
from .pipeline import Answer, Assistant
from .semantic import SemanticLayer


def show(ans: Answer) -> None:
    if not ans.ok:
        print(f"\n✗ Could not answer safely: {ans.error}")
        for i, a in enumerate(ans.attempts, 1):
            print(f"  attempt {i}: {a.error}")
        return
    print(f"\n{ans.summary}\n")
    print(ans.data.to_string(index=False, max_rows=30))
    print(f"\nSQL ({len(ans.attempts)} attempt(s), {ans.seconds}s):\n{ans.sql}\n")


def main() -> None:
    s = Settings.from_env()
    assistant = Assistant(make_provider(s), SemanticLayer.load(s.semantic_layer), s.db_path,
                          max_attempts=s.max_attempts, row_limit=s.row_limit,
                          query_timeout_s=s.query_timeout_s)
    if len(sys.argv) > 1:
        show(assistant.ask(" ".join(sys.argv[1:])))
        return
    print("Ask a question (empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(assistant.ask(q))


if __name__ == "__main__":
    main()
