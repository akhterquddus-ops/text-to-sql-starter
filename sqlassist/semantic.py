"""Load the semantic layer and turn it into prompt context for the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SemanticLayer:
    name: str
    dialect: str
    description: str
    views: dict[str, dict]
    rules: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)

    @property
    def allowed_tables(self) -> set[str]:
        """Only the documented views may be queried."""
        return {v.lower() for v in self.views}

    @classmethod
    def load(cls, path: str | Path) -> SemanticLayer:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        db = data.get("database", {})
        return cls(
            name=db.get("name", "Database"),
            dialect=db.get("dialect", "sqlite"),
            description=(db.get("description") or "").strip(),
            views=data.get("views", {}),
            rules=data.get("rules", []),
            examples=data.get("examples", []),
        )

    def schema_text(self) -> str:
        lines = []
        for view, spec in self.views.items():
            lines.append(f"VIEW {view}: {spec.get('description', '')}")
            for col, meaning in spec.get("columns", {}).items():
                lines.append(f"  - {col}: {meaning}")
            lines.append("")
        return "\n".join(lines).strip()

    def system_prompt(self) -> str:
        rules = "\n".join(f"- {r}" for r in self.rules)
        examples = "\n\n".join(
            f"Question: {e['question']}\nSQL:\n{e['sql'].strip()}" for e in self.examples
        )
        return f"""You translate business questions into a single {self.dialect.upper()} SELECT query.

Database: {self.name}
{self.description}

You may ONLY use these views and columns:
{self.schema_text()}

Business rules:
{rules}

Output rules:
- Return exactly one SELECT statement (a WITH ... SELECT is fine) inside a ```sql code block.
- Never modify data. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH or PRAGMA.
- Use only the views listed above, never the underlying tables.
- If the question cannot be answered from these views, return exactly:
  SELECT 'Not answerable from the available data' AS message

Worked examples:

{examples}
"""
