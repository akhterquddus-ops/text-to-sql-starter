"""Settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader so the project needs no extra dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    db_path: Path
    semantic_layer: Path
    provider: str
    ollama_model: str
    ollama_url: str
    openai_model: str
    openai_base_url: str
    openai_api_key: str
    max_attempts: int
    row_limit: int
    query_timeout_s: float

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv(ROOT / ".env")
        e = os.environ.get
        return cls(
            db_path=Path(e("SQLASSIST_DB", str(ROOT / "data" / "demo.db"))),
            semantic_layer=Path(e("SQLASSIST_SEMANTIC_LAYER", str(ROOT / "semantic_layer.yaml"))),
            provider=e("LLM_PROVIDER", "ollama").lower(),
            ollama_model=e("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            ollama_url=e("OLLAMA_URL", "http://localhost:11434"),
            openai_model=e("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=e("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_api_key=e("OPENAI_API_KEY", ""),
            max_attempts=int(e("SQLASSIST_MAX_ATTEMPTS", "3")),
            row_limit=int(e("SQLASSIST_ROW_LIMIT", "1000")),
            query_timeout_s=float(e("SQLASSIST_QUERY_TIMEOUT", "10")),
        )


def make_provider(s: Settings):
    from .llm import OllamaProvider, OpenAICompatibleProvider

    if s.provider == "ollama":
        return OllamaProvider(model=s.ollama_model, base_url=s.ollama_url)
    if s.provider in ("openai", "openai-compatible"):
        if not s.openai_api_key:
            raise ValueError("Set OPENAI_API_KEY to use the OpenAI-compatible provider")
        return OpenAICompatibleProvider(model=s.openai_model, api_key=s.openai_api_key,
                                        base_url=s.openai_base_url)
    raise ValueError(f"Unknown LLM_PROVIDER: {s.provider!r} (use 'ollama' or 'openai')")
