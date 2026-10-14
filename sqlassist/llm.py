"""Model providers.

- OllamaProvider: a local open-source model (default). Nothing leaves your machine.
- OpenAICompatibleProvider: any endpoint that speaks the OpenAI chat API
  (OpenAI, Azure OpenAI, Groq, LM Studio, vLLM, ...).
- ScriptedProvider: returns pre-set answers. Used by the tests, so CI runs
  without a model.
"""

from __future__ import annotations

import re
from typing import Protocol

import requests

SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict]) -> str: ...


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (code block if present, else the text)."""
    match = SQL_BLOCK.search(text or "")
    sql = match.group(1) if match else (text or "")
    return sql.strip().rstrip(";").strip()


class OllamaProvider:
    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete(self, system: str, messages: list[dict]) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatibleProvider:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 120.0):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"openai-compatible:{model}"

    def complete(self, system: str, messages: list[dict]) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "temperature": 0,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class ScriptedProvider:
    """Replies from a fixed list, in order. For tests and offline demos."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self.name = "scripted"

    def complete(self, system: str, messages: list[dict]) -> str:
        self.calls.append(list(messages))  # snapshot: the caller keeps appending
        if not self.replies:
            raise RuntimeError("ScriptedProvider ran out of replies")
        return self.replies.pop(0)
