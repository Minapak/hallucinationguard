"""Optional LLM and embedding hooks.

Everything in this package works without a model (rules only). When a
service passes an ``LLM`` the bank uses it for fact extraction, observation
summaries, reflection and wiki prose; results are marked ``llm:<model>``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str: ...


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class OpenAIChat:
    """Chat Completions over httpx (any OpenAI-compatible endpoint)."""
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        import httpx
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        r = httpx.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": self.model, "max_tokens": max_tokens, "temperature": 0.2,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


@dataclass
class AnthropicChat:
    model: str = "claude-sonnet-4-6"
    api_key: Optional[str] = None
    timeout: float = 30.0

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        import httpx
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=self.timeout,
                       headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                       json={"model": self.model, "max_tokens": max_tokens, "system": system,
                             "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx
        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        r = httpx.post(f"{self.base_url}/embeddings", timeout=self.timeout,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": self.model, "input": list(texts)})
        r.raise_for_status()
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


# ── prompts (original wording for this package) ──────────────────────────────

EXTRACT_SYSTEM = (
    "You turn a piece of text into short, self-contained facts worth remembering about the user or "
    "their work. Each fact must stand alone without the original text. Skip greetings and filler. "
    "Never invent details. Reply with JSON only: "
    '{"facts":[{"text":"...","kind":"world|experience","entities":["..."],"when":"YYYY-MM-DD or null",'
    '"key":"optional stable slot like profile:goal or empty","static":false}]}'
)

REFLECT_SYSTEM = (
    "You answer a question using only the numbered memories given. Cite the memories you use as [n]. "
    "If the memories do not answer the question, say so plainly. Keep it under 120 words. "
    "Stance: skepticism={skepticism}/5, literalism={literalism}/5, empathy={empathy}/5."
)

OBSERVE_SYSTEM = (
    "Merge the numbered facts about one subject into one or two sentences that state what is "
    "consistently true. Mention a contradiction if there is one. No new information. Plain text only."
)


def parse_json_block(text: str) -> Any:
    """Parse the first JSON object in a model reply (tolerates code fences)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def extract_facts(llm: LLM, text: str, max_facts: int = 8) -> list[dict[str, Any]]:
    data = parse_json_block(llm.complete(EXTRACT_SYSTEM, text[:6000], max_tokens=700)) or {}
    out = []
    for f in (data.get("facts") or [])[:max_facts]:
        t = str(f.get("text", "")).strip()
        if not t:
            continue
        when = None
        if f.get("when"):
            try:
                when = datetime.fromisoformat(str(f["when"])[:10]).replace(tzinfo=timezone.utc)
            except ValueError:
                when = None
        kind = f.get("kind") if f.get("kind") in ("world", "experience") else "world"
        out.append({"text": t[:500], "kind": kind, "entities": [str(e)[:80] for e in f.get("entities") or []][:8],
                    "occurred_at": when, "key": str(f.get("key") or "")[:200], "is_static": bool(f.get("static"))})
    return out
