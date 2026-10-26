"""Data types for the memory layer.

The record shape follows two published designs, re-implemented here:

* hindsight (vectorize-io, MIT): memory *banks*, fact kinds ``world`` /
  ``experience`` / ``observation``, retain → recall → reflect.
* supermemory (MIT schemas): a version chain (``version``, ``is_latest``,
  ``parent_id``), relations ``updates`` / ``extends`` / ``derives``,
  static vs dynamic profile facts and time-based forgetting.

See NOTICE.md for attribution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

MemoryKind = Literal["world", "experience", "observation", "opinion"]
Relation = Literal["updates", "extends", "derives"]

KINDS: tuple[str, ...] = ("world", "experience", "observation", "opinion")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryUnit:
    """One remembered fact. Immutable in meaning: an update is a new version."""

    bank_id: str
    text: str
    kind: str = "world"
    id: str = field(default_factory=new_id)
    entities: list[str] = field(default_factory=list)
    occurred_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=utcnow)
    source: str = ""                      # e.g. "tutor_turn:123", "assessment:9"
    key: str = ""                         # caller-defined slot, e.g. "profile:goal"
    bucket: str = ""                      # profile bucket, e.g. "preferences"
    is_static: bool = False               # long-lived profile fact
    proof_count: int = 1                  # how many retains supported it
    source_ids: list[str] = field(default_factory=list)   # for observations
    version: int = 1
    is_latest: bool = True
    parent_id: Optional[str] = None
    relations: dict[str, str] = field(default_factory=dict)  # other id -> Relation
    forget_after: Optional[datetime] = None
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def expired(self, now: Optional[datetime] = None) -> bool:
        return self.forget_after is not None and (now or utcnow()) >= _aware(self.forget_after)


def _aware(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass
class RecallHit:
    unit: MemoryUnit
    score: float
    arms: dict[str, int] = field(default_factory=dict)   # arm name -> 1-based rank


@dataclass
class RecallResult:
    query: str
    hits: list[RecallHit]
    arms_used: list[str]

    def texts(self) -> list[str]:
        return [h.unit.text for h in self.hits]

    def as_context(self, numbered: bool = True) -> str:
        """Prompt-ready block. Numbers let an answer cite ``[n]``."""
        lines = []
        for i, h in enumerate(self.hits, 1):
            when = h.unit.occurred_at.date().isoformat() if h.unit.occurred_at else ""
            tag = f"{h.unit.kind}{', ' + when if when else ''}"
            lines.append(f"[{i}] ({tag}) {h.unit.text}" if numbered else f"- {h.unit.text}")
        return "\n".join(lines)


@dataclass
class RetainResult:
    created: list[MemoryUnit]
    reinforced: list[MemoryUnit]     # duplicates that raised proof_count
    superseded: list[MemoryUnit]     # old versions replaced by an update


@dataclass
class Reflection:
    query: str
    answer: str
    citations: list[MemoryUnit]
    generated_by: str                # "llm:<model>" or "rules"
    stored: Optional[MemoryUnit] = None


@dataclass
class Profile:
    static: list[MemoryUnit]
    dynamic: list[MemoryUnit]
    buckets: dict[str, list[MemoryUnit]]

    def to_dict(self) -> dict[str, Any]:
        def t(us: list[MemoryUnit]) -> list[str]:
            return [u.text for u in us]
        return {"static": t(self.static), "dynamic": t(self.dynamic),
                "buckets": {k: t(v) for k, v in self.buckets.items()}}
