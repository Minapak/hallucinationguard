"""Storage for memory units and wiki pages.

``InMemoryStore`` is for tests and on-device use. ``SQLStore`` uses
SQLAlchemy Core on its own ``MetaData`` (tables prefixed ``sq_memory_`` /
``sq_wiki_``), so it never touches a service's ORM ``Base``. It accepts an
``Engine`` or a live sync ``Connection`` — async services can pass the
connection SQLAlchemy hands to ``AsyncConnection.run_sync``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Protocol, Sequence

from .types import MemoryUnit, utcnow


@dataclass
class WikiPage:
    bank_id: str
    slug: str
    title: str
    body: str = ""
    sources: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    content_hash: str = ""
    updated_at: datetime = field(default_factory=utcnow)


class MemoryStore(Protocol):
    def add(self, unit: MemoryUnit) -> None: ...
    def update(self, unit: MemoryUnit) -> None: ...
    def get(self, unit_id: str) -> Optional[MemoryUnit]: ...
    def list(self, bank_id: str, kinds: Optional[Sequence[str]] = None, latest_only: bool = True,
             limit: Optional[int] = None) -> list[MemoryUnit]: ...
    def delete(self, unit_id: str) -> None: ...
    def delete_bank(self, bank_id: str) -> int: ...
    def purge_expired(self, now: Optional[datetime] = None) -> int: ...
    # wiki
    def get_page(self, bank_id: str, slug: str) -> Optional[WikiPage]: ...
    def put_page(self, page: WikiPage) -> None: ...
    def list_pages(self, bank_id: str) -> list[WikiPage]: ...
    def delete_page(self, bank_id: str, slug: str) -> None: ...
    def source_hash(self, bank_id: str, source_id: str) -> Optional[str]: ...
    def set_source_hash(self, bank_id: str, source_id: str, digest: str) -> None: ...
    def append_log(self, bank_id: str, op: str, title: str) -> None: ...
    def read_log(self, bank_id: str, limit: int = 50) -> list[tuple[datetime, str, str]]: ...


def _aware(d: Optional[datetime]) -> Optional[datetime]:
    if d is None:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class InMemoryStore:
    def __init__(self) -> None:
        self.units: dict[str, MemoryUnit] = {}
        self.pages: dict[tuple[str, str], WikiPage] = {}
        self.hashes: dict[tuple[str, str], str] = {}
        self.log: list[tuple[str, datetime, str, str]] = []

    def add(self, unit: MemoryUnit) -> None:
        self.units[unit.id] = unit

    update = add

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        return self.units.get(unit_id)

    def list(self, bank_id, kinds=None, latest_only=True, limit=None):
        out = [u for u in self.units.values() if u.bank_id == bank_id
               and (not kinds or u.kind in kinds) and (u.is_latest or not latest_only)]
        out.sort(key=lambda u: _aware(u.created_at), reverse=True)
        return out[:limit] if limit else out

    def delete(self, unit_id: str) -> None:
        self.units.pop(unit_id, None)

    def delete_bank(self, bank_id: str) -> int:
        ids = [k for k, u in self.units.items() if u.bank_id == bank_id]
        for k in ids:
            del self.units[k]
        for k in [k for k in self.pages if k[0] == bank_id]:
            del self.pages[k]
        for k in [k for k in self.hashes if k[0] == bank_id]:
            del self.hashes[k]
        self.log = [e for e in self.log if e[0] != bank_id]
        return len(ids)

    def purge_expired(self, now=None) -> int:
        now = now or utcnow()
        ids = [k for k, u in self.units.items() if u.expired(now)]
        for k in ids:
            del self.units[k]
        return len(ids)

    def get_page(self, bank_id, slug):
        return self.pages.get((bank_id, slug))

    def put_page(self, page: WikiPage) -> None:
        self.pages[(page.bank_id, page.slug)] = page

    def list_pages(self, bank_id):
        return sorted((p for (b, _), p in self.pages.items() if b == bank_id), key=lambda p: p.slug)

    def delete_page(self, bank_id, slug):
        self.pages.pop((bank_id, slug), None)

    def source_hash(self, bank_id, source_id):
        return self.hashes.get((bank_id, source_id))

    def set_source_hash(self, bank_id, source_id, digest):
        self.hashes[(bank_id, source_id)] = digest

    def append_log(self, bank_id, op, title):
        self.log.append((bank_id, utcnow(), op, title))

    def read_log(self, bank_id, limit=50):
        return [(t, o, ti) for b, t, o, ti in self.log if b == bank_id][-limit:]


# ── SQL ──────────────────────────────────────────────────────────────────────

try:  # SQLAlchemy is present in every backend, but keep the import soft for on-device tools.
    from sqlalchemy import (JSON, Boolean, Column, DateTime, Integer, MetaData, String, Table, Text,
                            and_, delete, insert, select, update)
    from sqlalchemy.engine import Connection, Engine

    metadata = MetaData()

    memory_units = Table(
        "sq_memory_units", metadata,
        Column("id", String(32), primary_key=True),
        Column("bank_id", String(128), nullable=False, index=True),
        Column("kind", String(16), nullable=False),
        Column("text", Text, nullable=False),
        Column("entities", JSON, nullable=False, default=list),
        Column("occurred_at", DateTime(timezone=True)),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("source", String(200), nullable=False, default=""),
        Column("key", String(200), nullable=False, default="", index=True),
        Column("bucket", String(64), nullable=False, default=""),
        Column("is_static", Boolean, nullable=False, default=False),
        Column("proof_count", Integer, nullable=False, default=1),
        Column("source_ids", JSON, nullable=False, default=list),
        Column("version", Integer, nullable=False, default=1),
        Column("is_latest", Boolean, nullable=False, default=True, index=True),
        Column("parent_id", String(32)),
        Column("relations", JSON, nullable=False, default=dict),
        Column("forget_after", DateTime(timezone=True), index=True),
        Column("embedding", JSON),
        Column("meta", JSON, nullable=False, default=dict),
    )
    wiki_pages = Table(
        "sq_wiki_pages", metadata,
        Column("bank_id", String(128), primary_key=True),
        Column("slug", String(200), primary_key=True),
        Column("title", String(300), nullable=False),
        Column("body", Text, nullable=False, default=""),
        Column("sources", JSON, nullable=False, default=list),
        Column("links", JSON, nullable=False, default=list),
        Column("content_hash", String(64), nullable=False, default=""),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    wiki_sources = Table(
        "sq_wiki_sources", metadata,
        Column("bank_id", String(128), primary_key=True),
        Column("source_id", String(200), primary_key=True),
        Column("content_hash", String(64), nullable=False),
        Column("ingested_at", DateTime(timezone=True), nullable=False),
    )
    wiki_log = Table(
        "sq_wiki_log", metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("bank_id", String(128), nullable=False, index=True),
        Column("at", DateTime(timezone=True), nullable=False),
        Column("op", String(32), nullable=False),
        Column("title", String(300), nullable=False),
    )
    HAVE_SQLALCHEMY = True
except ImportError:  # pragma: no cover
    HAVE_SQLALCHEMY = False


def create_tables(bind) -> None:
    """Create the memory/wiki tables if missing (idempotent)."""
    metadata.create_all(bind, checkfirst=True)


_FIELDS = ("id", "bank_id", "kind", "text", "entities", "occurred_at", "created_at", "source", "key",
           "bucket", "is_static", "proof_count", "source_ids", "version", "is_latest", "parent_id",
           "relations", "forget_after", "embedding")


def _row(u: MemoryUnit) -> dict[str, Any]:
    d = {f: getattr(u, f) for f in _FIELDS}
    d["meta"] = u.metadata
    return d


def _unit(r) -> MemoryUnit:
    m = r._mapping
    return MemoryUnit(**{f: m[f] for f in _FIELDS if f not in ("occurred_at", "created_at", "forget_after")},
                      occurred_at=_aware(m["occurred_at"]), created_at=_aware(m["created_at"]),
                      forget_after=_aware(m["forget_after"]), metadata=m["meta"] or {})


class SQLStore:
    def __init__(self, bind) -> None:
        self.bind = bind

    @contextmanager
    def _conn(self) -> Iterator["Connection"]:
        if isinstance(self.bind, Engine):
            with self.bind.begin() as c:
                yield c
        else:
            yield self.bind

    def add(self, unit: MemoryUnit) -> None:
        with self._conn() as c:
            c.execute(insert(memory_units).values(**_row(unit)))

    def update(self, unit: MemoryUnit) -> None:
        row = _row(unit); uid = row.pop("id")
        with self._conn() as c:
            c.execute(update(memory_units).where(memory_units.c.id == uid).values(**row))

    def get(self, unit_id):
        with self._conn() as c:
            r = c.execute(select(memory_units).where(memory_units.c.id == unit_id)).first()
        return _unit(r) if r else None

    def list(self, bank_id, kinds=None, latest_only=True, limit=None):
        q = select(memory_units).where(memory_units.c.bank_id == bank_id)
        if kinds:
            q = q.where(memory_units.c.kind.in_(list(kinds)))
        if latest_only:
            q = q.where(memory_units.c.is_latest.is_(True))
        q = q.order_by(memory_units.c.created_at.desc())
        if limit:
            q = q.limit(limit)
        with self._conn() as c:
            return [_unit(r) for r in c.execute(q)]

    def delete(self, unit_id):
        with self._conn() as c:
            c.execute(delete(memory_units).where(memory_units.c.id == unit_id))

    def delete_bank(self, bank_id) -> int:
        with self._conn() as c:
            n = c.execute(delete(memory_units).where(memory_units.c.bank_id == bank_id)).rowcount or 0
            for t in (wiki_pages, wiki_sources, wiki_log):
                c.execute(delete(t).where(t.c.bank_id == bank_id))
        return n

    def purge_expired(self, now=None) -> int:
        now = now or utcnow()
        with self._conn() as c:
            return c.execute(delete(memory_units).where(and_(memory_units.c.forget_after.is_not(None),
                                                             memory_units.c.forget_after <= now))).rowcount or 0

    def get_page(self, bank_id, slug):
        with self._conn() as c:
            r = c.execute(select(wiki_pages).where(and_(wiki_pages.c.bank_id == bank_id,
                                                        wiki_pages.c.slug == slug))).first()
        if not r:
            return None
        m = dict(r._mapping); m["updated_at"] = _aware(m["updated_at"])
        return WikiPage(**m)

    def put_page(self, page: WikiPage) -> None:
        vals = asdict(page)
        with self._conn() as c:
            c.execute(delete(wiki_pages).where(and_(wiki_pages.c.bank_id == page.bank_id,
                                                    wiki_pages.c.slug == page.slug)))
            c.execute(insert(wiki_pages).values(**vals))

    def list_pages(self, bank_id):
        with self._conn() as c:
            rows = c.execute(select(wiki_pages).where(wiki_pages.c.bank_id == bank_id).order_by(wiki_pages.c.slug))
            out = []
            for r in rows:
                m = dict(r._mapping); m["updated_at"] = _aware(m["updated_at"]); out.append(WikiPage(**m))
            return out

    def delete_page(self, bank_id, slug):
        with self._conn() as c:
            c.execute(delete(wiki_pages).where(and_(wiki_pages.c.bank_id == bank_id, wiki_pages.c.slug == slug)))

    def source_hash(self, bank_id, source_id):
        with self._conn() as c:
            r = c.execute(select(wiki_sources.c.content_hash).where(and_(
                wiki_sources.c.bank_id == bank_id, wiki_sources.c.source_id == source_id))).first()
        return r[0] if r else None

    def set_source_hash(self, bank_id, source_id, digest):
        with self._conn() as c:
            c.execute(delete(wiki_sources).where(and_(wiki_sources.c.bank_id == bank_id,
                                                      wiki_sources.c.source_id == source_id)))
            c.execute(insert(wiki_sources).values(bank_id=bank_id, source_id=source_id,
                                                  content_hash=digest, ingested_at=utcnow()))

    def append_log(self, bank_id, op, title):
        with self._conn() as c:
            c.execute(insert(wiki_log).values(bank_id=bank_id, at=utcnow(), op=op, title=title[:300]))

    def read_log(self, bank_id, limit=50):
        with self._conn() as c:
            rows = c.execute(select(wiki_log.c.at, wiki_log.c.op, wiki_log.c.title)
                             .where(wiki_log.c.bank_id == bank_id).order_by(wiki_log.c.id.desc()).limit(limit))
            return [(_aware(a), o, t) for a, o, t in rows][::-1]
