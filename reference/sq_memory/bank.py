"""MemoryBank: retain → recall → reflect, plus profile and forgetting.

One bank per subject (a user, an org, a class). Bank ids are opaque strings
chosen by the service, e.g. ``"qn:user:42"``; never put an email in them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional, Sequence

from . import scoring
from .llm import LLM, OBSERVE_SYSTEM, REFLECT_SYSTEM, Embedder, extract_facts
from .store import MemoryStore
from .text import (approx_tokens, extract_entities, find_date, jaccard, parse_time_range,
                   split_sentences, tokenize)
from .types import (MemoryUnit, Profile, RecallHit, RecallResult, Reflection, RetainResult, new_id,
                    utcnow)


@dataclass
class BankConfig:
    max_units: int = 2000            # per bank; oldest low-value units are pruned past this
    duplicate_threshold: float = 0.85
    extend_threshold: float = 0.45
    candidate_limit: int = 2000      # units loaded for a recall
    recency_alpha: float = 0.2
    proof_alpha: float = 0.1
    temporal_alpha: float = 0.2
    half_life_days: float = 90.0
    default_ttl_days: Optional[int] = None   # e.g. 2 for a 48-hour retention promise
    observation_min_facts: int = 2


class MemoryBank:
    def __init__(self, store: MemoryStore, bank_id: str, *, llm: Optional[LLM] = None,
                 embedder: Optional[Embedder] = None, vocabulary: Optional[Iterable[str]] = None,
                 config: Optional[BankConfig] = None, disposition: Optional[dict[str, int]] = None) -> None:
        if not bank_id or "@" in bank_id:
            raise ValueError("bank_id must be an opaque id, not an email")
        self.store = store
        self.bank_id = bank_id
        self.llm = llm
        self.embedder = embedder
        self.vocabulary = list(vocabulary or [])
        self.cfg = config or BankConfig()
        self.disposition = {"skepticism": 3, "literalism": 3, "empathy": 3, **(disposition or {})}

    # ── retain ───────────────────────────────────────────────────────────────

    def retain(self, content: str = "", *, facts: Optional[Sequence[dict[str, Any]]] = None,
               kind: str = "world", source: str = "", occurred_at: Optional[datetime] = None,
               key: str = "", bucket: str = "", is_static: bool = False,
               ttl_days: Optional[int] = None, use_llm: bool = True,
               metadata: Optional[dict[str, Any]] = None) -> RetainResult:
        """Store facts from ``content`` (split into sentences) or from ``facts``.

        A fact with the same ``key`` as a live fact *updates* it (new version,
        old one kept with ``is_latest=False``). A near-identical fact only
        raises ``proof_count``. A related fact is linked with ``extends``.
        """
        if facts is None:
            if self.llm is not None and use_llm and content:
                try:
                    facts = extract_facts(self.llm, content)
                except Exception:
                    facts = None
            if not facts:
                sentences = split_sentences(content) if not key else [content.strip()]
                facts = [{"text": s} for s in sentences if s]
        ttl = ttl_days if ttl_days is not None else self.cfg.default_ttl_days
        now = utcnow()
        existing = self.store.list(self.bank_id, latest_only=True, limit=self.cfg.candidate_limit)
        existing = [u for u in existing if u.kind != "observation"]
        result = RetainResult([], [], [])
        new_units: list[MemoryUnit] = []
        for f in facts:
            text = str(f.get("text", "")).strip()[:2000]
            if not text:
                continue
            u = MemoryUnit(
                bank_id=self.bank_id, text=text,
                kind=f.get("kind") or kind,
                entities=list(f.get("entities") or extract_entities(text, self.vocabulary)),
                occurred_at=f.get("occurred_at") or occurred_at or find_date(text) or now,
                source=f.get("source") or source, key=f.get("key") or key,
                bucket=f.get("bucket") or bucket, is_static=bool(f.get("is_static", is_static)),
                forget_after=(now + timedelta(days=ttl)) if ttl else None,
                metadata=dict(metadata or {}),
            )
            toks = tokenize(text)
            if u.key:
                old = next((e for e in existing if e.key == u.key), None)
                if old is not None:
                    if old.text.strip().lower() == text.lower():
                        old.proof_count += 1; self.store.update(old); result.reinforced.append(old)
                        continue
                    old.is_latest = False
                    self.store.update(old)
                    existing.remove(old)
                    u.version = old.version + 1; u.parent_id = old.id
                    u.relations[old.id] = "updates"
                    result.superseded.append(old)
            else:
                dup = None
                best_rel = None
                for e in existing:
                    s = jaccard(toks, tokenize(e.text))
                    if s >= self.cfg.duplicate_threshold:
                        dup = e; break
                    if s >= self.cfg.extend_threshold and (best_rel is None or s > best_rel[0]):
                        best_rel = (s, e)
                if dup is not None:
                    dup.proof_count += 1
                    if u.source and u.source not in dup.source_ids:
                        dup.source_ids.append(u.source)
                    self.store.update(dup); result.reinforced.append(dup)
                    continue
                if best_rel is not None:
                    u.relations[best_rel[1].id] = "extends"
            new_units.append(u)
            existing.insert(0, u)
        if new_units and self.embedder is not None:
            try:
                for u, v in zip(new_units, self.embedder.embed([u.text for u in new_units])):
                    u.embedding = v
            except Exception:
                pass
        for u in new_units:
            self.store.add(u)
        result.created.extend(new_units)
        self._prune()
        return result

    def _prune(self) -> None:
        units = self.store.list(self.bank_id, latest_only=False)
        if len(units) <= self.cfg.max_units:
            return
        # drop superseded versions first, then the oldest single-proof non-static facts
        victims = sorted(units, key=lambda u: (u.is_latest, u.is_static, u.proof_count,
                                               scoring_aware(u.created_at)))
        for u in victims[: len(units) - self.cfg.max_units]:
            self.store.delete(u.id)

    # ── recall ───────────────────────────────────────────────────────────────

    def recall(self, query: str, *, k: int = 8, kinds: Optional[Sequence[str]] = None,
               max_tokens: int = 1200, now: Optional[datetime] = None,
               min_score: float = 0.0) -> RecallResult:
        now = now or utcnow()
        units = [u for u in self.store.list(self.bank_id, kinds=kinds, latest_only=True,
                                             limit=self.cfg.candidate_limit) if not u.expired(now)]
        if not units:
            return RecallResult(query, [], [])
        arms: dict[str, dict[int, int]] = {}
        qtok = tokenize(query)
        arms["keyword"] = scoring.rank_positions(scoring.bm25_scores(qtok, [tokenize(u.text) for u in units]))
        q_ents = extract_entities(query, self.vocabulary + sorted({e for u in units for e in u.entities}))
        if q_ents:
            arms["entity"] = scoring.rank_positions(scoring.entity_scores(q_ents, [u.entities for u in units]))
        if self.embedder is not None and any(u.embedding for u in units):
            try:
                qv = self.embedder.embed([query])[0]
                arms["semantic"] = scoring.rank_positions(
                    [scoring.cosine(qv, u.embedding) if u.embedding else 0.0 for u in units])
            except Exception:
                pass
        window = parse_time_range(query, now)
        if window:
            lo, hi = window
            inside = [(1.0 / (1 + abs((scoring_aware(u.occurred_at) - hi).total_seconds()) / 86400.0))
                      if u.occurred_at and lo <= scoring_aware(u.occurred_at) <= hi else 0.0 for u in units]
            arms["temporal"] = scoring.rank_positions(inside)
        arms = {n: r for n, r in arms.items() if r}
        if not arms:
            return RecallResult(query, [], [])
        fused = scoring.rrf(arms)
        hits = []
        for i, base in fused.items():
            u = units[i]
            s = base * scoring.boost(scoring.recency_signal(u.occurred_at or u.created_at, now, self.cfg.half_life_days),
                                     self.cfg.recency_alpha)
            s *= scoring.boost(scoring.proof_signal(u.proof_count), self.cfg.proof_alpha)
            if "temporal" in arms and i in arms["temporal"]:
                s *= 1 + self.cfg.temporal_alpha
            if s >= min_score:
                hits.append(RecallHit(u, s, {n: r[i] for n, r in arms.items() if i in r}))
        hits.sort(key=lambda h: -h.score)
        out, used = [], 0
        for h in hits:
            t = approx_tokens(h.unit.text)
            if out and used + t > max_tokens:
                break
            out.append(h); used += t
            if len(out) >= k:
                break
        return RecallResult(query, out, list(arms))

    # ── consolidation (observations) ─────────────────────────────────────────

    def consolidate(self, max_subjects: int = 20) -> list[MemoryUnit]:
        """Merge facts that share an entity into ``observation`` units.

        Re-running refreshes an observation in place (new version) when its
        supporting facts changed. Rules-only text when no LLM is configured.
        """
        facts = self.store.list(self.bank_id, kinds=("world", "experience"), latest_only=True)
        observations = {u.key: u for u in self.store.list(self.bank_id, kinds=("observation",)) if u.key}
        by_entity: dict[str, list[MemoryUnit]] = defaultdict(list)
        for u in facts:
            for e in u.entities:
                by_entity[e.lower()].append(u)
        subjects = sorted(((e, us) for e, us in by_entity.items() if len(us) >= self.cfg.observation_min_facts),
                          key=lambda x: -len(x[1]))[:max_subjects]
        made = []
        for ent, us in subjects:
            us = sorted(us, key=lambda u: scoring_aware(u.occurred_at or u.created_at))[-12:]
            ids = sorted(u.id for u in us)
            key = f"observation:{ent}"
            old = observations.get(key)
            if old is not None and sorted(old.source_ids) == ids:
                continue
            name = next((e for e in us[-1].entities if e.lower() == ent), ent)
            text, by = self._summarise(name, us)
            o = MemoryUnit(bank_id=self.bank_id, text=text, kind="observation", entities=[name], key=key,
                           proof_count=len(us), source_ids=ids, occurred_at=us[-1].occurred_at,
                           relations={i: "derives" for i in ids}, metadata={"generated_by": by})
            if old is not None:
                old.is_latest = False; self.store.update(old)
                o.version = old.version + 1; o.parent_id = old.id
            self.store.add(o); made.append(o)
        return made

    def _summarise(self, name: str, units: Sequence[MemoryUnit]) -> tuple[str, str]:
        if self.llm is not None:
            try:
                numbered = "\n".join(f"[{i}] {u.text}" for i, u in enumerate(units, 1))
                txt = self.llm.complete(OBSERVE_SYSTEM, f"Subject: {name}\n{numbered}", max_tokens=160).strip()
                if txt:
                    return txt[:600], f"llm:{self.llm.model}"
            except Exception:
                pass
        latest = [u.text.rstrip(".") for u in units[-3:]]
        return f"{name}: " + "; ".join(latest) + f" ({len(units)} related memories)", "rules"

    # ── reflect ──────────────────────────────────────────────────────────────

    def reflect(self, query: str, *, k: int = 8, store_opinion: bool = False) -> Reflection:
        """Answer from memory: observations first, raw facts as ground truth."""
        obs = self.recall(query, k=max(2, k // 3), kinds=("observation",))
        raw = self.recall(query, k=k, kinds=("world", "experience"))
        seen, hits = set(), []
        for h in obs.hits + raw.hits:
            if h.unit.id not in seen:
                seen.add(h.unit.id); hits.append(h)
        hits = hits[:k]
        cites = [h.unit for h in hits]
        if not cites:
            return Reflection(query, "", [], "rules")
        context = RecallResult(query, hits, obs.arms_used + raw.arms_used).as_context()
        answer, by = "", "rules"
        if self.llm is not None:
            try:
                answer = self.llm.complete(REFLECT_SYSTEM.format(**self.disposition),
                                           f"Memories:\n{context}\n\nQuestion: {query}", max_tokens=400).strip()
                by = f"llm:{self.llm.model}"
            except Exception:
                answer = ""
        if not answer:
            answer = " ".join(f"{u.text.rstrip('.')} [{i}]." for i, u in enumerate(cites[:3], 1))
            by = "rules"
        stored = None
        if store_opinion and by != "rules":
            stored = MemoryUnit(bank_id=self.bank_id, text=answer[:1000], kind="opinion",
                                source_ids=[u.id for u in cites], relations={u.id: "derives" for u in cites},
                                metadata={"query": query[:300], "generated_by": by})
            self.store.add(stored)
        return Reflection(query, answer, cites, by, stored)

    # ── profile / forget / export ────────────────────────────────────────────

    def profile(self, *, dynamic_days: int = 30, dynamic_limit: int = 8, now: Optional[datetime] = None) -> Profile:
        now = now or utcnow()
        units = [u for u in self.store.list(self.bank_id, kinds=("world", "experience"), latest_only=True)
                 if not u.expired(now)]
        static = [u for u in units if u.is_static or u.key.startswith("profile:") or u.proof_count >= 3]
        cutoff = now - timedelta(days=dynamic_days)
        dynamic = [u for u in units if u not in static and scoring_aware(u.created_at) >= cutoff][:dynamic_limit]
        buckets: dict[str, list[MemoryUnit]] = defaultdict(list)
        for u in units:
            if u.bucket:
                buckets[u.bucket].append(u)
        return Profile(static, dynamic, dict(buckets))

    def forget(self, unit_id: Optional[str] = None) -> int:
        """Delete one unit, or the whole bank (memories and wiki) when no id is given."""
        if unit_id:
            u = self.store.get(unit_id)
            if u is None or u.bank_id != self.bank_id:
                return 0
            self.store.delete(unit_id)
            return 1
        return self.store.delete_bank(self.bank_id)

    def export(self) -> list[dict[str, Any]]:
        return [{"id": u.id, "kind": u.kind, "text": u.text, "entities": u.entities,
                 "occurred_at": u.occurred_at.isoformat() if u.occurred_at else None,
                 "created_at": u.created_at.isoformat(), "source": u.source, "key": u.key,
                 "version": u.version, "is_latest": u.is_latest, "proof_count": u.proof_count,
                 "relations": u.relations}
                for u in self.store.list(self.bank_id, latest_only=False)]


def scoring_aware(d: Optional[datetime]) -> datetime:
    from datetime import timezone
    if d is None:
        return utcnow()
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
