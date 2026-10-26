"""Ranking arms and fusion.

Recall runs up to four arms and fuses them with reciprocal rank fusion
(RRF, Cormack et al. 2009), the same fusion hindsight uses:

* ``keyword``  — Okapi BM25 over the tokenised text
* ``semantic`` — cosine similarity, only when embeddings exist
* ``entity``   — overlap with the query's entities, plus one hop through
                 entities that co-occur with them (a small entity graph)
* ``temporal`` — facts inside a time window named in the query

The fused score is then multiplied by bounded boosts for recency and for
how many times a fact was confirmed (``proof_count``):
``boost = 1 + alpha * (signal - 0.5)``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

RRF_K = 60


def bm25_scores(query_tokens: Sequence[str], docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    n = len(docs)
    if n == 0 or not query_tokens:
        return [0.0] * n
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    q = Counter(query_tokens)
    out = []
    for d in docs:
        tf = Counter(d)
        dl = len(d) or 1
        s = 0.0
        for t, qf in q.items():
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = tf[t]
            s += qf * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out.append(s)
    return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def entity_scores(query_entities: Iterable[str], doc_entities: Sequence[Iterable[str]], hop_weight: float = 0.35) -> list[float]:
    """Direct overlap counts 1.0 per entity; a one-hop neighbour counts ``hop_weight``."""
    q = {e.lower() for e in query_entities}
    docs = [{e.lower() for e in d} for d in doc_entities]
    if not q:
        return [0.0] * len(docs)
    co: dict[str, Counter[str]] = defaultdict(Counter)
    for d in docs:
        for e in d:
            for f in d:
                if e != f:
                    co[e][f] += 1
    neighbours: set[str] = set()
    for e in q:
        neighbours.update(f for f, _ in co[e].most_common(8))
    neighbours -= q
    return [len(d & q) + hop_weight * len(d & neighbours) for d in docs]


def rank_positions(scores: Sequence[float]) -> dict[int, int]:
    """Index -> 1-based rank, only for positive scores."""
    order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
    return {i: r for r, i in enumerate(order, 1)}


def rrf(arms: dict[str, dict[int, int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = defaultdict(float)
    for ranks in arms.values():
        for i, r in ranks.items():
            fused[i] += 1.0 / (k + r)
    return dict(fused)


def recency_signal(when: Optional[datetime], now: datetime, half_life_days: float = 90.0) -> float:
    if when is None:
        return 0.5
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    age = max(0.0, (now - when).total_seconds() / 86400.0)
    return 0.5 ** (age / half_life_days)


def proof_signal(n: int) -> float:
    return min(1.0, 0.5 + math.log(max(1, n)) / 10.0)


def boost(signal: float, alpha: float) -> float:
    return 1.0 + alpha * (signal - 0.5)
