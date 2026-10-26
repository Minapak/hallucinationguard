"""Shared memory layer for SwiftQuantum services.

``MemoryBank`` (retain / recall / reflect / profile / forget) and ``Wiki``
(ingest / query / lint). Pure Python on top of SQLAlchemy Core; an LLM and
an embedder are optional. Attribution: NOTICE.md.
"""

from .bank import BankConfig, MemoryBank
from .store import InMemoryStore, MemoryStore, SQLStore, WikiPage, create_tables
from .types import MemoryUnit, Profile, RecallHit, RecallResult, Reflection, RetainResult
from .wiki import IngestReport, LintIssue, Wiki, WikiHit, slugify

__version__ = "1.0.0"

__all__ = [
    "BankConfig", "MemoryBank", "InMemoryStore", "MemoryStore", "SQLStore", "WikiPage", "create_tables",
    "MemoryUnit", "Profile", "RecallHit", "RecallResult", "Reflection", "RetainResult",
    "IngestReport", "LintIssue", "Wiki", "WikiHit", "slugify",
]
