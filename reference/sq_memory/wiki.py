"""A wiki that the system keeps up to date from incoming sources.

Pattern: Andrej Karpathy, "LLM Wiki" (gist 442a6bf555914893e9891c11519de94f):
immutable sources → a maintained, interlinked set of pages → three
operations, *ingest*, *query* and *lint*, plus an index and an append-only
log. This is an independent implementation of that idea; no code, prompts
or formats were taken from any GPL implementation of it.

Pages are markdown. Each note line carries the id of the source it came
from (``<!--src:ID-->``), so re-ingesting or removing a source only touches
its own lines, and every statement on a page can be traced to a source.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import scoring
from .llm import LLM, parse_json_block
from .store import MemoryStore, WikiPage
from .text import extract_entities, mentions, split_sentences, tokenize
from .types import utcnow

_NOTE = re.compile(r"^- (.*?) <!--src:(.*?)-->$")
_LINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
SOURCE_PREFIX = "source-"

WIKI_PAGE_SYSTEM = (
    "You maintain one page of a personal knowledge wiki. Rewrite the page summary so it reflects all "
    "notes below, in 2-4 sentences, without adding facts that are not in the notes. Reply as JSON: "
    '{"summary":"..."}'
)


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKC", name).strip().lower()
    s = re.sub(r"[^\w぀-ヿ㐀-鿿가-힯]+", "-", s).strip("-")
    return s[:120] or "page"


@dataclass
class IngestReport:
    source_id: str
    skipped: bool = False
    pages_touched: list[str] = field(default_factory=list)
    pages_created: list[str] = field(default_factory=list)


@dataclass
class LintIssue:
    kind: str          # orphan | broken-link | no-outlinks | empty
    slug: str
    detail: str = ""


@dataclass
class WikiHit:
    page: WikiPage
    score: float
    via: str           # "match" or "link"


class Wiki:
    def __init__(self, store: MemoryStore, bank_id: str, *, llm: Optional[LLM] = None,
                 vocabulary: Optional[Iterable[str]] = None, max_notes_per_page: int = 60) -> None:
        self.store = store
        self.bank_id = bank_id
        self.llm = llm
        self.vocabulary = list(vocabulary or [])
        self.max_notes = max_notes_per_page

    # ── page body helpers ────────────────────────────────────────────────────

    @staticmethod
    def notes(page: WikiPage) -> list[tuple[str, str]]:
        out = []
        for line in page.body.splitlines():
            m = _NOTE.match(line.strip())
            if m:
                out.append((m.group(1), m.group(2)))
        return out

    @staticmethod
    def summary(page: WikiPage) -> str:
        m = re.search(r"^> (.+)$", page.body, re.M)
        return m.group(1) if m else ""

    def _render(self, title: str, summary: str, notes: list[tuple[str, str]], links: list[str]) -> str:
        parts = [f"# {title}", ""]
        if summary:
            parts += [f"> {summary}", ""]
        parts.append("## Notes")
        parts += [f"- {t} <!--src:{s}-->" for t, s in notes[-self.max_notes:]]
        if links:
            parts += ["", "## Related"] + [f"- [[{l}]]" for l in links]
        return "\n".join(parts) + "\n"

    def _write(self, slug: str, title: str, notes: list[tuple[str, str]], links: set[str],
               summary: Optional[str] = None) -> WikiPage:
        old = self.store.get_page(self.bank_id, slug)
        if summary is None:
            summary = self.summary(old) if old else ""
        links_sorted = sorted(l for l in links if l != slug)
        body = self._render(title, summary, notes, links_sorted)
        page = WikiPage(bank_id=self.bank_id, slug=slug, title=title, body=body,
                        sources=sorted({s for _, s in notes}), links=links_sorted,
                        content_hash=hashlib.sha256(body.encode()).hexdigest(), updated_at=utcnow())
        self.store.put_page(page)
        return page

    # ── ingest ───────────────────────────────────────────────────────────────

    def ingest(self, source_id: str, text: str, *, title: str = "", entities: Optional[list[str]] = None) -> IngestReport:
        """Fold one source into the wiki. Unchanged sources are skipped by hash."""
        digest = hashlib.sha256((title + "\n" + text).encode()).hexdigest()
        if self.store.source_hash(self.bank_id, source_id) == digest:
            return IngestReport(source_id, skipped=True)
        report = IngestReport(source_id)
        self._drop_source_notes(source_id, report)
        known = self.vocabulary + [p.title for p in self.store.list_pages(self.bank_id)
                                   if not p.slug.startswith(SOURCE_PREFIX)]
        ents = entities or extract_entities(text, known, max_entities=15)
        sentences = split_sentences(text)
        per_entity: dict[str, list[str]] = defaultdict(list)
        co: dict[str, set[str]] = defaultdict(set)
        for s in sentences:
            here = [e for e in ents if mentions(s, e)]
            for e in here:
                per_entity[e].append(s)
                co[slugify(e)].update(slugify(x) for x in here)
        src_slug = SOURCE_PREFIX + slugify(source_id)
        for e, ss in per_entity.items():
            slug = slugify(e)
            page = self.store.get_page(self.bank_id, slug)
            notes = self.notes(page) if page else []
            for s in ss[:6]:
                if (s, source_id) not in notes:
                    notes.append((s, source_id))
            links = set(page.links if page else []) | co[slug] | {src_slug}
            summary = self._llm_summary(e, notes)
            self._write(slug, page.title if page else e, notes, links, summary)
            (report.pages_touched if page else report.pages_created).append(slug)
        src_notes = [(s, source_id) for s in sentences[:8]]
        self._write(src_slug, title or source_id, src_notes, {slugify(e) for e in per_entity})
        report.pages_created.append(src_slug)
        self.store.set_source_hash(self.bank_id, source_id, digest)
        self.store.append_log(self.bank_id, "ingest", title or source_id)
        return report

    def _llm_summary(self, title: str, notes: list[tuple[str, str]]) -> Optional[str]:
        if self.llm is None:
            return None
        try:
            listing = "\n".join(f"- {t}" for t, _ in notes[-20:])
            data = parse_json_block(self.llm.complete(WIKI_PAGE_SYSTEM, f"Page: {title}\nNotes:\n{listing}",
                                                      max_tokens=250)) or {}
            s = str(data.get("summary", "")).strip().replace("\n", " ")
            return s[:800] or None
        except Exception:
            return None

    def _drop_source_notes(self, source_id: str, report: Optional[IngestReport] = None) -> None:
        for page in self.store.list_pages(self.bank_id):
            notes = self.notes(page)
            kept = [n for n in notes if n[1] != source_id]
            if len(kept) == len(notes):
                continue
            if not kept:
                self.store.delete_page(self.bank_id, page.slug)
                continue
            self._write(page.slug, page.title, kept, set(page.links))

    def remove_source(self, source_id: str) -> None:
        self._drop_source_notes(source_id)
        self.store.delete_page(self.bank_id, SOURCE_PREFIX + slugify(source_id))
        self.store.set_source_hash(self.bank_id, source_id, "")
        self.store.append_log(self.bank_id, "remove", source_id)

    # ── query ────────────────────────────────────────────────────────────────

    def query(self, question: str, k: int = 5, link_decay: float = 0.5) -> list[WikiHit]:
        pages = [p for p in self.store.list_pages(self.bank_id)]
        if not pages:
            return []
        docs = [tokenize(p.title) * 3 + tokenize(p.body) for p in pages]
        scores = scoring.bm25_scores(tokenize(question), docs)
        by_slug = {p.slug: i for i, p in enumerate(pages)}
        hits: dict[int, WikiHit] = {}
        for i, s in enumerate(scores):
            if s > 0:
                hits[i] = WikiHit(pages[i], s, "match")
        for i, h in list(hits.items()):
            for l in pages[i].links:
                j = by_slug.get(l)
                if j is not None and j not in hits:
                    hits[j] = WikiHit(pages[j], h.score * link_decay, "link")
        ordered = sorted(hits.values(), key=lambda h: -h.score)
        topics = [h for h in ordered if not h.page.slug.startswith(SOURCE_PREFIX)]
        return (topics or ordered)[:k]

    def context(self, question: str, k: int = 5) -> str:
        out = []
        for i, h in enumerate(self.query(question, k), 1):
            notes = "; ".join(t for t, _ in self.notes(h.page)[-5:])
            out.append(f"[{i}] {h.page.title}: {self.summary(h.page) or notes}")
        return "\n".join(out)

    # ── lint / index / log ───────────────────────────────────────────────────

    def lint(self) -> list[LintIssue]:
        pages = self.store.list_pages(self.bank_id)
        slugs = {p.slug for p in pages}
        inbound: dict[str, int] = defaultdict(int)
        issues: list[LintIssue] = []
        for p in pages:
            for l in p.links:
                if l in slugs:
                    inbound[l] += 1
                else:
                    issues.append(LintIssue("broken-link", p.slug, l))
            # only the Related section holds wiki links; note text may contain
            # brackets that are not links (e.g. the code notation [[144,12,12]])
            related = p.body.split("## Related", 1)[1] if "## Related" in p.body else ""
            for l in _LINK.findall(related):
                if slugify(l) not in slugs:
                    issues.append(LintIssue("broken-link", p.slug, l))
        for p in pages:
            if p.slug.startswith(SOURCE_PREFIX):
                continue
            if not inbound[p.slug]:
                issues.append(LintIssue("orphan", p.slug))
            if not [l for l in p.links if not l.startswith(SOURCE_PREFIX)]:
                issues.append(LintIssue("no-outlinks", p.slug))
            if not self.notes(p):
                issues.append(LintIssue("empty", p.slug))
        return issues

    def index(self) -> str:
        pages = self.store.list_pages(self.bank_id)
        topics = [p for p in pages if not p.slug.startswith(SOURCE_PREFIX)]
        sources = [p for p in pages if p.slug.startswith(SOURCE_PREFIX)]
        lines = ["# Index", "", f"## Topics ({len(topics)})"]
        for p in sorted(topics, key=lambda p: (-len(self.notes(p)), p.title.lower())):
            first = self.summary(p) or (self.notes(p)[0][0] if self.notes(p) else "")
            lines.append(f"- [[{p.slug}|{p.title}]] — {first[:120]}")
        lines += ["", f"## Sources ({len(sources)})"] + [f"- [[{p.slug}|{p.title}]]" for p in sources]
        return "\n".join(lines) + "\n"

    def log(self, limit: int = 50) -> str:
        return "\n".join(f"## [{t.date().isoformat()}] {op} | {title}" for t, op, title in self.store.read_log(self.bank_id, limit))

    def graph(self) -> dict:
        """Nodes and edges for a client-side graph view."""
        pages = self.store.list_pages(self.bank_id)
        slugs = {p.slug for p in pages}
        return {"nodes": [{"id": p.slug, "title": p.title, "source": p.slug.startswith(SOURCE_PREFIX),
                           "notes": len(self.notes(p))} for p in pages],
                "edges": [{"from": p.slug, "to": l} for p in pages for l in p.links if l in slugs]}
