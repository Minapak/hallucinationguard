#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eunmin Park
"""HallucinationGuard — KB concept wiki behind a verification gate (zero-dependency).

Builds an interlinked wiki from the verified knowledge base in
fixtures/kb-examples/*.json, using the shared `Wiki` class vendored in
reference/sq_memory/ (an independent implementation of Karpathy's LLM-Wiki
pattern: sources -> maintained interlinked pages, ingest / query / lint;
attribution in reference/sq_memory/NOTICE.md). Rules mode only: no LLM, no
network, stdlib only.

  sources  : one per KB concept, source_id = concept id
  entities : the 12 concept titles (in the build locale) found in the text,
             plus every DOI the entry cites (sourceRef / certaintyNote).
             Concepts therefore link through shared TERM pages
             ("Superposition" mentioned by Qubit, Decoherence, ...) and shared
             DOI pages (Qubit and Superposition both cite Nielsen & Chuang,
             doi:10.1017/cbo9780511976667).

Verification gate — nothing enters the wiki without passing it; a failing
candidate is kept in `unverified` with its reasons and scores, never ingested.

  KB concepts (build):
    G1 certainty  : grade must be `established` with >= 1 DOI in sourceRef /
                    certaintyNote, or `timebound` with an asOfDate (YYYY-MM-DD).
                    `verify`, `open` and null are not verified grades -> queued.
                    (Certainty is never auto-promoted — docs/SCHEMA.md.)
    G2 locale     : the build locale must have its own detailedExplanation
                    (no silent en fallback).
    G3 score_v0   : forbidden-claim gate (union of every sample fixture's
                    forbidden_claims, the way the consuming apps union them)
                    must not fire on any ingested field (checked per field —
                    on the concatenated text the gate's whole-text token recall
                    false-blocks 3 of the 12 verified concepts), and the KB text must
                    tokenize to >= MIN_KB_TOKENS. The V0 overlap tier of a KB
                    entry against its own text is trivially 1.0, so for KB
                    concepts score_v0 contributes the misconception gate, not
                    a truth check — they are expert-verified upstream.
  Proposed candidates (`propose()`):
    P1 anchor     : the named concept, or (none named) the concept with the
                    highest V0 one-sided overlap.
    P2 score_v0   : classify(anchor detailedExplanation, candidate) must be
                    verifiedKnowledgeBase or aiValidatedAgainstKB, i.e.
                    overlap |KB ∩ candidate| / |KB| >= THRESHOLD (0.30, the
                    production threshold). Consequence, stated plainly: a short
                    but correct candidate that covers < 30 % of the KB entry's
                    tokens is queued, not accepted — the gate is a lexical
                    floor that errs toward review.
    P3 misconception : asserts_forbidden_claim() must return None (union of
                    sample claims, locale-aware, NegEx-style negation guard).
    P4 citations  : every structured identifier (DOI / arXiv / FIPS) in the
                    candidate must exist in the anchor entry's reference corpus
                    (dimensions_v1.unverified_citations — hallucinatedCitation).
    P5 NLI (optional) : only when harness/nli_v1.py is importable (torch +
                    transformers installed) or a checker is passed in, and only
                    for its measured gate scope GATE_LOCALES {en, es, fr}: a
                    candidate whose sentence-level verdict is
                    `nli_contradicted` is queued. This uses the plain
                    contradiction label (stricter than the V2 gate, which only
                    acts on low-confidence answers that P2 already rejects), so
                    it inherits the measured false alarms on formula-dense
                    correct text (docs/NLI-GAP-2026-07-27.md) — acceptable here
                    because a false alarm queues for review, it deletes nothing.
                    Without the deps the layer is reported as "unavailable" and
                    skipped; the zero-dependency core never imports torch.

Usage:
  python3 -m reference.kb_wiki build [--json] [--locale en]
  python3 -m reference.kb_wiki lint  [--json]
  python3 -m reference.kb_wiki query "what does measurement do to a qubit" [--json]
  python3 -m reference.kb_wiki propose "<candidate text>" [--concept qubit] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
from score_v0 import (  # noqa: E402
    LOCALES, MIN_KB_TOKENS, THRESHOLD, TIER_VALIDATED, TIER_VERIFIED,
    asserts_forbidden_claim, attribution_required, classify, one_sided_overlap,
    tokenize_v0,
)
import dimensions_v1 as dv  # noqa: E402

try:  # package-relative when run as `python -m reference.kb_wiki` or imported as reference.kb_wiki
    from .sq_memory import InMemoryStore, Wiki
except ImportError:  # run as a plain script: python3 reference/kb_wiki.py
    sys.path.insert(0, str(REPO))
    from reference.sq_memory import InMemoryStore, Wiki  # noqa: E402

DEFAULT_BANK_ID = "hg:wiki:kb"
KB_DIR = REPO / "fixtures" / "kb-examples"
SAMPLES_DIR = REPO / "fixtures" / "samples"
VERIFIED_GRADES = ("established", "timebound")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class GateResult:
    """Outcome of the verification gate for one candidate."""
    kind: str                      # "concept" | "proposal"
    candidate_id: str
    concept_id: Optional[str]
    locale: str
    accepted: bool
    text: str
    reasons: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_forbidden_claims(samples_dir: Path = SAMPLES_DIR) -> dict:
    """Union of every sample fixture's forbidden_claims, per locale. Legacy
    English-only lists count for en only (score_v0 semantics)."""
    union: dict[str, list[str]] = {loc: [] for loc in LOCALES}
    for path in sorted(samples_dir.glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")).get("items", []):
            fc = item.get("forbidden_claims") or {}
            per = fc if isinstance(fc, dict) else {"en": fc}
            for loc, claims in per.items():
                for c in claims or []:
                    if c not in union.setdefault(loc, []):
                        union[loc].append(c)
    return union


def nli_available() -> bool:
    """True when the optional M2 NLI harness could run (torch + transformers)."""
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))
    except (ImportError, ValueError):  # a module blocked via sys.modules[m] = None
        return False


def _dois(text: str) -> list[str]:
    out: list[str] = []
    for c in dv.extract_citations(text):
        if c.startswith("doi:") and c not in out:
            out.append(c)
    return out


class KBWiki:
    """The verified KB as a wiki, with every write behind the gate."""

    def __init__(self, kb_dir: Path = KB_DIR, *, locale: str = "en",
                 bank_id: str = DEFAULT_BANK_ID, store: Any = None,
                 samples_dir: Path = SAMPLES_DIR, nli: Any = "auto") -> None:
        if locale not in LOCALES:
            raise ValueError(f"locale must be one of {LOCALES}")
        self.locale = locale
        self.entries = dv.load_kb_entries(Path(kb_dir))
        self.claims = load_forbidden_claims(Path(samples_dir))
        self.titles = {cid: self._title(e) for cid, e in self.entries.items()}
        self.store = store if store is not None else InMemoryStore()
        self.wiki = Wiki(self.store, bank_id, vocabulary=list(self.titles.values()))
        self.accepted: list[GateResult] = []
        self.unverified: list[GateResult] = []
        # nli: "auto" (use harness/nli_v1 if its deps are importable), False/None
        # (off), or an object with .check(kb_text, answer_text) -> verdict.label.
        self._nli_mode = nli
        self._nli_checker: Any = None if nli in ("auto", False, None) else nli

    # ── helpers ──────────────────────────────────────────────────────────────

    def _block(self, entry: dict) -> dict:
        return (entry.get("translations", {}).get(self.locale) or {})

    def _title(self, entry: dict) -> str:
        return (self._block(entry).get("title")
                or (entry.get("translations", {}).get("en") or {}).get("title")
                or entry["id"])

    def _reference_lines(self, entry: dict) -> list[str]:
        """One generated line per cited DOI, built only from the entry's own
        sourceRef / certaintyNote. Raw citation strings are not ingested: their
        abbreviations ("Phys. Rev. A") split into meaningless note fragments."""
        title = self.titles[entry["id"]]
        primary = _dois(entry.get("sourceRef", ""))
        lines = []
        for d in _dois(" ".join([entry.get("sourceRef", ""), entry.get("certaintyNote", "")])):
            where = "primary source" if d in primary else "cited in the certainty note"
            lines.append(f"{title} cites {d} ({where}).")
        return lines

    def concept_fields(self, entry: dict) -> list[str]:
        """The units ingested for one concept, each gated separately: short
        definition, detailed explanation, each key takeaway, and the DOI lines.
        Analogies and the negated misconception lists are left out: analogies
        are deliberately loose (the spinning-coin analogy is itself a
        catalogued misconception, HGM01) and do not belong on a fact page."""
        b = self._block(entry)
        parts = [b.get("shortDefinition", ""), b.get("detailedExplanation", "")]
        parts += list(b.get("keyTakeaways") or [])
        parts += self._reference_lines(entry)
        return [p.strip() for p in parts if p and p.strip()]

    def concept_text(self, entry: dict) -> str:
        return "\n".join(self.concept_fields(entry))

    def entities_for(self, text: str) -> list[str]:
        low = text.casefold()
        ents = [t for t in self.titles.values() if t.casefold() in low]
        return ents + [d for d in _dois(text) if d in low]

    def _nli(self) -> Any:
        if self._nli_checker is None and self._nli_mode == "auto" and nli_available():
            sys.path.insert(0, str(REPO / "harness"))
            from nli_v1 import NLIChecker  # noqa: E402  (optional, torch-bearing)
            self._nli_checker = NLIChecker()
        return self._nli_checker

    def _record(self, result: GateResult) -> GateResult:
        (self.accepted if result.accepted else self.unverified).append(result)
        return result

    # ── gate ─────────────────────────────────────────────────────────────────

    def gate_concept(self, entry: dict) -> GateResult:
        cid = entry["id"]
        text = self.concept_text(entry)
        reasons: list[str] = []
        grade = entry.get("certainty")
        refs = " ".join([entry.get("sourceRef", ""), entry.get("certaintyNote", "")])
        dois = _dois(refs)
        if grade not in VERIFIED_GRADES:
            reasons.append(f"certainty {grade!r} is not a verified grade")
        elif grade == "established" and not dois:
            reasons.append("established entry without a DOI in sourceRef/certaintyNote")
        elif grade == "timebound" and not _DATE.match(str(entry.get("asOfDate") or "")):
            reasons.append("timebound entry without asOfDate (YYYY-MM-DD)")
        kb_text = self._block(entry).get("detailedExplanation", "")
        if not kb_text:
            reasons.append(f"no {self.locale} detailedExplanation (no silent en fallback)")
        elif len(tokenize_v0(kb_text)) < MIN_KB_TOKENS:
            reasons.append(f"detailedExplanation tokenizes to < {MIN_KB_TOKENS} tokens")
        # Per field, not on the concatenation: the gate's token recall is taken
        # over the whole text, and on the ~1-2k-char concatenation scattered
        # tokens reach 0.60 — measured: 3 of 12 verified concepts false-block
        # concatenated (grover_algorithm, measurement, quantum_error_correction),
        # 0 of 12 per field. Production applies the gate to answer-sized text.
        forbidden = None
        for unit in self.concept_fields(entry):
            forbidden = asserts_forbidden_claim(unit, self.claims.get(self.locale, []), self.locale)
            if forbidden:
                reasons.append(f"asserts forbidden claim: {forbidden}")
                break
        scores = {"certainty": grade, "dois": dois, "forbidden_claim": forbidden}
        return GateResult("concept", cid, cid, self.locale, not reasons, text, reasons, scores)

    def gate_candidate(self, candidate_text: str, concept_id: Optional[str] = None) -> GateResult:
        cand = (candidate_text or "").strip()
        cand_id = "proposal-" + hashlib.sha256(cand.encode()).hexdigest()[:10]
        reasons: list[str] = []
        scores: dict = {}
        if not cand:
            return GateResult("proposal", cand_id, concept_id, self.locale, False, cand,
                              ["empty candidate"], scores)
        # P1 anchor
        if concept_id is None:
            ranked = sorted(((one_sided_overlap(dv.kb_text_for(e, self.locale) or "", cand), cid)
                             for cid, e in self.entries.items()), reverse=True)
            concept_id = ranked[0][1] if ranked and ranked[0][0] > 0 else None
            scores["anchor"] = "auto"
        else:
            scores["anchor"] = "given"
        entry = self.entries.get(concept_id) if concept_id else None
        if entry is None:
            reasons.append("no KB anchor" if concept_id is None else f"unknown concept {concept_id!r}")
            return GateResult("proposal", cand_id, concept_id, self.locale, False, cand, reasons, scores)
        kb_text = self._block(entry).get("detailedExplanation", "")
        if not kb_text:
            reasons.append(f"anchor has no {self.locale} detailedExplanation")
            return GateResult("proposal", cand_id, concept_id, self.locale, False, cand, reasons, scores)
        # P2 score_v0 tier
        tier, overlap = classify(kb_text, cand)
        scores.update(tier=tier, overlap=round(overlap, 4), threshold=THRESHOLD,
                      attribution_required=attribution_required(tier, overlap))
        if tier not in (TIER_VERIFIED, TIER_VALIDATED):
            reasons.append(f"score_v0 tier {tier} (overlap {overlap:.2f} < {THRESHOLD})")
        # P3 misconception gate
        forbidden = asserts_forbidden_claim(cand, self.claims.get(self.locale, []), self.locale)
        scores["forbidden_claim"] = forbidden
        if forbidden:
            reasons.append(f"asserts forbidden claim: {forbidden}")
        # P4 citation integrity
        unknown, known = dv.unverified_citations(cand, entry)
        scores.update(unknown_citations=unknown, known_citations=known)
        if unknown:
            reasons.append(f"citations not in the KB reference corpus: {', '.join(unknown)}")
        # P5 optional NLI
        checker = self._nli() if self._nli_mode not in (False, None) else None
        if checker is None:
            scores["nli"] = "unavailable" if self._nli_mode == "auto" else "off"
        else:
            sys.path.insert(0, str(REPO / "harness"))
            try:
                from nli_v1 import GATE_LOCALES
            except ImportError:  # injected checker without the harness deps
                GATE_LOCALES = frozenset({"en", "es", "fr"})
            label = checker.check(kb_text, cand).label
            scores["nli"] = label
            if label == "nli_contradicted" and self.locale in GATE_LOCALES:
                reasons.append("NLI: a candidate sentence is contradicted by the KB entry")
            elif self.locale not in GATE_LOCALES:
                scores["nli_scope"] = "shadow (locale outside GATE_LOCALES)"
        return GateResult("proposal", cand_id, concept_id, self.locale, not reasons, cand,
                          reasons, scores)

    # ── operations ───────────────────────────────────────────────────────────

    def build(self) -> "KBWiki":
        """Gate and ingest every KB concept (sorted by id)."""
        for cid in sorted(self.entries):
            entry = self.entries[cid]
            result = self._record(self.gate_concept(entry))
            if result.accepted:
                self.wiki.ingest(cid, result.text, title=self.titles[cid],
                                 entities=self.entities_for(result.text))
        return self

    def propose(self, candidate_text: str, concept_id: Optional[str] = None,
                *, title: str = "") -> GateResult:
        """Gate an externally proposed text. Accepted -> ingested as its own
        source ("proposal-<sha>") linked to the same term/DOI pages; rejected ->
        appended to `unverified` with reasons and scores, not ingested."""
        result = self._record(self.gate_candidate(candidate_text, concept_id))
        if result.accepted:
            anchor = self.titles.get(result.concept_id or "", "")
            self.wiki.ingest(result.candidate_id, result.text,
                             title=title or f"Proposal (gate-passed) — {anchor}",
                             entities=self.entities_for(result.text))
        return result

    def lint(self) -> list:
        return self.wiki.lint()

    def query(self, q: str, k: int = 5) -> list:
        return self.wiki.query(q, k=k)

    def index(self) -> str:
        return self.wiki.index()

    def counts(self) -> dict:
        pages = self.store.list_pages(self.wiki.bank_id)
        return {
            "pages": len(pages),
            "source_pages": sum(p.slug.startswith("source-") for p in pages),
            "doi_pages": sum(p.slug.startswith("doi-") for p in pages),
            "accepted": len(self.accepted),
            "unverified": len(self.unverified),
            "unverified_concepts": sum(r.kind == "concept" for r in self.unverified),
            "unverified_proposals": sum(r.kind == "proposal" for r in self.unverified),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _summary_line(c: dict) -> str:
    return (f"pages {c['pages']} (sources {c['source_pages']}, DOI pages {c['doi_pages']}) · "
            f"accepted {c['accepted']} · unverified queue {c['unverified']} "
            f"(concepts {c['unverified_concepts']}, proposals {c['unverified_proposals']})")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m reference.kb_wiki",
                                     description="KB concept wiki behind the HallucinationGuard gate.")
    parser.add_argument("command", choices=["build", "lint", "query", "propose"])
    parser.add_argument("text", nargs="?", default="", help="query text (query) or candidate text (propose)")
    parser.add_argument("--concept", default=None, help="anchor concept id for propose")
    parser.add_argument("--locale", default="en", choices=LOCALES)
    parser.add_argument("--kb", type=Path, default=KB_DIR, help="KB directory")
    parser.add_argument("-k", type=int, default=5, help="max query hits")
    parser.add_argument("--no-nli", action="store_true", help="skip the optional NLI layer")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if args.command in ("query", "propose") and not args.text.strip():
        parser.error(f"{args.command} needs a text argument")

    kw = KBWiki(args.kb, locale=args.locale, nli=False if args.no_nli else "auto").build()
    out: dict = {"command": args.command, "locale": args.locale, "bank_id": kw.wiki.bank_id}
    if args.command == "build":
        out["index"] = kw.index()
        out["pages"] = sorted(p.slug for p in kw.store.list_pages(kw.wiki.bank_id))
    elif args.command == "lint":
        out["issues"] = [asdict(i) for i in kw.lint()]
    elif args.command == "query":
        out["query"] = args.text
        out["hits"] = [{"slug": h.page.slug, "title": h.page.title, "score": round(h.score, 4),
                        "via": h.via, "sources": h.page.sources} for h in kw.query(args.text, args.k)]
    else:
        out["result"] = kw.propose(args.text, args.concept).to_dict()
    out["counts"] = kw.counts()
    out["unverified"] = [r.to_dict() for r in kw.unverified]

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    print(f"\nHallucinationGuard KB wiki — {out['bank_id']} · locale={args.locale}\n")
    if args.command == "build":
        print(out["index"])
    elif args.command == "lint":
        print(f"lint issues: {len(out['issues'])}")
        for i in out["issues"]:
            print(f"  {i['kind']:<12} {i['slug']}" + (f" -> {i['detail']}" if i["detail"] else ""))
    elif args.command == "query":
        print(f"query: {args.text}")
        for n, h in enumerate(out["hits"], 1):
            print(f"  [{n}] {h['title']}  ({h['slug']}, {h['via']}, score {h['score']:.3f}; "
                  f"sources: {', '.join(h['sources'])})")
        if not out["hits"]:
            print("  no hits")
    else:
        r = out["result"]
        print(("ACCEPTED" if r["accepted"] else "QUEUED as unverified") +
              f" — anchor {r['concept_id']}, {r['candidate_id']}")
        for reason in r["reasons"]:
            print(f"  - {reason}")
        print(f"  scores: {json.dumps(r['scores'], ensure_ascii=False)}")
    for r in out["unverified"]:
        print(f"  unverified {r['kind']} {r['candidate_id']}: {'; '.join(r['reasons'])}")
    print(f"\n  {_summary_line(out['counts'])}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
