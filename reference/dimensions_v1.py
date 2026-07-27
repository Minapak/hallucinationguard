#!/usr/bin/env python3
"""HallucinationGuard — multi-dimension report (V1, zero-dependency).

Separate-dimension measurement, HELM-style (Liang et al., TMLR 2023): one
score blended across constructs hides trade-offs, so correctness, provenance,
citation integrity, and exposition depth are reported side by side and never
folded into a single number.

  correctness  : V0 tier + overlap + forbidden-claim gate (reference/score_v0.py)
  attribution  : near-verbatim KB reproduction without a KB-serving label
                 (First Proof second batch, Abouzaid et al. 2026 — uncited
                 line-by-line reuse "would have been flagged for plagiarism";
                 Leiden Declaration 2026, proper-attribution recommendation)
  citations    : structured identifiers (DOI / arXiv / FIPS) mentioned in the
                 answer, verified against the KB entry's reference corpus —
                 an identifier the corpus does not contain is a hallucinated
                 citation (First Proof: a system "hallucinates that a key
                 result exists in a previous paper" then "trivially deduces
                 the desired theorem" from it)
  depth        : hard-step coverage and hard-step sentence share, against the
                 item's annotated hard_step_concepts (Tao, ICM 2026: AI
                 exposition "dwells at length on trivialities, while passing
                 very briefly through … the most interesting and novel
                 portions"; First Proof: "glossing over the most difficult
                 steps, sometimes asserting that a key claim follows from
                 'standard arguments'")

Scope, stated plainly: every detector here is a lexical floor. The citation
check only sees structured identifiers (prose citations pass unseen); the
depth profile is a crude annotated-coverage proxy with hand-set thresholds,
and Goodhart's law applies to it as to every metric in this repo. The M2
harness replaces the depth proxy and augments attribution with an NLI/judge
backend behind the same `computed_flags()` interface; nothing in this module
assumes one exists today.

Usage:
  python3 reference/dimensions_v1.py fixtures/samples/hgx_provenance_6.json --locale en
  python3 reference/dimensions_v1.py fixtures/samples/hgx_provenance_6.json --locale ko --check-flags
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_v0 import (  # noqa: E402
    LOCALES, asserts_forbidden_claim, attribution_required, classify,
    _sentences, TIER_LOW_CONFIDENCE, TIER_VALIDATED,
)

FLAG_ATTRIBUTION = "attributionRequired"
FLAG_HALLUCINATED_CITATION = "hallucinatedCitation"
FLAG_DEPTH_INVERTED = "depthInverted"
KNOWN_FLAGS = {FLAG_ATTRIBUTION, FLAG_HALLUCINATED_CITATION, FLAG_DEPTH_INVERTED}

# Structured citation identifiers the zero-dependency checker can verify.
# Prose citations ("Smith et al. showed…") are out of scope — documented limit.
_DOI_RE = re.compile(r"(?:doi:\s*|doi\.org/)(10\.\d{4,9}/[^\s\"',;)\]}>]+)", re.IGNORECASE)
_ARXIV_RE = re.compile(r"arxiv:\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
_FIPS_RE = re.compile(r"fips[\s-]*(\d{3})", re.IGNORECASE)

# Depth-profile thresholds (crude, hand-set; tuned on the HGX05/HGX06 pair and
# subject to Goodhart's law — the M2 judge replaces this proxy):
# depthInverted = the hard step is *mentioned* (coverage >= 0.5) but barely
# *developed* (< 20% of answer characters live in hard-step sentences).
DEPTH_COVERAGE_MIN = 0.5
DEPTH_SHARE_FLOOR = 0.20


def load_kb_entries(kb_dir: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for path in sorted(kb_dir.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        entries[entry["id"]] = entry
    return entries


def kb_text_for(entry: dict | None, locale: str) -> str | None:
    if entry is None:
        return None
    translations = entry.get("translations", {})
    text = (translations.get(locale) or {}).get("detailedExplanation", "")
    return text or (translations.get("en") or {}).get("detailedExplanation", "") or None


def _reference_corpus(entry: dict) -> str:
    """Everything the KB entry legitimately references, casefolded: sourceRef,
    citationReferences, certaintyNote, and the en detailedExplanation (which
    may name standards/DOIs inline, e.g. ml_kem's FIPS sentences)."""
    parts = [entry.get("sourceRef", ""), entry.get("certaintyNote", "")]
    parts += entry.get("citationReferences", [])
    parts.append((entry.get("translations", {}).get("en") or {}).get("detailedExplanation", ""))
    reviewed = (entry.get("verifiedBy") or {}).get("reviewedSources", [])
    parts += reviewed if isinstance(reviewed, list) else []
    return " ".join(parts).casefold()


def extract_citations(text: str) -> list[str]:
    """Structured identifiers mentioned in the text, normalized."""
    t = text.casefold()
    cites = [f"doi:{m.rstrip('.')}" for m in _DOI_RE.findall(t)]
    cites += [f"arxiv:{m}" for m in _ARXIV_RE.findall(t)]
    cites += [f"fips {m}" for m in _FIPS_RE.findall(t)]
    return cites


def unverified_citations(answer_text: str, entry: dict | None) -> tuple[list[str], list[str]]:
    """Split the answer's structured identifiers into (unknown, known) against
    the KB entry's reference corpus. With no KB entry there is no corpus, so
    all identifiers are reported as unverifiable-but-unflagged (empty unknown)."""
    cites = extract_citations(answer_text)
    if entry is None:
        return [], cites
    corpus = _reference_corpus(entry)
    unknown, known = [], []
    for c in cites:
        ident = c.split(":", 1)[1] if ":" in c else c
        (known if (ident in corpus or c in corpus) else unknown).append(c)
    return unknown, known


def depth_profile(answer_text: str, hard_concepts: list[str]) -> tuple[float, float] | None:
    """(coverage, hard_share): fraction of annotated hard-step concepts the
    answer mentions (casefolded substring), and the share of answer characters
    in sentences mentioning at least one of them. None when unannotated."""
    if not hard_concepts:
        return None
    ans = answer_text.casefold()
    hits = [c for c in hard_concepts if c.casefold() in ans]
    coverage = len(hits) / len(hard_concepts)
    sents = _sentences(answer_text)
    total = sum(len(s) for s in sents) or 1
    hard_chars = sum(len(s) for s in sents
                     if any(c.casefold() in s.casefold() for c in hard_concepts))
    return coverage, hard_chars / total


def computed_flags(item: dict, answer: str, entry: dict | None, locale: str) -> set[str]:
    """The dimension flags V1 can compute. M2 judge backends extend this set
    behind the same signature; they never change the correctness tier."""
    flags: set[str] = set()
    kb_text = kb_text_for(entry, locale)
    tier, score = classify(kb_text, answer)
    if attribution_required(tier, score):
        flags.add(FLAG_ATTRIBUTION)
    unknown, _ = unverified_citations(answer, entry)
    if unknown:
        flags.add(FLAG_HALLUCINATED_CITATION)
    hard = (item.get("hard_step_concepts") or {}).get(locale) or []
    prof = depth_profile(answer, hard)
    if prof is not None:
        coverage, share = prof
        if coverage >= DEPTH_COVERAGE_MIN and share < DEPTH_SHARE_FLOOR:
            flags.add(FLAG_DEPTH_INVERTED)
    return flags


def report(fixture_path: Path, kb_dir: Path, locale: str, check_flags: bool) -> int:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    items = data["items"]
    if not items:
        print(f"error: {fixture_path} contains no items", file=sys.stderr)
        return 1
    entries = load_kb_entries(kb_dir)

    print(f"\nHallucinationGuard dimensions V1 — {fixture_path.name} · locale={locale}")
    print("(separate dimensions, never blended — correctness | attribution | citations | depth)\n")
    widths = [6, 24, 5, 6, 22, 13, 30]
    header = ("id", "tier", "ovl", "attr", "citations", "depth cov/shr", "flags vs expected")
    rows = [header]
    mismatches = 0
    for item in items:
        answer = item["ai_explanations"].get(locale) or item["ai_explanations"]["en"]
        entry = entries.get(item.get("kb_concept_id") or "")
        kb_text = kb_text_for(entry, locale)
        tier, score = classify(kb_text, answer)
        flagged = asserts_forbidden_claim(answer, item.get("forbidden_claims", []), locale)
        if flagged and kb_text is not None and tier == TIER_VALIDATED:
            tier = TIER_LOW_CONFIDENCE
        flags = computed_flags(item, answer, entry, locale)

        unknown, known = unverified_citations(answer, entry)
        if unknown:
            cite_cell = "HALLUCINATED: " + unknown[0]
        elif known:
            cite_cell = f"ok ({len(known)} verified)"
        else:
            cite_cell = "none found"
        hard = (item.get("hard_step_concepts") or {}).get(locale) or []
        prof = depth_profile(answer, hard)
        depth_cell = f"{prof[0]:.2f}/{prof[1]:.2f}" if prof else "n/a"

        expected = item.get("expected_flags")
        if expected is None:
            verdict = "·"
        else:
            ok = flags == set(expected)
            mismatches += not ok
            verdict = ("OK   " if ok else "MISS ") + (",".join(sorted(flags)) or "-") \
                + " / exp " + (",".join(sorted(expected)) or "-")
        rows.append((item["id"], tier, f"{score:.2f}",
                     "REQ" if FLAG_ATTRIBUTION in flags else "-",
                     cite_cell, depth_cell, verdict))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    print()
    if check_flags:
        n_expected = sum(1 for it in items if it.get("expected_flags") is not None)
        print(f"  flag agreement: {n_expected - mismatches}/{n_expected}")
        if mismatches:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="benchmark fixture JSON")
    parser.add_argument("--locale", default="en", choices=LOCALES)
    parser.add_argument("--kb", type=Path, default=None,
                        help="KB directory (default: fixtures/kb-examples next to fixture)")
    parser.add_argument("--check-flags", action="store_true",
                        help="exit non-zero unless computed flags match every item's expected_flags")
    args = parser.parse_args()
    kb_dir = args.kb or args.fixture.resolve().parent.parent / "kb-examples"
    if not kb_dir.is_dir():
        print(f"KB directory not found: {kb_dir}", file=sys.stderr)
        return 1
    try:
        return report(args.fixture, kb_dir, args.locale, args.check_flags)
    except OSError as e:
        print(f"error: cannot read fixture {args.fixture}: {e}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"error: {args.fixture} is not valid JSON: {e}", file=sys.stderr)
    except KeyError as e:
        print(f"error: {args.fixture} is missing required field {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
