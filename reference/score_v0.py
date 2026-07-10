#!/usr/bin/env python3
"""HallucinationGuard — production V0 baseline scorer.

Reference implementation of the guard that gates AI tutor output in production.
Zero dependencies (stdlib only). Python 3.9+.

  Tokenizer V0 : case-folded whitespace split (no stemming, no punctuation stripping)
  Metric       : one-sided overlap  |KB ∩ answer| / |KB|
  Threshold    : 0.30 (empirically tuned on the HG120 set)
  Extra gate   : forbidden-claim assertion check (misconception patterns)

Usage:
  python3 reference/score_v0.py fixtures/samples/hg120_sample_20.json --locale en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THRESHOLD = 0.30
FORBIDDEN_ASSERT_THRESHOLD = 0.60  # token recall of a forbidden claim inside the answer

TIER_VERIFIED = "verifiedKnowledgeBase"
TIER_VALIDATED = "aiValidatedAgainstKB"
TIER_LOW_CONFIDENCE = "aiWithLowConfidence"
TIER_NO_KB = "aiNoKBMatch"


def tokenize_v0(text: str) -> set[str]:
    """Tokenizer V0: case-folded whitespace split. Deliberately naive — this IS the baseline."""
    return set(text.casefold().split())


def one_sided_overlap(kb_text: str, answer_text: str) -> float:
    """|KB ∩ answer| / |KB|. Returns 0.0 when the KB side is empty (no anchor to validate)."""
    kb = tokenize_v0(kb_text)
    if not kb:
        return 0.0
    return len(kb & tokenize_v0(answer_text)) / len(kb)


def asserts_forbidden_claim(answer_text: str, forbidden_claims: list[str]) -> str | None:
    """Return the first forbidden claim whose tokens are substantially present in the answer.

    Heuristic: token recall of the claim inside the answer ≥ FORBIDDEN_ASSERT_THRESHOLD.
    Deliberately simple; the point of the benchmark is to measure where simple fails.
    """
    ans = tokenize_v0(answer_text)
    for claim in forbidden_claims:
        claim_tokens = tokenize_v0(claim)
        if claim_tokens and len(claim_tokens & ans) / len(claim_tokens) >= FORBIDDEN_ASSERT_THRESHOLD:
            return claim
    return None


def classify(kb_text: str | None, answer_text: str) -> tuple[str, float]:
    """Three-tier classification per paper §III-C. Returns (tier, overlap_score)."""
    if kb_text is None:
        return TIER_NO_KB, 0.0
    score = one_sided_overlap(kb_text, answer_text)
    if score >= THRESHOLD:
        return TIER_VALIDATED, score
    return TIER_LOW_CONFIDENCE, score


def load_kb(kb_dir: Path, locale: str) -> dict[str, str]:
    """Map concept id -> detailedExplanation in the requested locale (fallback: en)."""
    kb: dict[str, str] = {}
    for path in sorted(kb_dir.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        translations = entry.get("translations", {})
        content = translations.get(locale) or translations.get("en") or {}
        text = content.get("detailedExplanation", "")
        if text:
            kb[entry["id"]] = text
    return kb


def score_fixture(fixture_path: Path, kb_dir: Path, locale: str) -> int:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    items = data["items"]
    kb = load_kb(kb_dir, locale)

    agree = 0
    rows = []
    for item in items:
        answer = item["ai_explanations"].get(locale) or item["ai_explanations"]["en"]
        kb_text = kb.get(item.get("kb_concept_id") or "")
        tier, score = classify(kb_text, answer)
        flagged = asserts_forbidden_claim(answer, item.get("forbidden_claims", []))
        expected = item["expected_source"]
        ok = tier == expected
        agree += ok
        rows.append((item["id"], item["category"], f"{score:.2f}", tier, expected,
                     "OK" if ok else "MISS", "BLOCKED: " + flagged[:40] + "…" if flagged else ""))

    widths = [6, 22, 5, 24, 24, 4]
    header = ("id", "category", "ovl", "guard tier", "ground truth", "", "forbidden-claim gate")
    print(f"\nHallucinationGuard V0 baseline — {fixture_path.name} · locale={locale}\n")
    for row in (header, *rows):
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)) + "  " + str(row[6] if len(row) > 6 else ""))
    n = len(items)
    print(f"\n  tier agreement with ground truth: {agree}/{n} = {agree / n:.4f}")
    print(f"  (full 120-item set, paper-cited production baseline: 0.9083 = 109/120)\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="benchmark fixture JSON")
    parser.add_argument("--locale", default="en",
                        choices=["en", "ko", "ja", "zh-Hans", "es", "fr", "de"])
    parser.add_argument("--kb", type=Path, default=None,
                        help="KB directory (default: fixtures/kb-examples next to fixture)")
    args = parser.parse_args()
    kb_dir = args.kb or args.fixture.parent.parent / "kb-examples"
    if not kb_dir.is_dir():
        print(f"KB directory not found: {kb_dir}", file=sys.stderr)
        return 1
    return score_fixture(args.fixture, kb_dir, args.locale)


if __name__ == "__main__":
    raise SystemExit(main())
