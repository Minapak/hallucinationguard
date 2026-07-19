#!/usr/bin/env python3
"""Fixture schema + certainty-gate checks. Stdlib only; exits non-zero on violation.

  python3 tests/test_fixtures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCALES = ["en", "ko", "ja", "zh-Hans", "es", "fr", "de"]
TIERS = {"verifiedKnowledgeBase", "aiValidatedAgainstKB", "aiWithLowConfidence", "aiNoKBMatch"}
CERTAINTY = {"established", "verify", "open", "timebound", None}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def meaningful_source_ref(ref: str) -> bool:
    """established gate: DOI, URL, or citation carrying a year — no empty shells."""
    if not ref or not ref.strip():
        return False
    return bool(re.search(r"doi:|https?://|\b(19|20)\d{2}\b", ref))


def main() -> int:
    fixture = json.loads((REPO / "fixtures/samples/hg120_sample_20.json").read_text(encoding="utf-8"))
    items = fixture["items"]
    check(len(items) >= 20, f"expected ≥20 sample items, got {len(items)}")
    ids = [it["id"] for it in items]
    check(len(set(ids)) == len(ids), "duplicate item ids")

    for it in items:
        for key in ("id", "category", "expected_source", "difficulty", "prompts", "ai_explanations"):
            check(key in it, f"{it.get('id', '?')}: missing {key}")
        check(it["expected_source"] in TIERS, f"{it['id']}: bad tier {it['expected_source']}")
        for loc in LOCALES:
            check(bool(it["prompts"].get(loc, "").strip()), f"{it['id']}: empty prompt [{loc}]")
            check(bool(it["ai_explanations"].get(loc, "").strip()), f"{it['id']}: empty answer [{loc}]")
        # Metadata fields (annotation/production-only; not consumed by the V0
        # reference scorer — see docs/SCHEMA.md) still get bounds-checked so
        # silent corruption cannot pass CI.
        check(isinstance(it.get("preskill_part"), int) and 1 <= it["preskill_part"] <= 24,
              f"{it['id']}: preskill_part out of range 1-24")
        check(isinstance(it.get("ai_confidence"), (int, float)) and 0 <= it["ai_confidence"] <= 1,
              f"{it['id']}: ai_confidence out of range 0-1")
        check(isinstance(it.get("key_concepts"), list) and it["key_concepts"]
              and all(isinstance(k, str) and k.strip() for k in it["key_concepts"]),
              f"{it['id']}: key_concepts must be a non-empty list of non-empty strings")
        # expected_keywords: [] allowed; entries must appear (case-insensitive
        # substring) in the en answer, per the field's SCHEMA.md definition.
        kws = it.get("expected_keywords")
        check(isinstance(kws, list) and all(isinstance(k, str) and k.strip() for k in kws),
              f"{it['id']}: expected_keywords must be a list of non-empty strings")
        en_answer = it["ai_explanations"].get("en", "").casefold()
        for kw in kws or []:
            check(kw.casefold() in en_answer,
                  f"{it['id']}: expected_keyword {kw!r} absent from en answer")
        # forbidden_claims: either an empty list (no claims) or a 7-locale dict
        # of non-empty claim lists (schema v3.1) — an English-only list would
        # leave the misconception gate inert for 6 of 7 locales.
        fc = it.get("forbidden_claims", [])
        if fc:
            check(isinstance(fc, dict) and set(fc) >= set(LOCALES)
                  and all(isinstance(fc[loc], list) and fc[loc]
                          and all(isinstance(c, str) and c.strip() for c in fc[loc])
                          for loc in LOCALES),
                  f"{it['id']}: non-empty forbidden_claims must carry all locales {LOCALES}")

    dist = fixture.get("sample_distribution", {})
    actual_dist: dict = {}
    for it in items:
        actual_dist[it["expected_source"]] = actual_dist.get(it["expected_source"], 0) + 1
    check(dist == actual_dist, f"sample_distribution {dist} != actual {actual_dist}")
    cats = fixture.get("categories_in_sample", [])
    check(sorted(cats) == sorted({it["category"] for it in items}),
          "categories_in_sample does not match actual item categories")

    # Translation-parity floors for KB detailedExplanation, in chars vs en.
    # Motivation: stub translations shrink the overlap denominator until
    # stopwords cross the 0.30 threshold (audited false validation of HG039 in
    # es/fr against a 13-token Grover stub). Floors are script-density-aware:
    # CJK scripts carry more content per char than Latin.
    PARITY_FLOOR = {"ko": 0.40, "ja": 0.25, "zh-Hans": 0.18, "es": 0.50, "fr": 0.50, "de": 0.50}
    ACCENTED = set("áéíóúñüàâçèêëîïôùûüœÁÉÍÓÚÑÜÀÂÇÈÊËÎÏÔÙÛÜŒ")

    kb_ids = set()
    kb_categories: dict = {}
    for path in sorted((REPO / "fixtures/kb-examples").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        kb_ids.add(entry["id"])
        kb_categories[entry["id"]] = entry.get("category")
        cert = entry.get("certainty")
        check(cert in CERTAINTY, f"{entry['id']}: bad certainty {cert!r}")
        if cert == "established":
            check(meaningful_source_ref(entry.get("sourceRef", "")),
                  f"{entry['id']}: established requires meaningful sourceRef")
        if cert == "timebound":
            check(bool(DATE_RE.match(entry.get("asOfDate", ""))),
                  f"{entry['id']}: timebound requires asOfDate YYYY-MM-DD")
        translations = entry.get("translations", {})
        for loc in LOCALES:
            check(loc in translations, f"{entry['id']}: missing locale {loc}")
        en_len = len(translations.get("en", {}).get("detailedExplanation", ""))
        if en_len:
            for loc, floor in PARITY_FLOOR.items():
                loc_len = len(translations.get(loc, {}).get("detailedExplanation", ""))
                check(loc_len >= floor * en_len,
                      f"{entry['id']}: {loc} detailedExplanation is {loc_len} chars "
                      f"({loc_len / en_len:.0%} of en) — below the {floor:.0%} parity floor")
        # Orthography gate: a full es/fr paragraph without a single accented
        # character is unaccented text masquerading as a verified translation.
        for loc in ("es", "fr"):
            text = translations.get(loc, {}).get("detailedExplanation", "")
            if len(text) > 200:
                check(any(ch in ACCENTED for ch in text),
                      f"{entry['id']}: {loc} detailedExplanation contains no accented "
                      f"characters — diacritics missing")

    for it in items:
        cid = it.get("kb_concept_id")
        if it["expected_source"] in ("aiValidatedAgainstKB", "aiWithLowConfidence"):
            check(cid in kb_ids, f"{it['id']}: kb_concept_id {cid!r} not bundled in kb-examples")
        if cid:
            check(it.get("kb_category") == kb_categories.get(cid),
                  f"{it['id']}: kb_category {it.get('kb_category')!r} != KB entry "
                  f"category {kb_categories.get(cid)!r}")

    field_count = sum(len(it["prompts"]) + len(it["ai_explanations"]) for it in items)
    print(f"items: {len(items)} · multilingual fields: {field_count} · KB entries: {len(kb_ids)}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all fixture checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
