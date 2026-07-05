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

    kb_ids = set()
    for path in sorted((REPO / "fixtures/kb-examples").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        kb_ids.add(entry["id"])
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

    for it in items:
        cid = it.get("kb_concept_id")
        if it["expected_source"] in ("aiValidatedAgainstKB", "aiWithLowConfidence"):
            check(cid in kb_ids, f"{it['id']}: kb_concept_id {cid!r} not bundled in kb-examples")

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
