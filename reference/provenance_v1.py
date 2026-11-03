#!/usr/bin/env python3
"""HallucinationGuard — Provenance Ink, sentence-level verdicts with a receipt (V1).

The four-tier answer verdict (score_v0.classify + the forbidden-claim gate) is the
product's contract and is NOT changed here: this module runs beside it and adds
resolution, not a new verdict. What it adds:

  1. **Sentence resolution.** Every answer sentence gets its own anchor (the KB
     sentence it overlaps most, same V0.1 tokenizer, same one-sided overlap
     |anchor ∩ sentence| / |anchor|, same 0.30 floor) and its own ink state —
     so a reader sees WHICH sentence is unconfirmed instead of re-reading all
     of them after a whole-answer badge.
  2. **Silence becomes a state.** `unanchored` (no KB sentence shares enough
     vocabulary to compare against) is reported, not folded into "low":
     "we had nothing to check against" ≠ "we checked and it disagrees"
     (the same distinction v0.2.0 drew between aiNoKBMatch and aiWithLowConfidence,
     now per sentence).
  3. **The guard can say "I don't know".** When the lexical anchor says yes
     (overlap ≥ 0.30) and the optional NLI layer says no — bidirectionally
     confirmed contradiction (both directions ≥ 0.50) on a sentence with
     support < 0.50, in a measured-clean locale (en/es/fr, docs/NLI-GAP-2026-07-27.md
     §6) — the sentence is `disagree`: two checkers differ, so the guard reports
     the disagreement instead of picking one and looking certain. Evaluation that
     rewards a confident guess over an admitted uncertainty manufactures the
     exact failure a guard exists to catch (Kalai et al., OpenAI 2025; "Evaluating
     large language models for accuracy incentivizes hallucinations", Nature 2026).
     Applied to the guard itself.
  4. **A receipt.** Per answer: sha256 of answer and KB, the unchanged tier,
     per-state counts, which checks ran, and every sentence's anchor + scores.
     "Every claim traced upstream" becomes a thing the reader can open.

Ink states (one per sentence, conservative order — a sentence takes the
weakest state that applies):

  verified    sentence text == a KB sentence (normalized equality; the serving-
              path fact v0.2.0 reserves for verbatim KB text)
  validated   anchor overlap ≥ 0.30, no forbidden claim, NLI absent / supports /
              not bidirectionally contradicting
  disagree    anchor overlap ≥ 0.30 AND NLI bidirectional contradiction (gate
              rule R7) in a measured-clean locale — lexical yes, semantic no
  low         forbidden claim asserted in this sentence, or an anchor exists
              (overlap ≥ ANCHOR_MIN) but below the 0.30 floor
  unanchored  best overlap < ANCHOR_MIN — nothing in the KB to compare against

Zero dependencies (stdlib). The NLI layer is injected as a callable
`scores_batch([(premise, hypothesis), ...]) -> [{"entailment": p, "contradiction": p, ...}]`
— `harness/nli_v1.NLIChecker.scores_batch` fits; absent, `nli` is null in the receipt
and `disagree` can never fire (the guard does not pretend to have run a check it
did not run).

Usage:
  python3 reference/provenance_v1.py --kb "<kb text>" --answer "<answer>" [--locale en] [--json]
  python3 reference/provenance_v1.py fixtures/samples/hg120_sample_20.json --locale en
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_v0 import (  # noqa: E402
    LOCALES, THRESHOLD, TIER_LOW_CONFIDENCE, TIER_VALIDATED, _sentences,
    asserts_forbidden_claim, classify, load_kb, one_sided_overlap, tokenize_v0,
)

PROVENANCE_VERSION = "provenance-v1"

STATE_VERIFIED = "verified"
STATE_VALIDATED = "validated"
STATE_DISAGREE = "disagree"
STATE_LOW = "low"
STATE_UNANCHORED = "unanchored"
STATES = (STATE_VERIFIED, STATE_VALIDATED, STATE_DISAGREE, STATE_LOW, STATE_UNANCHORED)

# An anchor exists only when the sentence shares a CONTENT token with some KB
# sentence (a token of >= CONTENT_TOKEN_MIN characters, or any CJK bigram) and
# that KB sentence's one-sided overlap is >= ANCHOR_MIN. Function words alone
# ("the", "is", "a") make a 0.29 overlap against a seven-token KB sentence —
# topical noise, not an anchor. The overlap NUMBER stays the V0 metric; only
# the existence test for an anchor looks at token length.
ANCHOR_MIN = 0.10
CONTENT_TOKEN_MIN = 4
# Sentences shorter than this (characters) are fragments ("Yes.", "Q.E.D.")
# and are skipped, matching hg_nli_shadow's minimum.
MIN_SENTENCE_CHARS = 8

# NLI thresholds and the measured-clean locale scope — lockstep with
# harness/nli_v1.py (rule R7). Outside GATE_LOCALES the NLI scores are still
# recorded in the receipt but never produce `disagree` (ko/ja/zh-Hans/de still
# false-block faithful answers; docs/NLI-GAP-2026-07-27.md §6).
CONTRA_MIN = 0.50
SUPPORT_EXEMPT = 0.50
GATE_LOCALES = frozenset({"en", "es", "fr"})

ScoresBatch = Callable[[list], list]


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _content(tokens: set[str]) -> set[str]:
    return {t for t in tokens if len(t) >= CONTENT_TOKEN_MIN or any(ord(c) > 0x2E7F for c in t)}


def anchor_for(sentence: str, kb_sentences: list[str]) -> tuple[Optional[int], float]:
    """(index of the KB sentence with the highest one-sided overlap among those
    sharing a content token with the sentence, that overlap). Ties go to the
    earlier KB sentence (deterministic). (None, 0.0) when no KB sentence shares
    a content token."""
    best_i, best = None, 0.0
    sc = _content(tokenize_v0(sentence))
    for i, kb in enumerate(kb_sentences):
        if not (_content(tokenize_v0(kb)) & sc):
            continue
        ov = one_sided_overlap(kb, sentence)
        if ov > best:
            best_i, best = i, ov
    return best_i, best


def _nli_scores(nli: Optional[ScoresBatch], kb_sentences: list[str], sentence: str
                ) -> Optional[dict[str, float]]:
    """support = max entailment over KB sentences; conflict = max contradiction
    (forward); conflict_rev = reverse-direction contradiction of the argmax pair,
    computed only when the forward score is a gate candidate (R7)."""
    if nli is None or not kb_sentences:
        return None
    rows = nli([(k, sentence) for k in kb_sentences])
    support = max((float(r.get("entailment", 0.0)) for r in rows), default=0.0)
    ci = max(range(len(rows)), key=lambda i: float(rows[i].get("contradiction", 0.0)))
    conflict = float(rows[ci].get("contradiction", 0.0))
    conflict_rev = 0.0
    if conflict >= CONTRA_MIN and support < SUPPORT_EXEMPT:
        conflict_rev = float(nli([(sentence, kb_sentences[ci])])[0].get("contradiction", 0.0))
    return {"support": round(support, 4), "conflict": round(conflict, 4),
            "conflict_rev": round(conflict_rev, 4)}


def sentence_state(sentence: str, kb_sentences: list[str], locale: str,
                   forbidden_claims=None, nli: Optional[ScoresBatch] = None) -> dict[str, Any]:
    """Ink state for one sentence. Pure function of its inputs."""
    norm = _norm(sentence)
    reasons: list[str] = []
    rec: dict[str, Any] = {"text": sentence, "state": STATE_UNANCHORED, "anchor": None,
                           "overlap": 0.0, "forbidden": None, "nli": None, "reasons": reasons}
    flagged = asserts_forbidden_claim(sentence, forbidden_claims or [], locale)
    if flagged:
        rec["forbidden"] = flagged
    if any(norm == _norm(k) for k in kb_sentences):
        rec["anchor"] = next(i for i, k in enumerate(kb_sentences) if norm == _norm(k))
        rec["overlap"] = 1.0
        if flagged:   # a KB sentence can never be a forbidden claim; keep the gate anyway
            rec["state"] = STATE_LOW
            reasons.append("forbidden_claim")
        else:
            rec["state"] = STATE_VERIFIED
            reasons.append("verbatim_kb_sentence")
        return rec
    ai, ov = anchor_for(sentence, kb_sentences)
    rec["anchor"], rec["overlap"] = ai, round(ov, 4)
    if flagged:
        rec["state"] = STATE_LOW
        reasons.append("forbidden_claim")
        return rec
    if ai is None or ov < ANCHOR_MIN:
        rec["state"] = STATE_UNANCHORED
        reasons.append("no_kb_anchor")
        return rec
    if ov < THRESHOLD:
        rec["state"] = STATE_LOW
        reasons.append(f"overlap_below_{THRESHOLD}")
        return rec
    rec["state"] = STATE_VALIDATED
    reasons.append(f"overlap_ge_{THRESHOLD}")
    scores = _nli_scores(nli, kb_sentences, sentence)
    if scores is not None:
        rec["nli"] = scores
        bidir = (min(scores["conflict"], scores["conflict_rev"]) >= CONTRA_MIN
                 and scores["support"] < SUPPORT_EXEMPT)
        if bidir and locale in GATE_LOCALES:
            rec["state"] = STATE_DISAGREE
            reasons.append("nli_bidirectional_contradiction")
        elif bidir:
            reasons.append("nli_contradiction_unmeasured_locale")
        elif scores["support"] >= SUPPORT_EXEMPT:
            reasons.append("nli_supported")
    return rec


def provenance(kb_text: Optional[str], answer_text: str, locale: str = "en",
               forbidden_claims=None, nli: Optional[ScoresBatch] = None) -> dict[str, Any]:
    """The receipt. `tier` is score_v0's unchanged whole-answer verdict."""
    tier, overlap = classify(kb_text, answer_text)
    flagged_answer = asserts_forbidden_claim(answer_text, forbidden_claims or [], locale)
    if flagged_answer and kb_text is not None and tier == TIER_VALIDATED:
        tier = TIER_LOW_CONFIDENCE
    kb_sentences = [s for s in _sentences(kb_text or "") if len(s) >= MIN_SENTENCE_CHARS]
    sentences = []
    for s in _sentences(answer_text):
        if len(s) < MIN_SENTENCE_CHARS:
            continue
        if kb_text is None:
            rec = {"text": s, "state": STATE_UNANCHORED, "anchor": None, "overlap": 0.0,
                   "forbidden": asserts_forbidden_claim(s, forbidden_claims or [], locale),
                   "nli": None, "reasons": ["no_kb"]}
            if rec["forbidden"]:
                rec["state"] = STATE_LOW
                rec["reasons"].append("forbidden_claim")
        else:
            rec = sentence_state(s, kb_sentences, locale, forbidden_claims, nli)
        rec["index"] = len(sentences)
        sentences.append(rec)
    counts = {st: sum(1 for r in sentences if r["state"] == st) for st in STATES}
    return {
        "version": PROVENANCE_VERSION,
        "locale": locale,
        "tier": tier,
        "overlap": round(overlap, 4),
        "blocked_claim": flagged_answer,
        "answer_sha256": _sha(answer_text),
        "kb_sha256": _sha(kb_text) if kb_text is not None else None,
        "checks": {"lexical": True, "forbidden_claim": True,
                   "nli": nli is not None and kb_text is not None,
                   "nli_scope": locale in GATE_LOCALES},
        "counts": counts,
        "sentences": sentences,
    }


def _ink(state: str) -> str:
    return {STATE_VERIFIED: "██", STATE_VALIDATED: "▓▓", STATE_DISAGREE: "~~",
            STATE_LOW: "░░", STATE_UNANCHORED: "  "}[state]


def print_receipt(r: dict[str, Any]) -> None:
    print(f"\nProvenance Ink {r['version']} · locale={r['locale']} · tier={r['tier']} "
          f"(overlap {r['overlap']:.2f}) · nli={'on' if r['checks']['nli'] else 'off'}")
    print("  " + " · ".join(f"{k} {v}" for k, v in r["counts"].items()))
    for s in r["sentences"]:
        a = f"→ kb[{s['anchor']}]" if s["anchor"] is not None else "→ (no anchor)"
        print(f"  {_ink(s['state'])} {s['state']:<10} ovl {s['overlap']:.2f} {a:<14} {s['text'][:70]}")
    print()


def score_fixture(fixture_path: Path, kb_dir: Path, locale: str) -> int:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    kb = load_kb(kb_dir, locale)
    totals = {st: 0 for st in STATES}
    for item in data["items"]:
        answer = item["ai_explanations"].get(locale) or item["ai_explanations"]["en"]
        kb_text = kb.get(item.get("kb_concept_id") or "")
        r = provenance(kb_text, answer, locale, item.get("forbidden_claims", []))
        for st in STATES:
            totals[st] += r["counts"][st]
        print(f"{item['id']:<7} {r['tier']:<24} " +
              " ".join(f"{_ink(s['state'])}" for s in r["sentences"]))
    n = sum(totals.values())
    print(f"\n  {n} sentences · " + " · ".join(f"{k} {v}" for k, v in totals.items()))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fixture", nargs="?", type=Path)
    p.add_argument("--kb", default=None, help="KB text (with --answer) or KB directory (with a fixture)")
    p.add_argument("--answer", default=None)
    p.add_argument("--locale", default="en", choices=LOCALES)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    if a.answer is not None:
        r = provenance(a.kb, a.answer, a.locale)
        print(json.dumps(r, ensure_ascii=False, indent=1)) if a.json else print_receipt(r)
        return 0
    if a.fixture is None:
        p.print_usage()
        return 2
    kb_dir = Path(a.kb) if a.kb else a.fixture.resolve().parent.parent / "kb-examples"
    if not kb_dir.is_dir():
        print(f"KB directory not found: {kb_dir}", file=sys.stderr)
        return 1
    return score_fixture(a.fixture, kb_dir, a.locale)


if __name__ == "__main__":
    raise SystemExit(main())
