#!/usr/bin/env python3
"""HallucinationGuard's core finding, on your terminal in ~5 seconds.

Two sentences. Opposite physics. Watch what the usual metrics say —
then watch what HallucinationGuard says.

  python3 demo/metric_blindness_demo.py

Zero dependencies (stdlib only). Colors need an ANSI terminal.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
from score_v0 import (  # noqa: E402
    THRESHOLD, asserts_forbidden_claim, classify, load_kb, one_sided_overlap, tokenize_v0,
)

BOLD, DIM, RED, GREEN, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m")

CORRECT = "Measurement collapses the state."
WRONG = "Measurement never collapses the state."


def bow_cosine(a: str, b: str) -> float:
    """Bag-of-words cosine similarity — the metric family everyone uses first."""
    ca, cb = Counter(a.casefold().split()), Counter(b.casefold().split())
    dot = sum(ca[t] * cb[t] for t in ca)
    na, nb = math.sqrt(sum(v * v for v in ca.values())), math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def jaccard(a: str, b: str) -> float:
    sa, sb = tokenize_v0(a), tokenize_v0(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def bar(score: float, width: int = 30) -> str:
    filled = round(score * width)
    color = RED if score >= 0.7 else YELLOW if score >= 0.4 else GREEN
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET} {score:.3f}"


def pause(seconds: float = 0.9) -> None:
    if sys.stdout.isatty():
        time.sleep(seconds)


def main() -> int:
    print(f"\n{BOLD}Two sentences. Opposite physics.{RESET}\n")
    print(f'  A: {GREEN}"{CORRECT}"{RESET}')
    print(f'  B: {RED}"{WRONG}"{RESET}   {DIM}← wrong physics, right vocabulary{RESET}\n')
    pause()

    print(f"{BOLD}What the usual similarity metrics say about A vs B:{RESET}\n")
    print(f"  bag-of-words cosine   {bar(bow_cosine(CORRECT, WRONG))}")
    print(f"  token Jaccard         {bar(jaccard(CORRECT, WRONG))}")
    print(f"  char SequenceMatcher  {bar(SequenceMatcher(None, CORRECT, WRONG).ratio())}")
    print(f"\n  {YELLOW}An answer about the right topic, with the wrong physics, still scores high.{RESET}")
    print(f"  {DIM}(Word-embedding mean-pooling shows the same compositional blindness.){RESET}\n")
    pause(1.4)

    print(f"{BOLD}What HallucinationGuard says:{RESET}\n")
    repo = Path(__file__).resolve().parent.parent
    kb = load_kb(repo / "fixtures" / "kb-examples", "en")
    fixture = json.loads((repo / "fixtures" / "samples" / "hg120_sample_20.json").read_text(encoding="utf-8"))

    # Real benchmark items: a faithful answer, and two documented misconceptions.
    demo_ids = ["HG001", "HG002", "HG023"]
    items = {it["id"]: it for it in fixture["items"]}
    for hid in demo_ids:
        item = items[hid]
        answer = item["ai_explanations"]["en"]
        kb_text = kb.get(item.get("kb_concept_id") or "")
        tier, score = classify(kb_text, answer)
        flagged = asserts_forbidden_claim(answer, item.get("forbidden_claims", []))
        print(f'  {BOLD}{hid}{RESET} {DIM}[{item["category"]}]{RESET} "{item["prompts"]["en"]}"')
        if flagged:
            print(f"    {RED}✗ BLOCKED{RESET} — asserts forbidden claim: {RED}“{flagged}”{RESET}")
            print(f"    {DIM}→ guard serves the verified KB answer instead"
                  f" (source: {kb_source(repo, item.get('kb_concept_id'))}){RESET}")
        elif tier == "aiValidatedAgainstKB":
            print(f"    {GREEN}✓ VALIDATED{RESET} — overlap {score:.2f} ≥ {THRESHOLD}"
                  f" against verified KB entry “{item.get('kb_concept_id')}”")
        elif tier == "aiWithLowConfidence":
            print(f"    {YELLOW}⚠ LOW CONFIDENCE{RESET} — overlap {score:.2f} < {THRESHOLD};"
                  f" verified KB answer offered instead")
        else:
            print(f"    {CYAN}○ NO KB MATCH{RESET} — passed through with explicit caveat")
        pause(0.7)
        print()

    print(f"{BOLD}Same vocabulary is not same physics.{RESET}")
    print(f"{DIM}Benchmark: 120 items · 1,680 verified fields · 7 locales · Apache-2.0 (sample of 20 in this repo){RESET}\n")
    return 0


def kb_source(repo: Path, concept_id: str | None) -> str:
    if not concept_id:
        return "n/a"
    path = repo / "fixtures" / "kb-examples" / f"{concept_id}.json"
    if not path.exists():
        return "n/a"
    return json.loads(path.read_text(encoding="utf-8")).get("sourceRef", "n/a")


if __name__ == "__main__":
    raise SystemExit(main())
