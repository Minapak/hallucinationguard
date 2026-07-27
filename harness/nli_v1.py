#!/usr/bin/env python3
"""HallucinationGuard — NLI verification backend (M2 harness layer, V1).

This is the dependency-bearing layer the zero-dependency V0 core deliberately
excludes (torch + transformers). It turns the lexical-vs-entailment gap into a
published number: the V0 overlap gate and this NLI checker score the same
fixtures side by side, and the disagreements are the benchmark's product.

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (~279M params,
~560MB download on first use) — a multilingual NLI cross-encoder fine-tuned on
XNLI + multilingual-NLI-2mil7 across 27 languages. Chosen over the stronger
English-only grounding checkers (MiniCheck, EMNLP 2024; AlignScore, ACL 2023)
because this benchmark's contract is 7 locales including ko/ja/zh-Hans, and a
weaker multilingual judge beats a stronger monolingual one that cannot read 3
of the 7 locales. Trade-offs, stated plainly: XNLI's language list does not
include Korean or Japanese — coverage there rides on mDeBERTa's CC100
pretraining and the 2mil7 fine-tune mix, so per-locale results must be
measured, not assumed (docs/NLI-GAP-2026-07-27.md publishes exactly that).

Granularity: sentence-by-sentence on BOTH sides (SummaC's design, Laban et al.
TACL 2022 — "fixes the granularity mismatch"), measured here to matter: with
whole-KB premises this model rated HG002's faster-than-light answer entailed
(0.88) and a correct bra-ket answer contradicted (0.72); with sentence pairs it
flags HG002's FTL sentences (contradiction 0.50-0.61; 1.00 against the KB's
no-signalling sentence) and clears 3 of 4 HG006 sentences. Per answer sentence:
  support  = max entailment over KB sentences
  conflict = max contradiction over KB sentences

Decision rule (documented, Goodhart-prone like every rule in this repo):
  sentence contradicted : conflict >= 0.50 and conflict > support
  nli_contradicted      : any sentence contradicted
  nli_supported         : no contradicted sentence and mean support >= 0.50
  nli_unverified        : everything else

Measured residual limits, not papered over: one notation-dense correct sentence
(HG006 en, bra/⟨ψ| symbols) still false-alarms at conflict 0.71 — notation
verbalization was tested and made it worse, so it ships as a documented model
limitation; Korean reliability is lower (XNLI's languages exclude ko/ja).

Usage:
  .venv/bin/python harness/nli_v1.py fixtures/samples/hg120_sample_20.json --locale en
  .venv/bin/python harness/nli_v1.py --demo-pair          # the headline sentences
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "reference"))
from score_v0 import (  # noqa: E402
    LOCALES, THRESHOLD, TIER_LOW_CONFIDENCE, TIER_VALIDATED, _sentences,
    classify, load_kb,
)

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

ENTAIL_MIN = 0.50
CONTRA_MIN = 0.50


@dataclass
class SentenceVerdict:
    sentence: str
    support: float   # max entailment over KB sentences
    conflict: float  # max contradiction over KB sentences

    @property
    def contradicted(self) -> bool:
        return self.conflict >= CONTRA_MIN and self.conflict > self.support


@dataclass
class NLIVerdict:
    sentences: list

    @property
    def mean_support(self) -> float:
        return sum(s.support for s in self.sentences) / len(self.sentences) if self.sentences else 0.0

    @property
    def max_conflict(self) -> float:
        return max((s.conflict for s in self.sentences), default=0.0)

    @property
    def worst(self) -> "SentenceVerdict | None":
        return max(self.sentences, key=lambda s: s.conflict, default=None)

    @property
    def label(self) -> str:
        if any(s.contradicted for s in self.sentences):
            return "nli_contradicted"
        if self.sentences and self.mean_support >= ENTAIL_MIN:
            return "nli_supported"
        return "nli_unverified"


class NLIChecker:
    """Multilingual NLI cross-encoder wrapper. Loads lazily; ~560MB on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model.to(device).eval()
        # label order per model config (entailment/neutral/contradiction for this model)
        self.id2label = {i: lbl.lower() for i, lbl in self.model.config.id2label.items()}

    def scores(self, premise: str, hypothesis: str) -> dict:
        return self.scores_batch([(premise, hypothesis)])[0]

    def scores_batch(self, pairs: list) -> list:
        """One forward pass over a batch of (premise, hypothesis) pairs."""
        with self.torch.no_grad():
            enc = self.tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                                 truncation=True, max_length=512, padding=True,
                                 return_tensors="pt").to(self.device)
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        return [{self.id2label[i]: p for i, p in enumerate(row)} for row in probs]

    def check(self, kb_text: str, answer_text: str) -> NLIVerdict:
        """SummaC-style sentence-by-sentence grounding of answer against KB."""
        kb_sents = [s for s in _sentences(kb_text) if len(s) >= 8]
        ans_sents = [s for s in _sentences(answer_text) if len(s) >= 8]
        verdicts = []
        for asent in ans_sents:
            rows = self.scores_batch([(k, asent) for k in kb_sents]) if kb_sents else []
            verdicts.append(SentenceVerdict(
                sentence=asent,
                support=max((r["entailment"] for r in rows), default=0.0),
                conflict=max((r["contradiction"] for r in rows), default=0.0)))
        return NLIVerdict(sentences=verdicts)


def demo_pair(checker: NLIChecker) -> int:
    """The benchmark's headline sentences, scored by NLI instead of overlap."""
    a = "Measurement collapses the state."
    b = "Measurement never collapses the state."
    print("\nNLI on the metric-blindness pair (premise -> hypothesis):\n")
    for p, h in ((a, b), (a, a)):
        s = checker.scores(p, h)
        print(f'  "{p}" -> "{h}"')
        print(f"     entail {s['entailment']:.3f} · neutral {s['neutral']:.3f} "
              f"· contradiction {s['contradiction']:.3f}\n")
    return 0


def score_fixture(checker: NLIChecker, fixture_path: Path, kb_dir: Path, locale: str) -> int:
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    items = data["items"]
    kb = load_kb(kb_dir, locale)

    print(f"\nHallucinationGuard NLI V1 — {fixture_path.name} · locale={locale} "
          f"· model={DEFAULT_MODEL.split('/')[-1]}\n")
    widths = [6, 24, 5, 8, 9, 16]
    header = ("id", "V0 tier (lexical)", "ovl", "support", "conflict", "NLI label")
    rows = [header]
    lex_gap, results = [], []
    for item in items:
        answer = item["ai_explanations"].get(locale) or item["ai_explanations"]["en"]
        kb_text = kb.get(item.get("kb_concept_id") or "")
        tier, score = classify(kb_text, answer)
        if kb_text is None:
            rows.append((item["id"], tier, f"{score:.2f}", "-", "-", "no KB"))
            continue
        v = checker.check(kb_text, answer)
        nli_pass = v.label == "nli_supported"
        lex_pass = tier == TIER_VALIDATED
        if nli_pass != lex_pass:
            lex_gap.append((item["id"], tier, v.label))
        worst = v.worst
        results.append({"id": item["id"], "v0_tier": tier, "overlap": round(score, 4),
                        "mean_support": round(v.mean_support, 4),
                        "max_conflict": round(v.max_conflict, 4), "nli_label": v.label,
                        "worst_sentence": (worst.sentence[:120] if worst else ""),
                        "sentences": [{"support": round(s.support, 4),
                                       "conflict": round(s.conflict, 4),
                                       "contradicted": s.contradicted,
                                       "text": s.sentence[:120]} for s in v.sentences]})
        rows.append((item["id"], tier, f"{score:.2f}", f"{v.mean_support:.2f}",
                     f"{v.max_conflict:.2f}", v.label))
    for row in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    print(f"\n  lexical-vs-NLI disagreements (KB-backed items): {len(lex_gap)}")
    for hid, tier, label in lex_gap:
        print(f"    {hid}: V0 {tier} vs {label}")
    print()
    out = {"fixture": fixture_path.name, "locale": locale, "model": DEFAULT_MODEL,
           "thresholds": {"entail_min": ENTAIL_MIN, "contra_min": CONTRA_MIN},
           "results": results}
    out_path = REPO / "harness" / "runs" / f"nli_{fixture_path.stem}_{locale}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  raw scores saved: {out_path.relative_to(REPO)}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, nargs="?", help="benchmark fixture JSON")
    parser.add_argument("--locale", default="en", choices=LOCALES)
    parser.add_argument("--kb", type=Path, default=None)
    parser.add_argument("--demo-pair", action="store_true",
                        help="score the headline collapses/never-collapses pair and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    checker = NLIChecker(args.model)
    if args.demo_pair:
        return demo_pair(checker)
    if args.fixture is None:
        parser.error("fixture path required unless --demo-pair")
    kb_dir = args.kb or args.fixture.resolve().parent.parent / "kb-examples"
    if not kb_dir.is_dir():
        print(f"KB directory not found: {kb_dir}", file=sys.stderr)
        return 1
    return score_fixture(checker, args.fixture, kb_dir, args.locale)


if __name__ == "__main__":
    raise SystemExit(main())
