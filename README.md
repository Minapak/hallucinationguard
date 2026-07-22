# HallucinationGuard

**An open benchmark for measuring what LLMs get wrong about quantum computing.**

1,680 verified fields. 7 locales. Every claim traced upstream to primary literature — original papers with DOIs, NIST standards, canonical texts (Nielsen & Chuang, Preskill's lecture notes, Mermin). Running in production today, gating every AI-generated explanation shown to real learners.

When a language model explains quantum mechanics to a learner — who is checking the physics?

## The problem

The metrics everyone uses are part of the problem. Bag-of-words metrics cannot separate topic overlap from semantic agreement — an answer about the right topic with the wrong physics still scores high. Word-embedding mean-pooling inherits the same compositional blindness:

> "Measurement collapses the state."
> "Measurement **never** collapses the state."

Nearly identical, to an embedding. This is a documented failure class, not an anecdote: learned similarity metrics (BERTScore, BLEURT) rate a sentence and its negation as highly similar (Anschütz et al., INLG 2023), masked LMs complete negated and non-negated cloze prompts the same way (Ettinger, TACL 2020; Kassner & Schütze, ACL 2020), and lexical-overlap metrics are the weakest family in factual-consistency meta-evaluation (Honovich et al., TRUE, NAACL 2022). Try it yourself:

```bash
python3 demo/metric_blindness_demo.py
```

HallucinationGuard is the benchmark built around that finding. It is not a research prototype: it runs in production today, gating every AI-generated explanation shown to users of a live quantum-education app.

## What is in this repository (today)

This is the public skeleton — the project is being extracted from a commercial production pipeline into a standalone public artifact. Already here:

| Path | Contents |
|---|---|
| [`fixtures/samples/hg120_sample_20.json`](fixtures/samples/hg120_sample_20.json) | 20 of 120 benchmark items — all 16 categories, 3 of 4 ground-truth classes, 7 locales per item (280 multilingual fields in the sample), forbidden-claim misconception patterns localized to all 7 locales (schema v3.1) |
| [`fixtures/kb-examples/`](fixtures/kb-examples/) | 12 verified knowledge-base concepts — each with certainty grade, primary-source citation (DOIs), and verifier record. Korean terminology is aligned with the Korean Physical Society (KPS) 2020 glossary; time-sensitive status claims are explicitly date-qualified |
| [`reference/score_v0.py`](reference/score_v0.py) | The production baseline scorer (tokenizer V0.1 with CJK character bigrams, one-sided overlap, threshold 0.30, four-tier classification, locale-aware forbidden-claim gate) — zero dependencies, stdlib only |
| [`demo/metric_blindness_demo.py`](demo/metric_blindness_demo.py) | Reproduces the benchmark's core finding on your terminal in ~5 seconds |
| [`tests/`](tests/) | Fixture schema + certainty gates, scorer unit tests, and a pinned per-locale regression matrix — tier or gate drift fails CI instead of passing silently (how to run: [docs/TEST_GUIDE.md](docs/TEST_GUIDE.md)) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Repo layout, scorer data flow, and the four design invariants |
| [`docs/SCHEMA.md`](docs/SCHEMA.md) | One-page schema for benchmark items, KB entries, and the certainty taxonomy |
| [`docs/AUDIT-2026-07-19.md`](docs/AUDIT-2026-07-19.md) | Full audit trail of the 2026-07-19 hardening pass: 27 confirmed findings, fixes, and the primary sources behind each decision |
| [`CHANGELOG.md`](CHANGELOG.md) | Versioned change history (versions track `CITATION.cff`) |
| [`ROADMAP.md`](ROADMAP.md) | Four milestones to full release |

The full fixture set is 120 items × 14 multilingual fields (7 prompt locales + 7 model-answer locales) = **1,680 verified fields**. Full migration is Milestone 1 (see roadmap).

## Quickstart

No dependencies. Python 3.9+ (CI tests 3.9 and 3.14; note Python 3.9 reached end-of-life 2025-10-31 — the floor moves to 3.10 at the M2 packaging milestone).

```bash
git clone https://github.com/Minapak/hallucinationguard
cd hallucinationguard

# reproduce the metric-blindness finding
python3 demo/metric_blindness_demo.py

# score the sample benchmark with the production baseline
python3 reference/score_v0.py fixtures/samples/hg120_sample_20.json --locale en
```

## How the guard classifies

Every model answer lands in one of four outcomes (see [docs/SCHEMA.md](docs/SCHEMA.md)):

- 🟢 `verifiedKnowledgeBase` — served verbatim from the expert-verified KB (detected by string equality, not overlap)
- 🔵 `aiValidatedAgainstKB` — generated answer, one-sided token overlap with the KB entry ≥ 0.30
- 🟡 `aiWithLowConfidence` — KB entry exists, overlap below threshold → the answer is flagged and the verified KB answer is offered instead
- ⚪ `aiNoKBMatch` — no KB entry for the topic → passed through with an explicit caveat

In addition, each benchmark item carries `forbidden_claims`: known misconception patterns ("entanglement transmits information faster than light", "consciousness collapses the wave function") in **all 7 locales**. An answer that asserts one is never validated — it is demoted and the verified KB answer is served instead. The gate is direction-aware at zero dependency cost: NegEx-style negation cues (Chapman et al. 2001) keep a correct *denial* of a misconception from being flagged, and denial-framing cues ("the common overstatement that …") keep a *mention* inside a warning from counting as an assertion. Both are small curated heuristics with documented failure modes — see the limitations below.

### Baseline results, stated honestly

| Scope | Result |
|---|---|
| Full HG120 set (production, en) | **0.9083** (109/120) — 11 misses, of which 2 are adversarial semantic-inversion items constructed to defeat overlap scoring |
| Public 20-item sample — en, es, fr | 20/20 |
| Public 20-item sample — de | 19/20 (punctuation-attached tokens cost HG005) |
| Public 20-item sample — ko, ja, zh-Hans | 18/20 (ko: agglutinative eojeol mismatch; ja/zh: bigram-granularity boundary cases on HG004/HG005) |

Read the sample numbers with care, in both directions. None of the 11 full-set misses are in the public sample, so the sample alone cannot reproduce the 0.9083 figure or the adversarial failures — at least one genuinely failing item ships with the full set (Milestone 1). Conversely, the ja/zh numbers were *higher* before the 2026-07-19 audit (19/20) — and meaningless: whitespace tokenization had collapsed CJK text to ~1 pseudo-token, so those "agreements" rested on matching `=` and `+` symbols while the misconception gate silently never fired outside English. Honest 18/20 beats vacuous 19/20. The per-locale matrix is pinned in [`tests/test_scorer.py`](tests/test_scorer.py); any drift fails CI.

### Known limitations of the V0 baseline (and the upgrade path)

The V0 scorer is deliberately simple — its documented failures are the benchmark's central finding, not a bug list to hide:

- **Lexical overlap is the weakest metric family** for factual consistency (TRUE benchmark, NAACL 2022). It cannot see semantic inversion; two adversarial HG120 items defeat it by design.
- **The forbidden-claim gate is lexical.** Paraphrases under the 0.60 token-recall threshold escape it; a negation or framing cue anywhere in the decisive sentence suppresses it even when the cue belongs to a different clause. Rule-based negation detection is a floor, not a solution — NegEx itself reports 77.8% sensitivity / 84.5% PPV on clinical text.
- **Korean is scored on whole eojeol tokens** (no morphological analysis), which costs two sample agreements; MeCab-ko-style segmentation is the standard remedy but breaks the zero-dependency contract (Park et al., AACL 2020).
- **Upgrade path, with trade-offs**: small NLI-based checkers reach GPT-4-level fact-checking at ~1/400 the cost (MiniCheck, EMNLP 2024; AlignScore, ACL 2023) but add a model dependency and GPU-friendly latency budgets; sampling-based semantic entropy (Farquhar et al., Nature 2024) needs no labels but multiplies inference cost several-fold. The planned `pip` harness (Milestone 2) will report the V0 lexical baseline alongside an NLI-based scorer so the gap itself becomes a published number.

## Certainty taxonomy

Every KB claim carries one of four grades, enforced by CI gates — never auto-promoted:

| Grade | Meaning | Gate |
|---|---|---|
| `established` | Verified against primary sources | must carry a meaningful `sourceRef` (DOI / citation with year) |
| `verify` | Error suspected or unverified — re-check the primary source | — |
| `open` | No academic consensus (e.g. interpretations of measurement) | — |
| `timebound` | True now, changes over time (hardware numbers, tooling landscapes) | must carry an explicit `asOfDate` (YYYY-MM-DD) instead of pretending to be timeless |

`null` means needs-review — visible, never hidden. CI additionally enforces translation-parity floors and es/fr orthography on every KB entry (see [docs/SCHEMA.md](docs/SCHEMA.md)).

## Provenance & transparency

HallucinationGuard was born inside a commercial product — a paid quantum-education app where it gates AI tutor output daily. This repository is the extraction of that benchmark into a public good under Apache-2.0, which means direct competitors of the author's own app can use it freely. That's the point: a hallucination benchmark only works if the people being measured trust it wasn't built to favor its author.

- Product where the benchmark runs today: https://swiftquantum.tech (App Store: "SwiftQuantum")
- Author: Eunmin Park — IEEE member, solo developer, Seoul
- Content freshness: last full verification pass 2026-07-10 — KO terms aligned with the KPS 2020 physics glossary (e.g. decoherence = 결깨짐, with common-usage 결어긋남 noted); 2024–26 QEC milestones (Google Willow below-threshold, IBM gross qLDPC) and PQC standardization status (HQC, FIPS 206, EO 14412) added as explicitly year/date-qualified sentences; institution-confirmed misconception corrections (IBM/Caltech) applied, including replacing the "tries every answer at once" framing with amplitude interference
- Hardening pass 2026-07-19: full audit (27 confirmed findings, all fixed) — CJK-aware tokenizer, localized misconception gate, four-tier classifier, translation backfill with parity gates, pinned regression matrix, SHA-pinned CI. Method, evidence, and primary sources: [docs/AUDIT-2026-07-19.md](docs/AUDIT-2026-07-19.md). Backfilled translations are faithful renderings of the verified EN text and remain pending native-speaker review (Milestone 3)

## Related work

HallucinationGuard is designed to complement, not duplicate, existing efforts:

- **Quantum-Audit** (arXiv:2602.10092, 2026) — a 2,700-question benchmark measuring whether models *know* quantum computing, with a Spanish/French subset. Measures model capability; HallucinationGuard verifies whether a *given generated explanation* is fabricated and gates it in production.
- **PhysicBench** (Li et al., 2025) — quantum-physics QA benchmarking inside RAG pipelines. Pipeline evaluation, not an installable output-verification harness.
- **SCIFACTCHECK** (arXiv:2606.21359, 2026) — general-science hallucination across five domains (unverifiability / overclaim / attribution). Not quantum-specific, not multilingual.
- **Med-HALT** (Pal et al., 2023) — precedent for a domain-specific hallucination benchmark (medicine).

**What is distinct here:** an installable (`pip install hallucinationguard-bench`), Apache-2.0 verification tool for quantum-education LLM output — 7 languages (EN, KO, JA, ZH-Hans, DE, FR, ES), every claim traced to primary literature, results published on Metriq. Existing work asks *"how much does this model know?"*; this asks *"is this particular sentence something a learner can trust?"*

## License

[Apache-2.0](LICENSE). Fixture content, schema, and reference code alike.

---

*Never call a classical algorithm "quantum." Never claim more accuracy than you can prove.*
