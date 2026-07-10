# HallucinationGuard

**An open benchmark for measuring what LLMs get wrong about quantum computing.**

1,680 verified fields. 7 locales. Every claim traced upstream to primary literature — original papers with DOIs, NIST standards, canonical texts (Nielsen & Chuang, Preskill's lecture notes, Mermin). Running in production today, gating every AI-generated explanation shown to real learners.

When a language model explains quantum mechanics to a learner — who is checking the physics?

## The problem

The metrics everyone uses are part of the problem. Bag-of-words metrics cannot separate topic overlap from semantic agreement — an answer about the right topic with the wrong physics still scores high. Word-embedding mean-pooling inherits the same compositional blindness:

> "Measurement collapses the state."
> "Measurement **never** collapses the state."

Nearly identical, to an embedding. Try it yourself:

```bash
python3 demo/metric_blindness_demo.py
```

HallucinationGuard is the benchmark built around that finding. It is not a research prototype: it runs in production today, gating every AI-generated explanation shown to users of a live quantum-education app.

## What is in this repository (today)

This is the public skeleton — the project is being extracted from a commercial production pipeline into a standalone public artifact. Already here:

| Path | Contents |
|---|---|
| [`fixtures/samples/hg120_sample_20.json`](fixtures/samples/hg120_sample_20.json) | 20 of 120 benchmark items — all 16 categories, all 3 ground-truth classes, 7 locales per item (280 multilingual fields in the sample) |
| [`fixtures/kb-examples/`](fixtures/kb-examples/) | 12 verified knowledge-base concepts — each with certainty grade, primary-source citation (DOIs), and verifier record. Korean terminology is aligned with the Korean Physical Society (KPS) 2020 glossary; time-sensitive status claims are explicitly date-qualified |
| [`reference/score_v0.py`](reference/score_v0.py) | The production baseline scorer (V0 tokenizer, one-sided overlap, threshold 0.30, three-tier classification) — zero dependencies, stdlib only |
| [`demo/metric_blindness_demo.py`](demo/metric_blindness_demo.py) | Reproduces the paper's core finding on your terminal in ~5 seconds |
| [`docs/SCHEMA.md`](docs/SCHEMA.md) | One-page schema for benchmark items, KB entries, and the certainty taxonomy |
| [`ROADMAP.md`](ROADMAP.md) | Four milestones to full release |

The full fixture set is 120 items × 14 multilingual fields (7 prompt locales + 7 model-answer locales) = **1,680 verified fields**. Full migration is Milestone 1 (see roadmap).

## Quickstart

No dependencies. Python 3.9+.

```bash
git clone https://github.com/Minapak/hallucinationguard
cd hallucinationguard

# reproduce the metric-blindness finding
python3 demo/metric_blindness_demo.py

# score the sample benchmark with the production baseline
python3 reference/score_v0.py fixtures/samples/hg120_sample_20.json --locale en
```

## How the guard classifies

Every model answer lands in one of three tiers (paper §III-C):

- 🟢 `verifiedKnowledgeBase` — served directly from the expert-verified KB
- 🔵 `aiValidatedAgainstKB` — generated answer, one-sided token overlap with the KB entry ≥ 0.30
- 🟡 `aiWithLowConfidence` — KB entry exists, overlap below threshold → the answer is flagged and the verified KB answer is offered instead

Answers whose topic has no KB entry are `aiNoKBMatch` — passed through with an explicit caveat. In addition, each benchmark item carries `forbidden_claims`: known misconception patterns ("entanglement transmits information faster than light", "consciousness collapses the wave function") that must never be asserted.

**Production baseline accuracy: 0.9083 (109/120) on the full HG120 set.** Stated honestly: the V0 lexical baseline has documented semantic-inversion boundary cases — two adversarial items in the full set defeat pure overlap scoring. That limitation is not a footnote; it is the paper's central finding, and the reason this benchmark needs to exist.

## Certainty taxonomy

Every KB claim carries one of four grades, enforced by CI gates — never auto-promoted:

| Grade | Meaning | Gate |
|---|---|---|
| `established` | Verified against primary sources | must carry a meaningful `sourceRef` (DOI / citation with year) |
| `verify` | Error suspected or unverified — re-check the primary source | — |
| `open` | No academic consensus (e.g. interpretations of measurement) | — |
| `timebound` | True now, changes over time (hardware numbers, tooling landscapes) | must carry an explicit `asOfDate` (YYYY-MM-DD) instead of pretending to be timeless |

`null` means needs-review — visible, never hidden.

## Provenance & transparency

HallucinationGuard was born inside a commercial product — a paid quantum-education app where it gates AI tutor output daily. This repository is the extraction of that benchmark into a public good under Apache-2.0, which means direct competitors of the author's own app can use it freely. That's the point: a hallucination benchmark only works if the people being measured trust it wasn't built to favor its author.

- Product where the benchmark runs today: https://swiftquantum.tech (App Store: "SwiftQuantum")
- Author: Eunmin Park — IEEE member, solo developer, Seoul
- Content freshness: last full verification pass 2026-07-10 — KO terms aligned with the KPS 2020 physics glossary (e.g. decoherence = 결깨짐, with common-usage 결어긋남 noted); 2024–26 QEC milestones (Google Willow below-threshold, IBM gross qLDPC) and PQC standardization status (HQC, FIPS 206, EO 14412) added as explicitly year/date-qualified sentences; institution-confirmed misconception corrections (IBM/Caltech) applied, including replacing the "tries every answer at once" framing with amplitude interference

## License

[Apache-2.0](LICENSE). Fixture content, schema, and reference code alike.

---

*Never call a classical algorithm "quantum." Never claim more accuracy than you can prove.*
