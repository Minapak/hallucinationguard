# Architecture

This repository is a **benchmark artifact + reference implementation**, not a service. There is no
deployment surface, no runtime infrastructure, and no ops/monitoring component here — the production
guard runs inside the SwiftQuantum app's commercial pipeline, from which this public artifact is
being extracted (see [README — Provenance](README.md#provenance--transparency)). Everything in this
repo runs offline with zero dependencies on CPython 3.9+.

```
hallucinationguard/
├── fixtures/
│   ├── samples/hg120_sample_20.json   # 20 of 120 benchmark items (schema hg_benchmark_item_v3.1)
│   │                                  #   7-locale prompts + recorded model answers (frozen artifacts),
│   │                                  #   7-locale forbidden_claims, ground-truth tier per item
│   ├── samples/hgx_provenance_6.json  # 6 synthetic adversarial extension items (schema v3.2, HGX01-06):
│   │                                  #   uncited near-verbatim / hallucinated citation / depth inversion
│   │                                  #   + control — NOT part of HG120, not in the 1,680-field count
│   ├── samples/hgm_misconceptions_3.json # 3 synthetic misconception items (schema v3.1, HGM01-03):
│   │                                  #   spinning coin / "simply both 0 and 1" / hidden definite value
│   │                                  #   — authored wrong answers, NOT part of HG120
│   └── kb-examples/*.json             # 12 verified KB concepts: certainty grade, DOI sourceRef,
│                                      #   7-locale translations, verifier record
├── reference/score_v0.py              # the production V0 baseline scorer (single file, stdlib only)
├── reference/dimensions_v1.py         # separate-dimension report: correctness · attribution ·
│                                      #   citation integrity · exposition depth (stdlib only)
├── harness/                           # M2 dependency-bearing layer (torch+transformers, OPTIONAL):
│   ├── nli_v1.py                      #   multilingual NLI checker beside the V0 baseline + the V2
│   ├── requirements.txt               #   gate (nli_gate(): bidirectional confirmation, low-confidence
│   └── runs/*.json                    #   -only, GATE_LOCALES {en,es,fr} — 0 measured false blocks;
│                                      #   docs/NLI-GAP-2026-07-27.md §6); core stays zero-dep
├── demo/metric_blindness_demo.py      # 5-second reproduction of the core finding
├── tests/
│   ├── test_fixtures.py               # schema + certainty gates + parity/orthography gates (both fixtures)
│   ├── test_scorer.py                 # unit tests + pinned per-locale regression matrix (HG120 sample)
│   └── test_dimensions.py             # citation/depth unit tests + per-locale flag & tier pins (HGX)
├── docs/                              # SCHEMA.md · TEST_GUIDE.md · AUDIT-2026-07-19.md
└── .github/workflows/ci.yml           # SHA-pinned, least-privilege, Python 3.9+3.14 matrix
```

## Data flow of the scorer

```
fixture item ──► answer (active locale, en fallback + warning)
                     │
KB dir ──► load_kb ──┤  per-field locale fallback, stderr warning on cross-language comparison
                     ▼
              tokenize_v0 (V0.1)        whitespace split, case-folded; unspaced CJK runs → char bigrams
                     │
              classify()                verbatim string equality → verifiedKnowledgeBase
                     │                  overlap |KB ∩ answer|/|KB| ≥ 0.30 → aiValidatedAgainstKB
                     │                  else aiWithLowConfidence; no KB entry → aiNoKBMatch
                     ▼
       asserts_forbidden_claim()        locale claims, token recall ≥ 0.60,
                     │                  minus mention-framing and negation-cue suppression
                     ▼
              gate precedence           flagged + would-validate → demoted to aiWithLowConfidence
```

## Dimension layer (`reference/dimensions_v1.py`, v0.3.0)

Orthogonal to the correctness flow above, `computed_flags()` reports three provenance/exposition
flags side by side (HELM-style separate dimensions, never blended into one score):

```
answer ──► attribution        overlap ≥ 0.85 on a non-verbatim-served answer → attributionRequired
       ──► citation integrity DOI/arXiv/FIPS identifiers vs the KB entry's reference corpus
       │                      (sourceRef + citationReferences + certaintyNote + en text) → hallucinatedCitation
       └─► exposition depth   annotated hard_step_concepts: coverage ≥ 0.5 but < 20% of chars
                              in hard-step sentences → depthInverted (crude proxy; M2 judge replaces it)
```

Flags never change the correctness tier. `--check-flags` compares computed flags against each
item's `expected_flags` and exits non-zero on mismatch — CI runs it for all 7 locales against
the HGX extension fixture. The M2 NLI/judge backend slots in behind the same `computed_flags()`
signature.

Design invariants (enforced by `tests/`, documented in [docs/SCHEMA.md](docs/SCHEMA.md) and
[docs/AUDIT-2026-07-19.md](docs/AUDIT-2026-07-19.md)):

1. **Recorded answers are frozen.** `ai_explanations` are real model outputs, errors included;
   correcting them would falsify the benchmark. Verified content lives only in `fixtures/kb-examples/`.
2. **The V0 scorer is deliberately naive, and the naivety is pinned.** Its documented failures
   (semantic inversion, Korean agglutination, punctuation attachment) are the benchmark's subject.
   The per-locale regression matrix in `tests/test_scorer.py` pins agreement, miss sets, and
   blocked sets so any drift is a deliberate, reviewed change.
3. **Certainty is never auto-promoted**, translations carry parity/orthography CI gates, and
   time-sensitive claims must be date-qualified — see the schema.
4. **Zero dependencies is a contract.** Anything requiring a model or a segmenter (MeCab, NLI
   checkers) belongs to the M2 harness, reported *alongside* — not replacing — the V0 baseline.
5. **Dimensions are separate axes.** Provenance/exposition flags (`attributionRequired`,
   `hallucinatedCitation`, `depthInverted`) never alter the correctness tier, and the HGX
   extension items that exercise them are synthetic, tuned-to-fire, and kept outside HG120 —
   Goodhart's law applies to these detectors as to every metric here, and the docs say so.
