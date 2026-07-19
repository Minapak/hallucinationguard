# Changelog

All notable changes to this repository. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions track `CITATION.cff`.

## [0.2.0] — 2026-07-19 · Hardening pass

Full audit → adversarial re-verification → fix of all 27 confirmed findings (10 findings refuted and left unchanged, on record). Evidence and primary sources for every decision: [docs/AUDIT-2026-07-19.md](docs/AUDIT-2026-07-19.md).

### Fixed — scorer (`reference/score_v0.py`)
- **Tokenizer V0.1**: unspaced CJK runs (Han/Hiragana/Katakana) become character bigrams; whitespace splitting had collapsed ja/zh-Hans text to ~1 pseudo-token, making every ja/zh score rest on math-symbol matches (`=`, `+`). Korean deliberately keeps whitespace eojeol tokens (documented V0 naivety). Precedent: chrF (WMT 2015), WMT17 character-level Chinese scoring, SacreBLEU `zh` tokenizer.
- **Forbidden-claim gate localized** (schema v3.1): claims now carry all 7 locales; the en-only gate had been silently inert for 6 of 7 locales. Blocked set is now identical across locales: {HG002, HG023, HG031, HG039, HG051, HG073, HG115}.
- **Gate direction-awareness**: NegEx-style negation cues (correct denials no longer flagged), pseudo-negation handling (echoed negative claims still fire), denial-framing cues (mentions inside warnings — "the common overstatement that…" — no longer fire). HG002's paraphrase false negative closed with an added en claim variant.
- **Gate enforcement**: flagged answers are demoted from `aiValidatedAgainstKB` to `aiWithLowConfidence` (previously display-only, contradicting the schema).
- **Four-tier classification**: `verifiedKnowledgeBase` was unreachable; now detected by normalized string equality (not overlap 1.0, which superset answers also reach).
- Robustness: `--kb` default resolves correctly for bare filenames; clean stderr diagnostics + exit 1 for missing/malformed fixtures; empty-items guard; per-field KB locale fallback with stderr warnings on any cross-language comparison; small-KB-denominator warning; 7-column table printed with 7 widths.

### Fixed — fixtures
- Backfilled stub translations to parity in `grover_algorithm`, `phase_s`, `entanglement`, `ml_kem` (all six non-EN locales) and `quantum_error_correction` (fr repair — including the "intrigues"→"états intriqués" mistranslation — plus ja/zh expansion). Stub KBs had let a hallucinated Grover/NP answer validate on stopwords in es/fr. Translations are faithful renderings of the verified EN text, noted in `certaintyNote`, pending native-speaker review (M3); `verifiedBy` untouched.
- Restored full diacritics in all fr/es blocks that lacked them.
- `expected_keywords` of HG045/HG105 corrected to forms actually present in the en answers.
- Recorded `ai_explanations` declared frozen benchmark artifacts (never "corrected").

### Added — tests & CI
- `tests/test_scorer.py`: tokenizer/overlap/classifier/gate unit tests, CLI error-path tests, and a pinned per-locale regression matrix (agreement en/es/fr 20/20, de 19/20, ko/ja/zh-Hans 18/20; identical blocked sets) — drift now fails CI instead of passing silently.
- `tests/test_fixtures.py` extended: metadata bounds (preskill_part 1–24, ai_confidence 0–1), kb_category consistency, expected_keywords substring presence, 7-locale forbidden_claims shape, sample-distribution/category counts, translation-parity floors (es/fr/de ≥ 50%, ko ≥ 40%, ja ≥ 25%, zh-Hans ≥ 18% of en chars), es/fr diacritics gate.
- CI: actions pinned to full commit SHAs (checkout v7.0.0, setup-python v6.3.0 — verified against the GitHub API; cf. CVE-2025-30066), least-privilege `permissions: contents: read`, concurrency cancellation, Python matrix 3.9 + 3.14.

### Changed — docs
- README: four-tier classification, per-locale results table with the honest ja/zh caveat (pre-audit 19/20 was vacuous), reconciled 0.9083 / 11-miss accounting, known-limitations section with the NLI upgrade path and its costs, Python 3.9 EOL disclosure.
- `docs/SCHEMA.md`: schema v3.1 (localized forbidden_claims), tokenizer V0.1, gate semantics incl. demotion and no-KB block reporting, metadata-field annotations, parity/orthography gates, frozen-artifact rule.
- `ROADMAP.md`: M1 must ship ≥1 failing item; M2 adds an NLI-based scorer next to the lexical baseline and moves the Python floor to 3.10.
- `CITATION.cff`: added `version: 0.2.0`, `date-released: 2026-07-19`.
- New: `docs/AUDIT-2026-07-19.md` (full audit trail, 27 sources), `ARCHITECTURE.md`, `docs/TEST_GUIDE.md`, this changelog.

### Trade-offs, stated plainly
- ja/zh-Hans sample agreement moved 19/20 → 18/20: the old number was inflated by vacuous symbol matches; the new one measures language. Honest 18 beats vacuous 19.
- SHA-pinned actions do not auto-receive backported fixes shipped to floating tags (June 2026 checkout backport) — mitigated by Dependabot tracking of SHA pins.
- The gate remains lexical: NegEx-class cue matching is a floor (sensitivity 77.8% / PPV 84.5% on clinical text), not a solution; measuring that floor is the benchmark's purpose.

## [0.1.2] — 2026-07-19
- Neutralized residual "paper" wording; tiers/baseline point at `docs/SCHEMA.md`. (84a05ba)

## [0.1.1] — 2026-07-10
- Reframed around the production benchmark; removed paper-submission references. (f6d0ffb)
- Content verification pass: KPS terminology, interference KB entry, dated QEC/PQC milestones, IBM misconception corrections. (c59f37d)

## [0.1.0] — 2026-07-05
- Public skeleton: 20-item benchmark sample, reference V0 scorer, metric-blindness demo. (2eeb426)
