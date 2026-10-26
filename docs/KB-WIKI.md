# KB concept wiki with a verification gate (v0.6.0)

`reference/kb_wiki.py` turns the 12 verified concepts in `fixtures/kb-examples/` into an
interlinked wiki, and puts every write behind the HallucinationGuard gate. Zero dependencies,
CPython 3.9+, offline, no LLM (rules mode).

```bash
python3 -m reference.kb_wiki build              # gate + ingest, print the index
python3 -m reference.kb_wiki lint               # wiki lint issues
python3 -m reference.kb_wiki query "what does measurement do to a qubit"
python3 -m reference.kb_wiki propose "<candidate text>" --concept qubit
# all commands: --json, --locale {en,ko,ja,zh-Hans,es,fr,de}, --no-nli, --kb DIR
```

Every command builds a fresh in-memory wiki and ends with the page count and the
**unverified queue** counts (concepts / proposals).

## Structure

- Wiki engine: the shared `Wiki` class in `reference/sq_memory/` (bank id `hg:wiki:kb`,
  `InMemoryStore`). It is a verbatim copy of the SwiftQuantum memory module, installed with
  the owner's sync tool and never hand-edited here. It implements Karpathy's LLM-Wiki pattern
  (sources → interlinked pages; ingest / query / lint; index and log) independently. Attribution:
  `reference/sq_memory/NOTICE.md` and the repo `NOTICE`.
- One **source per concept** (`source_id` = concept id, page `source-<id>`).
- Explicit entities, so pages link through what concepts share:
  - **term pages**: the 12 concept titles in the build locale ("Superposition" collects notes
    from Qubit, Decoherence, Bloch Sphere, Interference, …);
  - **DOI pages**: one per DOI cited in `sourceRef` / `certaintyNote` (12 in total). Qubit and
    Superposition both cite Nielsen & Chuang, so `doi-10-1017-cbo9780511976667` links the two.
- Ingested per concept: short definition, detailed explanation, key takeaways, and one generated
  line per cited DOI (`"Qubit cites doi:… (primary source)."`). Raw citation strings are not
  ingested, because abbreviations like "Phys. Rev. A" break into meaningless note fragments.
  Analogies and the misconception lists are left out: analogies are loose on purpose (the
  spinning-coin analogy is itself the HGM01 misconception).
- Every note line keeps its source id, so each statement on a page traces to a KB concept or
  an accepted proposal.

## The gate

A candidate that fails is added to `KBWiki.unverified` with its reasons and scores. It is
**not ingested**. Nothing is deleted, so the queue is the review list.

### KB concepts (at build time)

| Check | Pass criterion |
|---|---|
| G1 certainty | `established` with ≥ 1 DOI in sourceRef/certaintyNote, or `timebound` with `asOfDate` (YYYY-MM-DD). `verify` / `open` / `null` are queued. Grades are never auto-promoted. |
| G2 locale | the build locale has its own `detailedExplanation`. There is no silent fallback to en. |
| G3 score_v0 | the forbidden-claim gate (`asserts_forbidden_claim`, union of every sample fixture's `forbidden_claims`, locale-aware) does not fire on **any ingested field**, and the KB text has ≥ `MIN_KB_TOKENS` (5) tokens |

The gate checks each field separately, not the concatenated text. The gate's token recall is
taken over the whole text. On the ~1–2k-character concatenation, scattered tokens reach the 0.60
recall threshold, and 3 of the 12 verified concepts false-block (grover_algorithm, measurement,
quantum_error_correction). Checked field by field, 0 of 12 do. This result is pinned in
`tests/test_kb_wiki.py`. For KB concepts, the V0 overlap tier is trivially 1.0 against their own
text. So score_v0 adds the misconception gate here, not a truth check: the concepts are verified
upstream (verifier records, DOIs).

Result today: all 12 concepts pass in all 7 locales (`unverified queue 0`).

### Proposed text (`KBWiki.propose(candidate_text, concept_id=None)`)

| Check | Pass criterion |
|---|---|
| P1 anchor | the named concept. If none is named, the concept with the highest V0 one-sided overlap is used. An unknown or absent anchor is queued. |
| P2 score_v0 tier | `classify(anchor detailedExplanation, candidate)` ∈ {`verifiedKnowledgeBase`, `aiValidatedAgainstKB`}, i.e. overlap \|KB ∩ candidate\| / \|KB\| ≥ **0.30** (the production `THRESHOLD`) |
| P3 misconception | `asserts_forbidden_claim()` returns nothing (claims as in G3) |
| P4 citations | every DOI / arXiv / FIPS identifier in the candidate exists in the anchor's reference corpus (`dimensions_v1.unverified_citations`, the `hallucinatedCitation` check) |
| P5 NLI (optional) | only when `harness/nli_v1.py` can run (torch + transformers importable) or a checker is passed in, and only in its measured gate scope {en, es, fr}: a sentence-level `nli_contradicted` verdict queues the candidate. Outside that scope the verdict is recorded as shadow. Without the deps it is reported as `"unavailable"` and skipped. |

Accepted proposals are ingested as their own source (`proposal-<sha256[:10]>`), linked to the
same term and DOI pages. `scores.attribution_required` (overlap ≥ 0.85) is recorded but not
used to block.

### Thresholds, stated honestly

- **0.30 overlap is a lexical floor that errs toward review.** A short but correct sentence
  covering < 30 % of the KB entry's tokens is queued (pinned: "A qubit is the basic unit of
  quantum information." → `aiWithLowConfidence`). Semantic inversion with high overlap passes
  P2–P4. That is the benchmark's central finding, and it is the reason for P5.
- **P3 inherits every documented V0 gate limit**: paraphrases below 0.60 recall escape it, and a
  negation cue anywhere in the decisive sentence suppresses it, even when the cue belongs to a
  different clause. Example: "Entanglement transmits information faster than light, so …
  cannot be factored …" in one sentence is *not* caught. The same claim as its own sentence is
  caught.
- **P5 uses the plain contradiction label**, which is stricter than the V2 gate. The V2 gate
  acts only on low-confidence answers, and P2 already rejects those. P5 therefore inherits the
  measured false alarms on formula-dense correct text
  ([NLI-GAP-2026-07-27.md](NLI-GAP-2026-07-27.md)). That is tolerable only because a false
  alarm queues the text for review and deletes nothing. Measured with the real model (harness
  venv): a qubit text whose collapse sentence is inverted to "Measurement never changes the
  qubit" passes P2–P4 (overlap 0.95) and is queued by P5. The verbatim KB text is accepted.

### Lint and bracket notation

The QEC entry writes the gross code's parameters as `[[144,12,12]]`. The shared `Wiki` lint
only reads wiki links from a page's `## Related` section, so bracket notation inside note text
is not reported as a broken link (`lint` → 0 issues on the current KB).

## Zero-dependency contract

`reference/sq_memory/store.py` imports SQLAlchemy softly (`HAVE_SQLALCHEMY = False` when it is
absent). Importing `reference.sq_memory`, `Wiki`, `InMemoryStore` and `reference.kb_wiki` needs
the standard library only. This was verified in fresh, package-free venvs on CPython 3.9.6 and
3.14.4, and `tests/test_kb_wiki.py` re-checks it with `sqlalchemy`, `torch` and `transformers`
blocked. `create_tables` / `SQLStore` need SQLAlchemy and are not used here. A caller that has
SQLAlchemy may pass `KBWiki(store=SQLStore(engine))`.

## Scope

This is a reference build over 12 sample concepts, not a production service. The queue lives in
memory for the life of the `KBWiki` object. The wiki does not decide truth: it only admits text
that the repo's documented gates fail to object to.
