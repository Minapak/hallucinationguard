#!/usr/bin/env python3
"""Tests for the KB concept wiki and its verification gate (reference/kb_wiki.py).
Stdlib only; exits non-zero on violation.

  python3 tests/test_kb_wiki.py

Covers: building the wiki from fixtures/kb-examples (12 concepts, one source
each), links through shared term pages and shared DOI pages, the gate
(verified text accepted; misconception, fabricated-citation and low-overlap
text queued as unverified with reasons, never ingested), the optional NLI
layer via an injected checker (the real model is exercised only when
torch/transformers are installed), lint, query, the CLI, and the
zero-dependency import contract (no sqlalchemy, no torch).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from reference.kb_wiki import DEFAULT_BANK_ID, KBWiki, nli_available  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def kb_text(kw: KBWiki, cid: str) -> str:
    return kw.entries[cid]["translations"]["en"]["detailedExplanation"]


def build_tests() -> None:
    kw = KBWiki(nli=False).build()
    pages = {p.slug: p for p in kw.store.list_pages(DEFAULT_BANK_ID)}
    check(kw.wiki.bank_id == "hg:wiki:kb", "bank id must be hg:wiki:kb")
    check(len(kw.entries) == 12, f"expected 12 KB concepts, got {len(kw.entries)}")
    check(len(kw.accepted) == 12 and not kw.unverified,
          f"all 12 verified concepts must pass the gate, unverified: "
          f"{[(r.candidate_id, r.reasons) for r in kw.unverified]}")
    sources = sorted(s for s in pages if s.startswith("source-"))
    check(sources == sorted("source-" + cid for cid in kw.entries),
          f"one source page per concept (source_id = concept id), got {sources}")
    c = kw.counts()
    check(c["doi_pages"] == 12, f"expected 12 distinct DOI pages, got {c['doi_pages']}")
    # Rebuild is idempotent (unchanged sources skipped by hash).
    before = {s: p.content_hash for s, p in pages.items()}
    kw.build()
    after = {p.slug: p.content_hash for p in kw.store.list_pages(DEFAULT_BANK_ID)}
    check(before == after, "re-ingesting unchanged sources must not change any page")

    # Shared DOI: Qubit and Superposition both cite Nielsen & Chuang.
    nc = pages.get("doi-10-1017-cbo9780511976667")
    check(nc is not None and set(nc.sources) == {"qubit", "superposition"},
          f"N&C DOI page must carry notes from qubit and superposition, got {nc and nc.sources}")
    check(nc is not None and {"source-qubit", "source-superposition"} <= set(nc.links),
          "N&C DOI page must link both concept sources")
    # Shared term: the Superposition page gathers notes from several concepts,
    # and concept sources reach each other through it.
    sp = pages["superposition"]
    check(len(sp.sources) >= 3 and {"qubit", "superposition"} <= set(sp.sources),
          f"term page 'superposition' must collect several concepts, got {sp.sources}")
    check("superposition" in pages["source-qubit"].links
          and "source-decoherence" in sp.links,
          "source-qubit -> superposition -> source-decoherence path must exist")
    # Notes stay traceable to their source.
    for slug, notes in (("superposition", kw.wiki.notes(sp)),):
        check(all(src in kw.entries for _, src in notes),
              f"{slug}: every note must carry a concept source id")

    # Gate per field, not on the concatenation (documented measurement).
    from score_v0 import asserts_forbidden_claim
    concat_blocks = [cid for cid, e in kw.entries.items()
                     if asserts_forbidden_claim(kw.concept_text(e), kw.claims["en"], "en")]
    check(sorted(concat_blocks) == ["grover_algorithm", "measurement", "quantum_error_correction"],
          f"documented concatenation false blocks changed: {concat_blocks}")

    # Non-en build uses that locale's titles and still passes the gate.
    ko = KBWiki(locale="ko", nli=False).build()
    check(len(ko.accepted) == 12 and not ko.unverified, "ko build: all 12 must pass")
    check(ko.store.get_page(DEFAULT_BANK_ID, "source-qubit").title == "큐비트",
          "ko build: source titles come from the ko block")


def gate_tests() -> None:
    kw = KBWiki(nli=False).build()
    n_pages = len(kw.store.list_pages(DEFAULT_BANK_ID))

    # Accepted: KB text served verbatim; a faithful extract above threshold.
    r = kw.propose(kb_text(kw, "qubit"), "qubit")
    check(r.accepted and r.scores["tier"] == "verifiedKnowledgeBase",
          f"verbatim KB text must be accepted, got {r.reasons}")
    check(kw.store.get_page(DEFAULT_BANK_ID, "source-" + r.candidate_id) is not None,
          "accepted proposal must be ingested as its own source page")
    ent = kb_text(kw, "entanglement")
    r = kw.propose(ent[: len(ent) * 2 // 3])
    check(r.accepted and r.concept_id == "entanglement" and r.scores["anchor"] == "auto"
          and r.scores["tier"] == "aiValidatedAgainstKB",
          f"faithful extract must auto-anchor to entanglement and validate, got {r.concept_id} {r.reasons}")
    check(len(kw.store.list_pages(DEFAULT_BANK_ID)) > n_pages, "accepted proposals add pages")
    n_pages = len(kw.store.list_pages(DEFAULT_BANK_ID))

    # Queued: misconception asserted (V0 forbidden-claim gate).
    r = kw.propose("Entanglement transmits information faster than light. " + ent, "entanglement")
    check(not r.accepted and r.scores["forbidden_claim"]
          == "Entanglement transmits information faster than light",
          f"FTL assertion must be queued by the forbidden-claim gate, got {r.reasons}")
    # A correct denial of the same misconception is not blocked.
    r = kw.propose("Entanglement does not transmit information faster than light. " + ent,
                   "entanglement")
    check(r.accepted, f"correct denial must pass, got {r.reasons}")
    n_pages = len(kw.store.list_pages(DEFAULT_BANK_ID))
    # Queued: fabricated citation (hallucinatedCitation).
    r = kw.propose(kb_text(kw, "qubit") + " See doi:10.1103/PhysRevX.99.123456.", "qubit")
    check(not r.accepted and r.scores["unknown_citations"] == ["doi:10.1103/physrevx.99.123456"],
          f"fabricated DOI must be queued, got {r.reasons}")
    # Queued: correct but too short to clear the 0.30 overlap floor.
    r = kw.propose("A qubit is the basic unit of quantum information.", "qubit")
    check(not r.accepted and r.scores["tier"] == "aiWithLowConfidence",
          f"short candidate below threshold must be queued, got {r.scores}")
    # Queued: unknown concept and empty text.
    check(not kw.propose(kb_text(kw, "qubit"), "no_such_concept").accepted, "unknown concept queued")
    check(not kw.propose("   ").accepted, "empty candidate queued")

    check(len(kw.store.list_pages(DEFAULT_BANK_ID)) == n_pages,
          "queued candidates must never be ingested")
    c = kw.counts()
    check(c["unverified_proposals"] == 5 and c["unverified_concepts"] == 0,
          f"unverified queue counts wrong: {c}")
    check(all(q.reasons and q.text is not None for q in kw.unverified),
          "every queued entry must carry reasons and its text")

    # Concept-level gate: grades that are not verified are queued, not ingested.
    entry = dict(kw.entries["superposition"], id="superposition", certainty="open")
    res = kw.gate_concept(entry)
    check(not res.accepted and "not a verified grade" in res.reasons[0], "open grade must queue")
    entry = dict(kw.entries["superposition"], certainty="established", sourceRef="Somebody (2020)",
                 certaintyNote="")
    check(not kw.gate_concept(entry).accepted, "established without DOI must queue")
    entry = dict(kw.entries["superposition"], certainty="timebound")
    check(not kw.gate_concept(entry).accepted, "timebound without asOfDate must queue")
    entry["asOfDate"] = "2026-07-10"
    check(kw.gate_concept(entry).accepted, "timebound with asOfDate passes")


class _FakeVerdict:
    def __init__(self, label: str) -> None:
        self.label = label


class _FakeChecker:
    """Stands in for harness/nli_v1.NLIChecker: contradicts any 'never'."""
    def check(self, kb: str, answer: str) -> _FakeVerdict:
        return _FakeVerdict("nli_contradicted" if " never " in f" {answer} " else "nli_supported")


def nli_tests() -> None:
    kw = KBWiki(nli=_FakeChecker()).build()
    q = kb_text(kw, "qubit")
    inverted = q.replace("Upon measurement, the qubit collapses to either",
                         "Measurement never changes the qubit; it simply reveals")
    r = kw.propose(inverted, "qubit")
    check(r.scores["tier"] == "aiValidatedAgainstKB" and r.scores["forbidden_claim"] is None,
          "inversion must pass the lexical layers (the gap the NLI layer exists for)")
    check(not r.accepted and r.scores["nli"] == "nli_contradicted",
          f"NLI contradiction must queue the candidate in en, got {r.reasons}")
    check(kw.propose(q, "qubit").accepted, "NLI-supported verbatim text accepted")
    check(KBWiki(nli=False).build().propose(inverted, "qubit").scores["nli"] == "off",
          "nli=False reports the layer as off")
    if not nli_available():
        check(KBWiki().build().propose(q, "qubit").scores["nli"] == "unavailable",
              "auto mode without torch reports the layer as unavailable")
    # Outside GATE_LOCALES the NLI verdict is shadow-only.
    de = KBWiki(locale="de", nli=_FakeChecker()).build()
    dq = de.entries["qubit"]["translations"]["de"]["detailedExplanation"]
    r = de.propose(dq + " Die Messung ändert never nichts.", "qubit")
    check(r.scores.get("nli_scope", "").startswith("shadow")
          and not any(x.startswith("NLI") for x in r.reasons),
          f"de NLI must be shadow-only, got {r.scores} {r.reasons}")


def lint_query_tests() -> None:
    kw = KBWiki(nli=False).build()
    issues = [(i.kind, i.slug, i.detail) for i in kw.lint()]
    # QEC's "[[144,12,12]]" code-parameter notation sits in note text, not in the
    # Related section, so it must not be reported as a broken wiki link.
    check(issues == [], f"unexpected lint issues: {issues}")
    check(not [i for i in issues if i[0] in ("orphan", "no-outlinks", "empty")],
          "no orphan / no-outlinks / empty pages expected")
    hits = kw.query("what does measurement do to a qubit")
    check(hits and hits[0].page.slug == "qubit", f"top hit should be qubit, got {[h.page.slug for h in hits]}")
    hits = kw.query("Nielsen Chuang textbook doi")
    check(any(h.page.slug == "doi-10-1017-cbo9780511976667" for h in hits),
          "DOI page must be reachable by query")
    check(kw.query("zzzz-no-such-term") == [], "nonsense query returns no hits")


def cli_tests() -> None:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "reference.kb_wiki", *args, "--no-nli"],
                              capture_output=True, text=True, cwd=str(REPO))

    r = run("build", "--json")
    out = json.loads(r.stdout) if r.returncode == 0 else {}
    check(r.returncode == 0 and out.get("counts", {}).get("source_pages") == 12
          and out["counts"]["unverified"] == 0 and "## Sources (12)" in out.get("index", ""),
          f"cli build --json failed: {r.returncode} {r.stderr[-300:]}")
    r = run("lint")
    check(r.returncode == 0 and "lint issues:" in r.stdout and "unverified queue 0" in r.stdout,
          f"cli lint failed: {r.returncode} {r.stderr[-300:]}")
    r = run("query", "decoherence and error correction", "--json")
    check(r.returncode == 0 and json.loads(r.stdout)["hits"], f"cli query failed: {r.stderr[-300:]}")
    r = run("propose", "Entanglement transmits information faster than light.", "--concept",
            "entanglement", "--json")
    res = json.loads(r.stdout) if r.returncode == 0 else {}
    check(res.get("result", {}).get("accepted") is False and res["counts"]["unverified_proposals"] == 1,
          f"cli propose must queue a misconception: {r.stdout[-300:]} {r.stderr[-300:]}")
    r = run("query")
    check(r.returncode == 2, f"cli query without text must exit 2, got {r.returncode}")
    # Plain-script invocation also works.
    r = subprocess.run([sys.executable, str(REPO / "reference/kb_wiki.py"), "lint", "--no-nli"],
                       capture_output=True, text=True, cwd=str(REPO / "fixtures"))
    check(r.returncode == 0 and "lint issues:" in r.stdout,
          f"script invocation failed: {r.returncode} {r.stderr[-300:]}")


def zero_dependency_tests() -> None:
    # Block sqlalchemy and torch outright: any import attempt raises ImportError.
    code = (
        "import sys\n"
        "for m in ('sqlalchemy', 'torch', 'transformers'): sys.modules[m] = None\n"
        "import reference.sq_memory as m\n"
        "from reference.sq_memory import Wiki, InMemoryStore\n"
        "from reference.sq_memory import store\n"
        "assert store.HAVE_SQLALCHEMY is False\n"
        "from reference.kb_wiki import KBWiki, nli_available\n"
        "assert not nli_available()\n"
        "kw = KBWiki().build()\n"
        "assert kw.counts()['source_pages'] == 12\n"
        "assert kw.propose(kw.concept_text(kw.entries['qubit'])).scores['nli'] == 'unavailable'\n"
        "print('ok', m.__version__)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO))
    check(r.returncode == 0 and r.stdout.startswith("ok"),
          f"zero-dependency import failed: {r.stdout} {r.stderr[-500:]}")


def main() -> int:
    build_tests()
    gate_tests()
    nli_tests()
    lint_query_tests()
    cli_tests()
    zero_dependency_tests()
    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all kb wiki checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
