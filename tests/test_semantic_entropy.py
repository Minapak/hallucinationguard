#!/usr/bin/env python3
"""Tests for harness/semantic_entropy_v1.py with an injected fake judge (stdlib only).

  python3 tests/test_semantic_entropy.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "harness"))
from semantic_entropy_v1 import (  # noqa: E402
    cluster_by_meaning, consistency, discrete_semantic_entropy,
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


# Fake judge: answers entail each other iff they share the same first word ("meaning key").
def _key(s: str) -> str:
    return s.split()[0].casefold().strip(".,!?")


def fake(pairs):
    out = []
    for p, h in pairs:
        same = _key(p) == _key(h)
        out.append({"entailment": 0.9 if same else 0.05, "neutral": 0.05, "contradiction": 0.05 if same else 0.9})
    return out


def one_way(pairs):
    # p entails h only when p is the longer string: asymmetric → must NOT cluster.
    return [{"entailment": 0.9 if len(p) > len(h) else 0.1, "neutral": 0.05, "contradiction": 0.05} for p, h in pairs]


answers = ["Paris is the capital.", "paris, in France.", "Berlin is the capital.", "Paris obviously.", "Rome."]
cl = cluster_by_meaning(answers, fake)
check(cl == [[0, 1, 3], [2], [4]], f"clusters by meaning key: {cl}")
se = discrete_semantic_entropy(cl, len(answers))
expect = -(3 / 5 * math.log(3 / 5) + 2 * (1 / 5 * math.log(1 / 5)))
check(abs(se - expect) < 1e-9, f"discrete SE {se} != {expect}")

one = cluster_by_meaning(["Paris.", "Paris again.", "Paris third."], fake)
check(one == [[0, 1, 2]] and discrete_semantic_entropy(one, 3) == 0.0, "single meaning → entropy 0")
check(discrete_semantic_entropy([], 0) == 0.0, "empty")

allsep = cluster_by_meaning(["A x.", "B y.", "C z."], fake)
r = consistency(["A x.", "B y.", "C z."], fake)
check(allsep == [[0], [1], [2]] and r["normalized"] == 1.0, f"all different → normalized 1.0: {r}")
check(consistency(["Paris."], fake)["normalized"] == 0.0, "single sample → 0")

asym = cluster_by_meaning(["short", "a much longer answer"], one_way)
check(asym == [[0], [1]], f"one-way entailment never merges (bidirectional rule): {asym}")

r = consistency(answers, fake)
check(set(r) == {"version", "samples", "clusters", "semantic_entropy", "normalized", "entail_min"}
      and r["samples"] == 5 and 0 < r["normalized"] < 1, f"consistency receipt: {r}")
check(cluster_by_meaning(answers, fake) == cl, "deterministic")

if failures:
    print("\n".join(f"FAIL: {f}" for f in failures))
    sys.exit(1)
print("all semantic-entropy checks passed")
