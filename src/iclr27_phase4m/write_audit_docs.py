"""Write Phase 4M audit docs (numbers are from the audit CSVs)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
DOC = ROOT / "docs" / "iclr27_phase4m"


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "FORCED_DECISION_AUDIT.md").write_text(
        """# Phase 4M Forced-Decision Audit

The corrected decision dataset (v2) replays the event streams in true
causal order and reconstructs the exact online quantities:

- sticky same-track reuses are **continuations**, not decisions;
- a `birth` event only creates a global prototype the first time its
  `sem_id` appears; later `birth` events with an existing `sem_id` were
  routed by `NovelSemanticMemory.propose` back to the existing prototype,
  so the actual memory action is EXISTING while the branch label says NEW
  (the `is_new` return was previously discarded);
- track prefix P1 is the mean of the last <=8 single-frame M2 embeddings,
  reconstructed from the association history and the shared embedding
  cache (not a single-frame proxy).

Decision counts (all novel-identity-relevant events, final-code M0):

| tag | EXISTING (soft) | NEW branch -> actual EXISTING | NEW branch -> actual NEW | total |
|---|---:|---:|---:|---:|
| j1b | 10990 | 325 | 106 | 11421 |
| b1 | 7385 | 353 | 535 | 8273 |
| b2 | 3839 | 5 | 672 | 4516 |

The Phase 4L "birth" labels therefore **over-state NEW decisions**: 75%
of j1b NEW-branch events (325/431) actually reused an existing
prototype, and the same-track sticky stream (19,123 reuse events in j1b)
contains no identity decision at all.

**Anchor note.**  Phase 4L stored dev provenance (`prov_dev_j1b`, 116
novel IDs) predates the final two pipeline fixes; the final code
reproduces the Phase 4L held-out anchor exactly and produces 106 novel
IDs on dev.  Phase 4M uses the final-code M0 dev run as the comparison
anchor and rebuilds this audit from `prov/dev_j1b`.
""")
    (DOC / "OVERBIRTH_AUDIT.md").write_text(
        """# Phase 4M Overbirth Audit

Overbirth = the system commits NEW_NOVEL (a new global prototype) while
retrospective GT shows the query shares the real category of the best
existing prototype.

Corrected counts (GT-novel queries only):

| tag | actual NEW births | overbirths | overbirth rate |
|---|---:|---:|---:|
| j1b | 106 | 0 | 0.0% |
| b1 | 535 | 0 | 0.0% |
| b2 | 672 | 9 | 1.3% |

Key finding: **overbirth is not the dominant failure in the corrected
anchor** (j1b: 0/106).  Phase 4L's prototype explosion (116 -> 364-672)
is mostly FP births (j1b 98, b1 513, b2 632) plus genuinely-new births,
not overbirth.  The B2 entropy gate, which suppresses reuse, produces the
largest overbirth share (9/47 novel decisions), confirming that
suppressing reuse transfers merge error into overbirth, but the absolute
overbirth count remains small.
""")
    (DOC / "WRONG_REUSE_AUDIT.md").write_text(
        """# Phase 4M Wrong-Reuse Audit

Wrong reuse = EXISTING_NOVEL decision whose best prototype's majority GT
category differs from the query's real category (or the query is
known/FP).

GT-novel queries:

| tag | novel decisions | CORRECT_EXISTING | WRONG_EXISTING | correct-new | overbirth |
|---|---:|---:|---:|---:|---:|
| j1b | 108 | 12 | 93 | 3 | 0 |
| b1 | 101 | 16 | 75 | 10 | 0 |
| b2 | 47 | 7 | 22 | 9 | 9 |

Wrong reuse dominates the corrected anchor (86% of novel decisions).
Including FP/known queries, j1b has 11,303 WRONG_EXISTING events (96% of
the EXISTING decisions), i.e. the polluted prototypes absorb the FP
stream.  The B candidates trade this error for prototype explosion.
""")
    (DOC / "AMBIGUITY_GEOMETRY_AUDIT.md").write_text(
        """# Phase 4M Ambiguity Geometry Audit

Question H2: is the EXISTING-vs-NEW correctness separable online from
causal geometry?

Median geometry per class (novel queries):

| tag | class | best cos | margin | entropy | novel-known | z-score |
|---|---|---:|---:|---:|---:|---:|
| j1b | CORRECT_EXISTING | 0.680 | 0.041 | 1.606 | 0.166 | 1.19 |
| j1b | WRONG_EXISTING | 0.684 | 0.045 | 1.608 | 0.178 | 1.95 |
| j1b | CORRECT_NEW | 0.351 | 0.061 | 1.607 | 0.083 | 0.59 |
| b1 | CORRECT_EXISTING | 0.790 | 0.132 | 1.606 | 0.369 | 1.39 |
| b1 | WRONG_EXISTING | 0.710 | 0.034 | 1.609 | 0.234 | 1.49 |
| b2 | CORRECT_EXISTING | 0.705 | 0.021 | 1.608 | 0.303 | 1.04 |
| b2 | WRONG_EXISTING | 0.665 | 0.019 | 1.609 | 0.271 | 1.12 |

AUROC of each feature for correct-vs-wrong on novel decisions:

| feature | j1b | b1 | b2 |
|---|---:|---:|---:|
| best cos | 0.366 | 0.706 | 0.556 |
| margin | 0.513 | 0.614 | 0.575 |
| entropy | 0.338 | 0.281 | 0.506 |
| novel-known | 0.457 | 0.718 | 0.526 |
| z-score | 0.138 | 0.496 | 0.482 |

Conclusion: on the primary corrected anchor (j1b) every feature is
<=0.52; on b1, best-cos and novel-known reach 0.71-0.72.  High absolute
similarity is *anti-predictive* on j1b because the prototypes are
FP-polluted.  A clean high-confidence region exists only for b1's
relative-matching memory.
""")
    (DOC / "RETROSPECTIVE_DEFERRAL_AUDIT.md").write_text(
        """# Phase 4M Retrospective Deferral Audit

For every GT-novel decision we replay a counterfactual in which the
track's global memory writes at/after t are skipped (no
self-confirming prototype update) and ask whether the first observation
at/after t+1, t+2, t+4, t+8 resolves correctly under the anchor rule
(best cos >= 0.6 -> EXISTING, else NEW).  Other tracks' events are held
fixed (documented approximation; the deployed policy is tested directly
on dev/held-out).

| tag | decisions | resolved correctly by t+8 | terminated before t+8 |
|---|---:|---:|---:|
| j1b | 108 | 31 (28.7%) | 66 (61.1%) |
| b1 | 101 | 30 (29.7%) | 60 (59.4%) |
| b2 | 47 | 20 (42.6%) | 36 (76.6%) |

Per-k outcomes (j1b): t+1: 24 correct / 86 with observation; t+2: 18/73;
t+4: 10/56; t+8: 5/42.  Correct resolutions *decrease* with k because
the counterfactual memory never receives this track's updates, while the
rest of the polluted world stays fixed.

Conclusion: future causal evidence resolves a real minority (24-43%),
but most wrong decisions remain wrong or terminate unresolved.  This is
a PARTIAL deferral signal, not a strong one.
""")
    (DOC / "TIME_TO_RESOLUTION_AUDIT.md").write_text(
        """# Phase 4M Time-to-Resolution Audit

For deferred novel decisions, the resolution latency (first observation
at/after t+k that is correct under the anchor rule) is dominated by
short horizons:

- j1b: most correct resolutions occur by t+1 (24/31); by t+8 only 5 more.
- b1: 22/30 correct by t+1; 14 by t+8.
- b2: 17/20 correct by t+1.

This means that when deferral works, it works quickly (1-2 frames), and
longer waiting mostly accumulates unresolved-at-termination rather than
new correct resolutions.  The key bottleneck is not latency but the
~60-77% of tracks that terminate before t+8 and the ~55-74% of observed
future prefixes that remain wrong even after deferral.

Data: `outputs/iclr27_phase4m/audit/retrospective_<tag>_k.csv`.
""")
    (DOC / "DEFERRAL_PARETO_AUDIT.md").write_text(
        """# Phase 4M Deferral Pareto Audit

Rules evaluated on the combined dev novel decisions (j1b rows shown;
full table in `deferral_pareto.csv`):

| rule | defer % | immediate decided error % | eventual coverage of deferred | unresolved-at-term |
|---|---:|---:|---:|---:|
| none (forced) | 0 | 86.1 | - | - |
| margin < 0.05 | 53.7 | 86.0 | 25.9 | 37 |
| margin < 0.10 | 78.7 | 82.6 | 27.1 | 53 |
| best < 0.70 | 55.6 | 89.6 | 33.3 | 32 |
| best < 0.75 | 73.2 | 93.1 | 27.9 | 47 |
| novel-known < 0.20 | 59.3 | 86.4 | 26.6 | 37 |
| novel-known < 0.30 | 78.7 | 87.0 | 27.1 | 52 |
| entropy > 1.6 | 87.0 | 85.7 | 26.6 | 59 |
| best<.75 or margin<.05 | 83.3 | 94.4 | 28.9 | 53 |

No rule lowers the decided-set error below ~83%; the best rules defer
~50-80% of decisions to gain at most ~33% eventual coverage on the
deferred set, with 32-59 unresolved-at-termination.  The coverage-risk
trade-off is therefore poor on the corrected anchor: ambiguity scores
are too weak to separate defer-worthy cases from harmfully-deferred
correct cases (14-20% of deferred decisions were correct at t).
""")


if __name__ == "__main__":
    main()
