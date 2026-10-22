# CVPR-0 FULL=0 diagnostic synthesis — 2026-10-22

Status: `COMPLETE_DIAGNOSTIC_EVIDENCE_NOT_SCIENTIFIC_ACCEPTANCE`.

This document closes the CVPR-0 diagnostic substage only. It does not close
CVPR-0 as a scientific method stage, Goal A/B/C, formal v2 frontend
qualification, or OCD/MOT evaluation. No TAO Test labels were read, and no
Val result was used for model or mechanism selection.

## Evidence-backed diagnosis

The old FULL=0 behavior is not supported by a single proven root. The current
evidence supports a coupled failure bundle:

1. **Representation geometry / R.** The old Train-development Raw p16
   representation has uncentered effective rank `55.5517`, while A1/A2 are
   approximately `6.79–9.07`. On only nine pseudo-Novel development queries,
   Raw macro Rank-1 is `0.25`, while adapted A1/A2 values are `0.0833–0.1667`.
   The same-capacity static control is also `0.1667`, so the evidence does not
   establish that temporal adaptation improves Novel generalization.

2. **Candidate/action separation / C.** All `249/249` Known queries have a
   Known prototype available; Known Top-1 hits are `199/194/198` and Top-8
   hits are `248/245/248` for the three seeds. The candidate-only Top-8 oracle
   remains `H=0`, whereas an action-only oracle reaches diagnostic upper-bound
   H values `0.8884/0.8304/0.8394`. Thus candidate rank contributes errors,
   but candidate availability alone cannot explain the collapse. These oracle
   values are not deployed-method scores because they use GT actions and alter
   later history.

3. **First-error propagation / M.** In the representative seed-1027 p16
   trace, the first bad matched-Known anonymous write is at sequence `144`,
   the first proven cross-category conflict at `146`, and `106/113` later
   Existing writes reuse a state that already has cross-video history after it
   is proven mixed. Suppressing only the first bad write still gives `H=0`.
   This supports irreversible contamination as a contributor, not a proof that
   a particular reversible-memory rule will succeed.

4. **Protocol confounding.** Old numeric physical-ID tie ordering is not
   invariant: learned ID-renumber pairs changed H in `2/90` cases and fixed
   Raw-B1 pairs changed H in `7/10`. The new current-observation router passes
   the registered 57-video mutation audit, but it has not yet been integrated
   with a learned controller and dense frontend. Old FULL=0 results cannot be
   interpreted without this protocol repair.

The resulting scientific statement is therefore:

> FULL=0 is reproducibly associated with representation collapse, controller
> action mismatch, and irreversible mixed-memory propagation under a protocol
> that also had ID-order dependence. The relative causal contribution of these
> factors is not uniquely identified by the existing retrospective traces.

## Mechanism readiness matrix

| mechanism | current evidence | what remains before acceptance |
|---|---|---|
| R — raw-preserving temporal representation | Geometry and matched static controls support a falsifiable hypothesis; no general Novel gain | Fresh legal Train folds, fixed capacity/data, causal prefixes, and prospective predicted-stream evaluation |
| C — candidate-conditioned calibrated decision | Top-K/action oracle separation identifies a controller bottleneck and a candidate ceiling | Candidate expansion versus scoring under one frozen candidate protocol, with no GT action oracle and independent orders |
| M — versioned contradiction-aware memory | First-error chronology and mixed-state reuse are directly observed; current-source version audits pass plumbing checks | Prospective closed-loop memory, identity/reverse/future probes, purity/coverage accounting, and independent orders |

The two-video joint R/C/M pilot reached the graph and performed two optimizer
steps, but it is an interface/training smoke only: it is not a development
metric, frontend qualification, model selection, or Goal acceptance.

## Scope and preserved boundaries

- The CVPR-0 ledger includes the frozen FULL audit, checkpoint replays,
  Top-K/oracle controls, first-error and state trajectories, representation
  geometry, capacity/known-count controls, matched predicted-vs-reference
  inputs, future-information shadows, and the current-input router audit.
- The minimum source-review requirement is independently verified: 12 scoped
  actual paper/source audits plus 3 licensed extras. Those reviews constrain
  what may be borrowed; they do not constitute upstream reproduction.
- The formal v2 admission gate remains closed because the public predicted
  stream, four frozen Val causal-order streams, formal frontend provenance,
  and selected-front-end cache manifest are absent. No formal OCD/MOT metric
  is reported here.

## Authoritative receipts

| receipt | SHA256 |
|---|---|
| `outputs/trackocd_cvpr/cvpr0/root_cause_matrix.json` | `fee1e1452db5f3a695d2a4a7bf298d53e4e3c345ad1f585c8a23de2b18076caa` |
| `outputs/trackocd_cvpr/cvpr0/experiment_manifest.json` | `775f858690f114b7a5f67204c5f917e2ba74a76203b18392b25d45972d5bc3d0` |
| `outputs/trackocd_cvpr/cvpr0/matched_train_semantics_v1/result.json` | `a0ee9f8d5891c3b9abc30ad5ff5b9d7c8cfb018e223bd3edcea460c96b79111d` |
| `outputs/trackocd_cvpr/cvpr0/fixed_history_shadow_v1/result.json` | `baff0a80b892f59981ff722d7ba9b25d1752621114534483579b1b5d476984f5` |
| `outputs/trackocd_cvpr/cvpr0/causal_router_interface_v1/result.json` | `5d0a1f80f35a5e870f3a5bb9289e719199d37dbf5e63169d17a87f56c838039e` |
| `outputs/trackocd_cvpr/cvpr1/review_coverage_verification.json` | `f5653edb762e299e7b07f7ab69244107ea538d0609ad7f9204cbe695718e5f9a` |

Next scientific gate: recover and admit the lawful formal frontend/provenance
assets, then run the frozen predicted Train-development protocol with the R,
C, and M arms under the repaired causal ordering. Until that gate opens, all
existing R/C/M numbers remain diagnostics or interface evidence.
