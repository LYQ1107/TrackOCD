"""Phase 4N results docs + final report (numbers embedded)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4n" / "audit"
HO = ROOT / "outputs" / "iclr27_phase4n" / "heldout"
DOC = ROOT / "docs" / "iclr27_phase4n"


def load_csv(p):
    return list(csv.DictReader(open(p)))


def get(rows, tag, key, default=""):
    for r in rows:
        if r["tag"] == tag:
            return r.get(key, default)
    return default


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    comp = load_csv(ROOT / "outputs" / "iclr27_phase4n" / "dev" /
                    "component_comparison.csv")

    # ---- memory quality re-audit ----
    dev_rows = []
    for tag in ("m3", "n2"):
        r = next(x for x in comp if x["tag"] == tag)
        dev_rows.append(r)
    (DOC / "MEMORY_QUALITY_REAUDIT.md").write_text(
        """# Phase 4N Memory Quality Re-Audit

Dev (M3 anchor vs N2 validity gate):

| tag | protos | USEFUL | POLLUTING | LOW_EVIDENCE | fp_reuse | net |
|---|---:|---:|---:|---:|---:|---:|
| m3 | 40 | 0 | 40 | 0 | 0.9658 | -11 |
| n2 | 39 | 0 | 38 | 0 | 0.9651 | -12 |

Corrected-GT held-out (Phase 4M frozen candidates re-evaluated):

| tag | protos | USEFUL | POLLUTING | LOW_EVIDENCE | fp_reuse | net | birth fp/known/novel |
|---|---:|---:|---:|---:|---:|---:|---:|
| j1b | 117 | 0 | 114 | 3 | 0.9598 | 0 | 110/4/3 |
| m1 | 55 | 0 | 55 | 0 | 0.9599 | 0 | 55/0/0 |
| m3 | 39 | 0 | 38 | 1 | 0.9628 | 0 | 36/3/0 |

Verdict: `MEMORY_STILL_FP_DOMINATED`.  The validity gate does not change
the GT-based FP reuse share (members are counted regardless of weight),
and USEFUL remains 0 on dev and corrected held-out.
""")

    # ---- held-out results (corrected GT) ----
    (DOC / "HELDOUT_RESULTS.md").write_text(
        """# Phase 4N Held-Out Results (corrected GT)

## Erratum: Phase 4L/4M held-out GT was category-collapsed

The Phase 4L held-out GT was built from
`third_party/SimOWT/datasets/tao/annotations/val_split/all.json`, in
which every annotation has category id 1 ("coco").  Under TrackOCD's
known set this makes **every held-out object GT-novel**, so:

- the old held-out N2K (0.52) was an artifact (all objects were
  "novel", and the gate routed many as known);
- known-class accuracy and known/novel routing were undefined on
  held-out;
- the Phase 4M "resolved accuracy 0.76->0.87" on held-out was inflated
  because all GT categories were equal (prototype majority category
  matched trivially).

Phase 4N rebuilt the same 24-video / 887-frame GT from the original
TAO validation.json (28 categories; 67.8% known-category annotations;
`outputs/iclr27_phase4n/audit/validation_heldout_tao_corrected.json`).
Tracking metrics are category-agnostic (TAO_OW) and unchanged.

## Corrected results (frozen Phase 4M candidates, one-shot)

| tag | routing | K2N | N2K | known_class_acc | novel_cons | protos | novel_resolved_acc | novel decisions (CE/WE/CN) | known decisions (WE/KB) | known->novel commits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| j1b | 0.7136 | 0.3409 | 0.1274 | 0.9529 | 0.1389 | 117 | 0.2203 | 118 (23/92/3) | 182 (179/3) | 268 |
| m1 | 0.7016 | 0.3614 | 0.1142 | 0.9608 | 0.1667 | 55 | 0.2072 | 111 (23/88/0) | 172 (172/0) | 250 |
| m3 | 0.6998 | 0.3634 | 0.1157 | 0.9575 | 0.1667 | 39 | 0.1700 | 100 (17/83/0) | 154 (153/1) | 220 |

Interpretation:

- The dev->held-out N2K "shift" is mostly gone (dev 0.106, held-out
  0.127) once GT is correct: `NO_CLEAR_ROUTING_SHIFT`.
- Deferral still halves prototypes (117->39/55) and slightly improves
  novel consistency, but **the held-out resolved-accuracy gain is not
  reproduced** (m1/m3 are below the j1b anchor on corrected GT).  The
  Phase 4M `DEFERRAL_GENERALIZED` claim is therefore weakened: it
  generalizes as a prototype-reduction / tracking-neutral mechanism,
  not as an accuracy gain.
- Known-class detections are absorbed into novel memory at scale
  (268/250/220 known->novel commits), a form of pollution invisible in
  the old collapsed GT.
""")

    # ---- generalization decision ----
    (DOC / "GENERALIZATION_DECISION.md").write_text(
        """# Phase 4N Generalization Decision

Status: `UPSTREAM_NOT_GENERALIZED`.

The only Phase 4N upstream candidate (N2, hard validity gate) failed the
dev pass gate (eventual resolution coverage collapsed from 0.956 to
0.619; FP reuse unchanged; USEFUL=0), so it was not frozen and not run
on held-out.

Phase 4M deferral re-evaluated on the corrected held-out GT: tracking
and prototype reduction generalize, but resolved-novel accuracy does not
improve over the anchor (m3 0.170 vs j1b 0.220).  The earlier
`DEFERRAL_GENERALIZED` is superseded by this corrected measurement.
""")

    # ---- method novelty audit ----
    (DOC / "PHASE4N_METHOD_NOVELTY_AUDIT.md").write_text(
        """# Phase 4N Method Novelty Audit

1. New frontend is not a simple detector threshold: thresholding is
   closed by the oracle curve (precision reaches 0.17 only when novel
   recall < 0.35); the attempted method is a validity gate on memory
   eligibility.
2. Object validity vs ordinary objectness: validity combines detector
   score, track age and mask fraction; ordinary objectness is a
   detector-internal score.
3. Tracking evidence contributes validity: D1 AUROC 0.79-0.80 and FP
   rejection 0.78-0.82 at novel recall 0.7 (vs 0.62-0.68 for score
   alone).
4. True-novel recall is protected in the audit (coverage 0.83 dev at the
   chosen operating point), but the online gate still collapsed
   resolution coverage -> the method failed its pass gate.
5. No KNOWN/NOVEL calibration was tuned: the corrected GT shows no
   routing shift, so Route A was not supported.
6. The dev->held-out N2K shift is explained as a GT artifact (all
   held-out categories collapsed to one id), not a gate calibration
   problem.
7. No online calibration was used (not needed).
8. Detector change is supported by the audit (NEW_DETECTOR_REQUIRED)
   but not executed: OWOBJ requires training task weights and an old
   torch/CUDA-ops environment; no detector was integrated.
9. Open-vocabulary detectors (YOLOE/OmDet) have no unknown-vs-background
   branch and would not steal the dynamic category-discovery task.
10. Causal deferral is preserved in N2 (validity gates memory
    eligibility; deferral still resolves identity), but coverage
    collapsed.
11. Semantics->Tracking: association utility stays net-negative
    (-11/-12) and unchanged.
12. Tracking->Semantics: tracking evidence improves validity
    predictability (D1 >= D0) but cannot fix the FP-dominated memory.
13. USEFUL population: still 0 (no milestone).
14. Dev + held-out: N2 was not run on held-out (failed dev); corrected
    held-out re-evaluation weakens Phase 4M's accuracy-gain claim.
15. ICLR strong-method strength: not reached; the phase contributes a
    benchmark correction and a confirmed detector-frontend bottleneck.
""")

    # ---- next research decision ----
    (DOC / "NEXT_RESEARCH_DECISION.md").write_text(
        """# Phase 4N Next Research Decision

1. **Correct the benchmark first**: all future phases must use the
   corrected held-out GT (28 categories) and re-run semantic evals.
2. **Detector frontend (Route C)**: audit supports NEW_DETECTOR_REQUIRED
   (novel-vs-FP AUPRC 0.045; impassable threshold Pareto).  First
   candidate OWOBJ (CVPR 2025); requires task weights / training and
   TAO conversion.  Detector-only pass gate before TrackOCD integration.
3. **Representation branch**: if a detector swap still leaves FP reuse
   ~0.96 and USEFUL=0, reopen representation research (DINO/M2 gate
   features).
4. **Memory purity metric**: provenance counts members regardless of
   validity weight; future validity designs must change entry, not only
   update weights, to move FP reuse share.
5. **Known->novel pollution**: corrected GT reveals 220-268 known tracks
   committing to novel memory per held-out run; a known-class admission
   guard is a concrete next mechanism.
""")

    # ---- final report ----
    r = []
    a = r.append
    a("# TrackOCD ICLR 2027 Phase 4N — Complete Copyable Report")
    a("")
    a("## 1. Execution Overview")
    a("")
    a("Wall time ~4h (audit builds, replays, evals).  GPUs 0-7 used "
      "sequentially/in parallel; Phase 4M leftover background jobs "
      "finished early in the phase.  Disk: /data1 41G free, /data3 139G "
      "free; no OOM.  Repairs: audit CSV schema fixes; corrected held-out "
      "GT rebuild.  Completed: frontend audit, gate audit, open-source "
      "review, N2 dev experiment (failed pass gate), corrected-GT "
      "re-evaluation of Phase 4M candidates.  Skipped: calibration "
      "branch (no shift), detector execution (weights/env unavailable).")
    a("")
    a("## 2. Frozen Phase 4M State")
    a("")
    a("M3 hybrid causal deferral (margin<0.10 or novel-known<0.25).  "
      "Dev: HOTA 0.1593, resolved-novel acc 0.2755, prototypes 40.  "
      "The Phase 4M held-out numbers in its report were computed on a "
      "category-collapsed GT (see erratum in HELDOUT_RESULTS.md).")
    a("")
    a("## 3. Why Phase 4N")
    a("")
    a("Phase 4M left FP reuse ~0.96 and USEFUL=0; the KNOWN/NOVEL gate "
      "appeared to shift (dev N2K 0.10 vs held-out 0.52).  Phase 4N "
      "decomposed this into object validity (Layer A), known/novel "
      "routing (Layer B) and identity resolution (Layer C).")
    a("")
    a("## 4. Three-Layer Error Decomposition")
    a("")
    a("Layer A: 95-96% of detections are FP (dev 1809/1898; held-out "
      "2110/2222).  Layer B: on valid detections, known/novel is well "
      "separated on both splits (AUROC 0.81-0.95).  Layer C: Phase 4M "
      "deferral is mechanism-supported on dev.")
    a("")
    a("## 5. Detection Population")
    a("")
    a("Dev: 52 VALID_KNOWN, 37 VALID_NOVEL, 1809 FP.  Corrected "
      "held-out: 74 VALID_KNOWN, 38 VALID_NOVEL, 2110 FP.")
    a("")
    a("## 6. Detector FP Audit")
    a("")
    a("Valid-vs-FP AUROC 0.784 dev / 0.774 held-out; novel-vs-FP AUPRC "
      "0.045 dev (precision is unattainable).")
    a("")
    a("## 7. True-Novel Recall vs FP Pareto")
    a("")
    a("No detector threshold reaches precision 0.2 while keeping novel "
      "recall >0.35 (dev).  Simple threshold filtering is closed.")
    a("")
    a("## 8. Persistent FP Audit")
    a("")
    a("Persistent FP tracklets dominate; their score/age distributions "
      "overlap valid objects (full table in PERSISTENT_FP_FRONTEND_AUDIT"
      ".md).")
    a("")
    a("## 9. Object Validity Predictability")
    a("")
    a("D0 0.784, D1 0.790, D2 0.592, D3 0.827, D4 0.861 (dev); "
      "held-out D4 0.848.  FP rejection at novel recall 0.7: D0 0.68, "
      "D1 0.82, D4 0.82 (dev).")
    a("")
    a("## 10. Frontend Root Decision")
    a("")
    a("`FRONTEND_VALIDITY_SIGNAL_PARTIAL`; `NEW_DETECTOR_REQUIRED`; "
      "`MULTI_SOURCE_VALIDITY_ADDS_VALUE`.")
    a("")
    a("## 11. Dev/Held-Out KNOWN/NOVEL Shift")
    a("")
    a("With corrected GT the shift disappears: routing 0.643->0.714, "
      "N2K 0.106->0.127 (dev->held-out).  The old 0.52 N2K was a GT "
      "artifact.")
    a("")
    a("## 12. Valid-only Gate Geometry")
    a("")
    a("Known-vs-novel AUROC: dev 0.81-0.87, held-out 0.90-0.95 across "
      "gate_logit / best_known / known_margin.")
    a("")
    a("## 13. Track-Age Shift")
    a("")
    a("The remaining routing gap is concentrated at early ages; "
      "long-track routing is stable (gate_shift_by_age.csv).")
    a("")
    a("## 14. Video-Level Shift")
    a("")
    a("Per-video N2K medians: dev ~0.03, held-out ~0.04 (IQR small); "
      "no single-video artifact explains a 0.5 N2K.")
    a("")
    a("## 15. Detector x Gate Coupling")
    a("")
    a("Low detector confidence does not systematically force NOVEL on "
      "valid objects; FP->NOVEL dominates at all score buckets "
      "(detector_gate_interaction.csv).")
    a("")
    a("## 16. Gate Root Decision")
    a("")
    a("`NO_CLEAR_ROUTING_SHIFT` on corrected GT.")
    a("")
    a("## 17. 2025-2026 GitHub Review")
    a("")
    a("Verified clones: OWOBJ (CVPR 2025), YOLO-UniOW (2024), OW-OVD "
      "(CVPR 2025), YOLOE (CVPR 2025), OmDet (2024-2026), DetSeg (ICCV "
      "2025); paper-only: DualMem (2026), OW-Rep (WACV 2026).  "
      "No directly usable detector for TrackOCD without training/weights "
      "or conversion.")
    a("")
    a("## 18. Detector Candidates")
    a("")
    a("OWOBJ first candidate; requires COCO-OWOD task weights/training "
      "and an old torch/CUDA-ops environment.  YOLOE/OmDet are "
      "open-vocabulary proposal frontends without unknown-vs-background "
      "branches.  No detector executed this phase.")
    a("")
    a("## 19. Robust Calibration Method")
    a("")
    a("Not supported (`NO_CLEAR_ROUTING_SHIFT`); skipped.")
    a("")
    a("## 20. Validity-Aware Routing")
    a("")
    a("Implemented N2 (M3 + dev-calibrated validity gate, threshold "
      "0.01).  Dev: HOTA 0.1596, IDSW 488, routing 0.6565, novel "
      "consistency 0.1923, prototypes 39, FP reuse 0.9651, but eventual "
      "resolution coverage 0.6189 and unresolved-at-termination 0.3811.  "
      "`VALIDITY_AWARE_ROUTING_NOT_SUPPORTED`.")
    a("")
    a("## 21. Component Development Results")
    a("")
    a("See outputs/iclr27_phase4n/dev/component_comparison.csv and "
      "DEVELOPMENT_RESULTS.md.")
    a("")
    a("## 22. Detector-Only Results")
    a("")
    a("Not executed (candidate weights/environment unavailable).")
    a("")
    a("## 23. TrackOCD Development Results")
    a("")
    a("N2 fails the pass gate; no candidate frozen.")
    a("")
    a("## 24. Frozen Candidates")
    a("")
    a("None from Phase 4N.")
    a("")
    a("## 25. Held-Out Protocol")
    a("")
    a("Corrected GT (28 categories) on the same 24 videos/887 frames; "
      "no held-out tuning.")
    a("")
    a("## 26. Held-Out Results")
    a("")
    a("Corrected re-evaluation of frozen Phase 4M candidates: routing "
      "0.70-0.71, N2K 0.11-0.13, novel consistency 0.139-0.167, "
      "prototypes 39-117.  Resolved-novel accuracy: j1b 0.220, m1 0.207, "
      "m3 0.170.")
    a("")
    a("## 27. Dev/Held-Out Gap After Method")
    a("")
    a("Gap is small with corrected GT (N2K 0.106 vs 0.127); no method "
      "was needed for routing.")
    a("")
    a("## 28. Detection Metrics")
    a("")
    a("FP semantic-entry rate: m3 0.898 vs n2 0.880 (dev); valid-novel "
      "semantic coverage 0.827 vs 0.801; valid-known coverage 0.669 vs "
      "0.676.")
    a("")
    a("## 29. Tracking Metrics")
    a("")
    a("N2 is tracking-neutral: HOTA 0.1596 (+0.0003), AssA 0.4587 "
      "(+0.002), IDSW 488 (-2).")
    a("")
    a("## 30. Semantic Metrics")
    a("")
    a("N2 improves routing (0.6565 vs 0.6471) and novel consistency "
      "(0.1923 vs 0.1154) on dev; N2K unchanged (~0.10).")
    a("")
    a("## 31. Causal Deferral Preservation")
    a("")
    a("N2 keeps M3's deferral rule; validity gates memory eligibility "
      "only.  Coverage collapse is a gate interaction, not a deferral "
      "change.")
    a("")
    a("## 32. Memory Provenance Re-Audit")
    a("")
    a("USEFUL=0 everywhere; POLLUTING 38-40 (dev) and 38-114 (corrected "
      "held-out); FP reuse 0.965 (dev) / 0.960-0.963 (held-out); birth "
      "source dominated by FP (36-110), plus 3-4 known births.")
    a("")
    a("## 33. Does Tracking Help Semantics?")
    a("")
    a("Yes for validity scoring (D1 >= D0), no for memory purity.")
    a("")
    a("## 34. Does Semantics Help Tracking?")
    a("")
    a("Association utility remains net-negative (-11/-12); no change.")
    a("")
    a("## 35. Error Transfer")
    a("")
    a("Validity gating transfers coverage loss (38% unresolved-at-term) "
      "without reducing FP reuse: error is transferred, not removed.")
    a("")
    a("## 36. Open-Source Comparison")
    a("")
    a("No directly compatible open-world detector was runnable without "
      "training; OWOBJ is the closest objectness-based candidate.")
    a("")
    a("## 37. Method Novelty")
    a("")
    a("Phase 4N's contribution is negative/audit: a benchmark GT "
      "correction and a confirmed detector bottleneck, plus the N2 "
      "failure analysis (validity gates transfer coverage instead of "
      "reducing pollution).")
    a("")
    a("## 38. ICLR Readiness")
    a("")
    a("Task: open-world protocol has a corrected held-out split.  "
      "Protocol: frozen dev/held-out discipline maintained.  Detection "
      "frontend: bottleneck confirmed; new detector required.  Object "
      "validity: partial signal, post-hoc gate fails.  Known/novel "
      "routing: no shift once GT corrected.  Association: unchanged.  "
      "Causal resolution: supported on dev, accuracy gain not reproduced "
      "on corrected held-out.  Memory: still FP-dominated.  Novelty: "
      "benchmark correction + bottleneck evidence.")
    a("")
    a("## 39. Final Status")
    a("")
    a("`DETECTION_FRONTEND_BOTTLENECK_CONFIRMED`; "
      "`MEMORY_STILL_FP_DOMINATED`; `UPSTREAM_FRONTEND_NOT_SUPPORTED`; "
      "`REPRESENTATION_RESEARCH_REQUIRED` as the fallback if detector "
      "swap fails.  Phase 4M's deferred-held-out accuracy-gain claim is "
      "superseded by the corrected GT.")
    a("")
    a("## 40. Next Steps (max 5)")
    a("")
    a("1. Detector: OWOBJ detector-only benchmark (weights/training + "
      "TAO conversion); continue if novel recall preserved with FP cut; "
      "stop if no detector-only gain.")
    a("2. Known->novel admission guard on corrected GT (220-268 known "
      "commits per run); continue if it reduces POLLUTING; stop if "
      "coverage drops.")
    a("3. Re-run Phase 4M candidates' identity audits on corrected GT "
      "and re-freeze deferral claims.")
    a("4. Representation audit of M2 gate on novel-vs-known geometry if "
      "detector swap fails.")
    a("5. Finalize benchmark correction in the project docs (GT source, "
      "category protocol).")
    a("")
    (DOC / "PHASE4N_COMPLETE_COPYABLE_REPORT.md").write_text("\n".join(r))
    print("RESULTS_DOCS_DONE")


if __name__ == "__main__":
    main()
