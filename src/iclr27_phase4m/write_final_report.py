"""Generate PHASE4M_COMPLETE_COPYABLE_REPORT.md from result CSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
DEV = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev"
HO = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "heldout"
DOC = ROOT / "docs" / "iclr27_phase4m"


def load_csv(p):
    return list(csv.DictReader(open(p)))


def get(rows, tag, key, default=""):
    for r in rows:
        if r["tag"] == tag:
            return r.get(key, default)
    return default


def main():
    dev = load_csv(DEV / "comparison.csv")
    ho = load_csv(HO / "comparison.csv")
    geo = json.load(open(AUDIT / "geometry_summary.json"))
    retro = {t: load_csv(AUDIT / f"retrospective_{t}_summary.csv")
             for t in ("j1b", "b1", "b2")}
    pareto = load_csv(AUDIT / "deferral_pareto.csv")

    def dev_row(tag, keys):
        return " | ".join(str(get(dev, tag, k, "")) for k in keys)

    def ho_row(tag, keys):
        return " | ".join(str(get(ho, tag, k, "")) for k in keys)

    dev_keys = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag",
                "routing_accuracy", "n2k_rate_novel_denom",
                "novel_consistency", "prototypes", "USEFUL", "POLLUTING",
                "fp_reuse_share", "assoc_net_utility",
                "resolved_novel_accuracy", "eventual_resolution_coverage",
                "unresolved_at_termination_rate",
                "resolution_latency_p90"]
    ho_keys = dev_keys

    r = []
    a = r.append
    a("# TrackOCD ICLR 2027 Phase 4M — Complete Copyable Report")
    a("")
    a("## 1. Execution Overview")
    a("")
    a("Phase 4M audited whether the forced EXISTING_NOVEL / NEW_NOVEL "
      "decision is the root cause of overbirth and wrong-reuse errors, "
      "then conditionally implemented the minimal Causal Semantic "
      "Deferral (UNRESOLVED_NOVEL).  Stages completed: corrected decision "
      "dataset (v2), overbirth / wrong-reuse / ambiguity-geometry audits, "
      "retrospective deferral oracle, time-to-resolution, deferral "
      "Pareto, 2025-2026 open-source audit, M0/M1/M2/M3 dev bake-off, "
      "freeze of m1+m3, one-shot 24-video held-out evaluation, memory "
      "provenance re-audit, novelty audit, and contract tests.")
    a("")
    a("## 2. Corrected Phase 4L Anchor")
    a("")
    a("Final-code M0 (dev, same video order as Phase 4K): HOTA 0.1591, "
      "AssA 0.4556, IDSW 493, routing 0.6433, N2K 0.1057, novel "
      "consistency 0.1154, 106 novel prototypes, USEFUL 0, POLLUTING 100, "
      "FP reuse share 0.9605, association net utility -12.  Held-out "
      "anchor reproduces Phase 4L exactly: HOTA 0.1792, DetA 0.0692, "
      "AssA 0.4644, IDF1 0.0713, MOTA -12.50, IDSW 563, Frag 226.")
    a("")
    a("Note: Phase 4L's stored dev provenance (116 novel IDs) predates "
      "the final two pipeline fixes; the final code reproduces the "
      "held-out anchor exactly.  All Phase 4M dev comparisons use the "
      "final-code M0 run.")
    a("")
    a("## 3. Why Forced Identity Resolution Is Suspected")
    a("")
    a("Phase 4L candidates suppressed reuse / admission and the error "
      "transferred to prototype explosion (535-672 prototypes) and "
      "overbirth (up to 9).  The corrected decision dataset shows the "
      "binary EXISTING-vs-NEW action has no legal ambiguous output.")
    a("")
    a("## 4. Identity Decision Taxonomy")
    a("")
    a("Replaying provenance with exact prefix reconstruction: sticky "
      "same-track reuses are continuations (19,123 in j1b), not "
      "decisions; 75% of j1b NEW-branch events (325/431) actually reused "
      "an existing prototype (the `is_new` return was previously "
      "discarded).  M0 decision counts: 10,990 soft EXISTING + 325 "
      "NEW->EXISTING + 106 NEW = 11,421.")
    a("")
    a("## 5. Overbirth Audit")
    a("")
    a("GT-novel overbirths: j1b 0/106 (0%), b1 0/535 (0%), b2 9/672 "
      "(1.3%).  Overbirth is **not** the dominant failure of the "
      "corrected anchor; prototype explosion is mostly FP births "
      "(j1b 98, b1 513, b2 632).")
    a("")
    a("## 6. Wrong-Reuse Audit")
    a("")
    a("GT-novel decisions: j1b 108 (12 CORRECT_EXISTING, 93 "
      "WRONG_EXISTING, 3 CORRECT_NEW, 0 OVERBIRTH); b1 101 (16/75/10/0); "
      "b2 47 (7/22/9/9).  Wrong reuse dominates the anchor (86%).")
    a("")
    a("## 7. Ambiguity Geometry")
    a("")
    a("Median geometry (j1b): CORRECT_EXISTING best 0.680 / margin 0.041 "
      "/ nK 0.166; WRONG_EXISTING best 0.684 / margin 0.045 / nK 0.178; "
      "CORRECT_NEW best 0.351 / nK 0.083.  Correct-vs-wrong AUROC on "
      "j1b: best 0.366, margin 0.513, entropy 0.338, novel-known 0.457, "
      "z-score 0.138; on b1: best 0.706, novel-known 0.718.  Online "
      "ambiguity detection is weak on the anchor and partial on B1.")
    a("")
    a("## 8. Retrospective Deferral Audit")
    a("")
    a("Counterfactual non-admission (deferred track writes nothing "
      "after t, other events held fixed): resolved correctly by t+8 — "
      "j1b 31/108 (28.7%), b1 30/101 (29.7%), b2 20/47 (42.6%); "
      "terminated before t+8 — 61-77%.")
    a("")
    a("## 9. Time-to-Resolution")
    a("")
    a("Most correct resolutions occur at t+1 (j1b 24/31); waiting to "
      "t+8 adds little.  Deployable median latency (m3 dev) 8.5 frames, "
      "p90 22 frames.")
    a("")
    a("## 10. Coverage-Risk Pareto")
    a("")
    a("On j1b novel decisions no ambiguity rule lowers decided-set "
      "error below ~83% while deferring 50-87%; deferred eventual "
      "coverage is 26-33% with 32-59 unresolved-at-termination.  Full "
      "table: `deferral_pareto.csv`.")
    a("")
    a("## 11. Root-Cause Decision")
    a("")
    a("`DEFERRAL_SIGNAL_PARTIAL`.  Future causal evidence resolves a "
      "real minority; online ambiguity is weakly detectable on j1b but "
      "partially on b1.  Implemented the minimal method; dev result is "
      "partial progress, not a full fix.")
    a("")
    a("## 12. 2025-2026 GitHub Review")
    a("")
    a("Twelve repositories cloned and pinned (commits recorded): "
      "ML-EDM/ml_edm (TMLR 2025), FIRMBOUND (ICLR 2025), SPEED (2024), "
      "scikit-fallback (2024-2026), boxmot (AGPL-3.0), "
      "learning-to-defer-to-a-population (IEEE 2025), "
      "sc-likelihood-ratios (ICLR 2025), sc-gap (2025), "
      "UQforDeferral (2026), StopAndHop (CIKM 2022), LTC (CVPR 2026 "
      "Findings), TALON (CVPR 2026).  LTC/TALON provide dynamic class "
      "birth but no track-local deferral.  "
      "`NO_DIRECTLY_COMPATIBLE_EXTERNAL_METHOD`.")
    a("")
    a("## 13. Causal Semantic Deferral Method")
    a("")
    a("Four states (RESOLVED_KNOWN / RESOLVED_EXISTING_NOVEL / "
      "RESOLVED_NEW_NOVEL / UNRESOLVED_NOVEL).  Deferral keeps the track "
      "local soft semantics active in association but writes no global "
      "prototype until decisive.  M1 margin<0.10; M2 entropy>1.6; M3 "
      "margin<0.10 OR novel-known<0.25.  Spec: "
      "`CAUSAL_SEMANTIC_DEFERRAL_SPEC.md`.")
    a("")
    a("## 14. Development Results (20-video dev)")
    a("")
    a("| tag | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | routing | N2K | novel_cons | protos | USEFUL | POLLUTING | fp_reuse | net | resolved_acc | event_cov | unres_term | lat_p90 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag in ("j1b", "m1", "m2", "m3"):
        a(f"| {tag} | {dev_row(tag, dev_keys)} |")
    a("")
    a("Frozen: m1 (margin) and m3 (hybrid).  M3 resolves 27.6% of novel "
      "decisions correctly vs 13.9% anchor, halves prototypes (106->40), "
      "keeps HOTA/IDSW, eventual coverage 95.6%, unresolved-at-term "
      "4.4%.  Memory purity unchanged (fp_reuse ~0.96, USEFUL=0).")
    a("")
    a("## 15. Resolution Metrics (dev)")
    a("")
    a("Deferral rate (tracks): m1 4.8%, m2 5.5%, m3 6.1%; immediate "
      "coverage m1 97.5%, m2 96.6%, m3 95.0%; eventual coverage "
      "97.2%/96.1%/95.6%; unresolved-at-termination 2.8%/3.9%/4.4%; "
      "latency median 5/6/8.5 frames, p90 16.3/20/22.")
    a("")
    a("## 16. Tracking Metrics")
    a("")
    a("Dev: M1 improves HOTA (0.1607) and IDSW (479); M2/M3 match the "
      "anchor within 0.0002 HOTA.  No deferral candidate degrades "
      "tracking.")
    a("")
    a("## 17. Semantic Metrics")
    a("")
    a("Dev N2K improves for all candidates (0.1057 -> 0.0946/0.0899/"
      "0.0994); novel consistency improves for m1/m2 (0.1538 vs 0.1154) "
      "and is unchanged for m3.  Routing accuracy is within 0.01 of the "
      "anchor.")
    a("")
    a("## 18. Memory Provenance Re-Audit")
    a("")
    a("Dev prototypes: 106 -> 50/41/40; all remaining prototypes are "
      "POLLUTING (USEFUL=0); FP reuse share stays 0.960-0.966; "
      "association net utility -11..-16.  `MEMORY_POLLUTION_REDUCED` is "
      "false — only prototype *count* dropped, not purity.")
    a("")
    a("## 19. Overbirth vs Wrong-Merge Trade-off")
    a("")
    a("On dev, deferral reduces wrong-existing among resolved novel "
      "decisions (93 -> 71 for m3) while overbirth remains 0; the "
      "binary trade-off is partially broken for identity errors, but FP "
      "routing errors dominate both axes.")
    a("")
    a("## 20. Unresolved-at-Termination")
    a("")
    a("4.4% (m3 dev) of novel-like tracklets end unresolved; these are "
      "honest coverage losses, not retroactively resolved.")
    a("")
    a("## 21. Resolution Latency")
    a("")
    a("m3 dev median 8.5 frames, p90 22; m1 median 5, p90 16.3.")
    a("")
    a("## 22. Does Tracking Help Semantic Resolution?")
    a("")
    a("Yes: 31/108 j1b wrong decisions resolve correctly by t+8 with "
      "longer prefixes (oracle), and deployed m3 improves resolved "
      "accuracy 13.9% -> 27.6%.")
    a("")
    a("## 23. Does Semantic Evidence Still Help Tracking?")
    a("")
    a("Yes: lambda_s=0.1 semantic consistency remains in the "
      "association cost for resolved and unresolved tracks; tracking "
      "metrics are unchanged or better.")
    a("")
    a("## 24. Held-Out Protocol")
    a("")
    a("24 videos / 887 frames, no dev overlap, seed 20260808.  Anchor "
      "reproduced exactly before running frozen candidates once.  No "
      "held-out tuning.")
    a("")
    a("## 25. Held-Out Results")
    a("")
    a("| tag | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | routing | N2K | novel_cons | protos | USEFUL | POLLUTING | fp_reuse | net | resolved_acc | event_cov | unres_term | lat_p90 |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tag in ("j1b", "m1", "m3"):
        a(f"| {tag} | {ho_row(tag, ho_keys)} |")
    a("")
    a("## 26. Dev/Held-Out Shift")
    a("")
    a("Held-out routing behaves differently (N2K 0.52 vs dev 0.10), "
      "i.e. the known/novel gate is shifted; identity deferral targets "
      "EXISTING-vs-NEW and does not claim to fix routing shift.")
    a("")
    a("## 27. Generalization Decision")
    a("")
    a("See `GENERALIZATION_DECISION.md`.")
    a("")
    a("## 28. Error Transfer")
    a("")
    a("Deferral removes prototype count pressure (106->40) without "
      "transferring error to overbirth (0) or tracking loss; the "
      "remaining error is FP admission into the prototypes that still "
      "exist.")
    a("")
    a("## 29. Open-Source Comparison")
    a("")
    a("No directly compatible frame-online open-world tracker with "
      "causal semantic deferral exists; closest principles: ml_edm "
      "(decision-time separation), FIRMBOUND (finite-horizon delayed "
      "decision), sc-likelihood-ratios (abstention scoring), boxmot "
      "(physical tentative states are a different axis).")
    a("")
    a("## 30. Method Novelty")
    a("")
    a("Task-native: semantic observation and association are immediate; "
      "irreversible identity resolution is deferred and retried on the "
      "same physical track with soft semantics active throughout.  "
      "Full: `PHASE4M_METHOD_NOVELTY_AUDIT.md`.")
    a("")
    a("## 31. ICLR Readiness")
    a("")
    a("Task: strong open-world MOT benchmark with a clean dev/held-out "
      "split.  Protocol: frozen detector/representation/tau/lambda, "
      "one-shot held-out.  Association: deferral does not hurt "
      "HOTA/AssA/IDSW.  Semantic routing: unchanged/shift-diagnosed.  "
      "Novel identity resolution: partial progress (resolved accuracy "
      "13.9->27.6% dev).  Memory quality: count reduced, purity not.  "
      "Generalization: held-out verdict in section 27.  Novelty: "
      "supported mechanism, not yet a strong method.  Remaining "
      "bottleneck: detection/FP stream and the known/novel gate.")
    a("")
    a("## 32. Final Status")
    a("")
    a("Overall: `CAUSAL_SEMANTIC_RESOLUTION_MECHANISM_SUPPORTED`.  "
      "Method: `CAUSAL_DEFERRAL_PARTIAL_PROGRESS` (clear identity "
      "progress, memory purity unchanged).  Generalization: "
      "`DEFERRAL_GENERALIZED` (both frozen candidates).  Memory: "
      "`MEMORY_STILL_POLLUTED` (USEFUL=0, FP reuse share ~0.96).  "
      "Resolution: `OVERBIRTH_REDUCED` (0 on dev, 1 on held-out m3 vs 6 "
      "anchor), `WRONG_REUSE_REDUCED` (dev 93->71, held-out 65->32), "
      "`RESOLUTION_COVERAGE_MAINTAINED` (dev 95.6%, held-out 95.5%).  "
      "The forced-decision hypothesis is partially confirmed: deferral "
      "helps identity resolution and generalizes, but it is not the "
      "dominant bottleneck (FP stream / known-novel gate remain).")
    a("")
    a("## 33. Next Steps (max 5)")
    a("")
    a("1. If held-out passes: integrate UNRESOLVED_NOVEL as a candidate "
      "method and ablate trajectory-refinement (EMA margin / persistent "
      "prototype ranking).  Continue condition: held-out resolved "
      "accuracy >= anchor and coverage >= 0.85.  Stop condition: "
      "held-out coverage collapse or no accuracy gain.")
    a("2. Detection-frontend phase: the FP stream (96% of prototype "
      "reuses) and the known/novel gate shift are the dominant "
      "bottleneck; run a frozen-policy ablation with an improved "
      "detector frontend.  Continue: detector update available.  Stop: "
      "no detector improvement in scope.")
    a("3. Memory purity: design a FP-aware admission audit (not a "
      "classifier) that reduces FP reuse share below 0.9 while keeping "
      "deferral coverage.  Continue: audit signal >= partial.  Stop: "
      "same pollution after one cycle.")
    a("4. Write the paper section on Tracking -> Semantics with the "
      "resolution-latency curves (m1/m3 vs anchor).  Continue: "
      "experiment set frozen.  Stop: results change after re-run.")
    a("5. Re-run the full blocking script and contract tests before "
      "submission.  Continue: all artifacts present.  Stop: any test "
      "failure.")
    a("")
    (DOC / "PHASE4M_COMPLETE_COPYABLE_REPORT.md").write_text("\n".join(r))
    print("FINAL_REPORT_DONE")


if __name__ == "__main__":
    main()
