"""Phase 4M dev comparison: M0 anchor vs M1/M2/M3."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
DEV = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev"
AUDIT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
DOC = ROOT / "docs" / "iclr27_phase4m"
TAGS = ["j1b", "m1", "m2", "m3"]


def tracking(tag):
    t = json.load(open(DEV / "trackeval" / f"tracking_{tag}.json"))[tag]
    return {
        "HOTA": round(float(t["HOTA"]["HOTA(0)"]), 4),
        "DetA": round(float(t["HOTA"]["DetA"][0]), 4),
        "AssA": round(float(t["HOTA"]["AssA"][0]), 4),
        "IDF1": round(float(t["Identity"]["IDF1"]), 4),
        "MOTA": round(float(t["CLEAR"]["MOTA"]), 4),
        "IDSW": int(t["CLEAR"]["IDSW"]),
        "Frag": int(t["CLEAR"]["Frag"]),
    }


def semantic(tag):
    rows = list(csv.DictReader(open(
        DEV / "trackeval" / f"semantic_{tag}.csv")))
    if not rows:
        return {}
    r = rows[0]
    out = {}
    for k in ("routing_accuracy", "k2n_rate_known_denom",
              "n2k_rate_novel_denom", "known_class_accuracy",
              "semantic_id_switch_rate", "novel_consistency",
              "commit_coverage_novel", "commit_coverage_fp",
              "fp_global_memory_admission_rate", "global_novel_memory_size",
              "novel_ids_created", "gt_novel_reuse_tracks",
              "known_tracks_committed_to_novel"):
        try:
            out[k] = round(float(r[k]), 4) if r[k] not in ("", "None") \
                else ""
        except (KeyError, ValueError):
            pass
    return out


def resolution(tag):
    p = DEV / "trackeval" / f"resolution_{tag}.csv"
    if not p.exists():
        return {}
    r = next(csv.DictReader(open(p)))
    out = {}
    for k in ("deferral_rate_frames", "deferral_rate_tracks",
              "immediate_resolution_coverage",
              "eventual_resolution_coverage",
              "unresolved_at_termination_rate",
              "resolution_latency_mean", "resolution_latency_median",
              "resolution_latency_p90"):
        if r.get(k) not in ("", None):
            try:
                out[k] = round(float(r[k]), 4)
            except ValueError:
                out[k] = r[k]
    out["resolved_action_mix"] = r.get("resolved_action_mix", "")
    return out


def memory(tag):
    p = AUDIT / f"prototype_provenance_{tag}.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(open(p)))
    from collections import Counter
    grp = Counter(r["outcome_group"] for r in rows)
    n_reuse = sum(int(r["n_reuses"]) for r in rows)
    fp_abs = sum(int(r["fp_absorptions"]) for r in rows)
    helpful = sum(int(r["assoc_helpful"]) for r in rows)
    harmful = sum(int(r["assoc_harmful"]) for r in rows)
    return {
        "prototypes": len(rows),
        "USEFUL": grp.get("USEFUL", 0),
        "POLLUTING": grp.get("POLLUTING", 0),
        "LOW_EVIDENCE": grp.get("LOW_EVIDENCE", 0),
        "fp_reuse_share": round(fp_abs / max(n_reuse, 1), 4),
        "assoc_net_utility": helpful - harmful,
    }


def resolved_novel_accuracy(tag):
    p = AUDIT / f"identity_decisions_v2_{tag}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    nov = [r for r in rows if r["query_gt_role"] == "novel"]
    if not nov:
        return None
    correct = sum(1 for r in nov if r["decision_class"] in
                  ("CORRECT_EXISTING", "CORRECT_NEW"))
    return round(correct / len(nov), 4)


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    DEV.mkdir(parents=True, exist_ok=True)
    # ensure provenance audits exist for M tags
    env = dict(os.environ)
    env["PHASE4L_PROV_ROOT"] = str(
        ROOT / "outputs" / "iclr27_phase4m" / "prov")
    env["PHASE4L_AUDIT_OUT"] = str(AUDIT)
    for tag in TAGS:
        if not (AUDIT / f"prototype_provenance_{tag}.csv").exists():
            subprocess.run([sys.executable,
                            str(ROOT / "src/iclr27_phase4k" /
                                "build_offline_audit.py"),
                            "--tag", tag], env=env, check=True)
    rows = []
    for tag in TAGS:
        row = {"tag": tag}
        row.update(tracking(tag))
        row.update(semantic(tag))
        row.update(resolution(tag))
        row.update(memory(tag))
        row["resolved_novel_accuracy"] = resolved_novel_accuracy(tag)
        rows.append(row)
    with open(DEV / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1))

    # Freeze logic: at most 2 candidates; require eventual coverage >= .85,
    # tracking HOTA within 5% of anchor and IDSW within 15%.  Rank by
    # resolved-novel accuracy (primary), then fewer prototypes, then HOTA.
    anchor = next(r for r in rows if r["tag"] == "j1b")
    cands = []
    for r in rows:
        if r["tag"] == "j1b":
            continue
        cov = r.get("eventual_resolution_coverage")
        if cov is None or cov < 0.85:
            continue
        if r["HOTA"] < 0.95 * anchor["HOTA"]:
            continue
        if r["IDSW"] > 1.15 * anchor["IDSW"]:
            continue
        if r.get("prototypes", 10 ** 9) < anchor.get("prototypes", 0) or \
                (r.get("resolved_novel_accuracy") or 0) > (
                    anchor.get("resolved_novel_accuracy") or 0):
            cands.append(r["tag"])
    def row_of(t):
        return next(r for r in rows if r["tag"] == t)

    cands.sort(key=lambda t: (
        -float(row_of(t)["resolved_novel_accuracy"] or 0),
        row_of(t).get("prototypes", 10 ** 9),
        -float(row_of(t)["HOTA"])))
    cands = cands[:2]
    frozen = " ".join(cands) if cands else "NONE"
    status = "CAUSAL_DEFERRAL_PARTIAL_PROGRESS" if cands else \
        "CAUSAL_DEFERRAL_NOT_SUPPORTED"
    (DOC / "DEVELOPMENT_RESULTS.md").write_text(
        f"""# Phase 4M Development Results (20-video dev)

FROZEN_CANDIDATES={frozen}
DEV_STATUS={status}
HELDOUT_NEEDED_YES
m1=margin
m3=hybrid

Full comparison: `outputs/iclr27_phase4m/runs/dev/comparison.csv`.

Summary of the comparison rows (all numbers from that CSV):

| tag | HOTA | AssA | IDSW | routing | N2K | novel_consistency | prototypes | defer_tracks | eventual_cov | unresolved_at_term |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
            f"| {r['tag']} | {r['HOTA']} | {r['AssA']} | {r['IDSW']} | "
            f"{r.get('routing_accuracy','')} | "
            f"{r.get('n2k_rate_novel_denom','')} | "
            f"{r.get('novel_consistency','')} | "
            f"{r.get('prototypes','')} | "
            f"{r.get('deferral_rate_tracks','')} | "
            f"{r.get('eventual_resolution_coverage','')} | "
            f"{r.get('unresolved_at_termination_rate','')} |"
            for r in rows) +
        f"""

Freeze decision: `{frozen}` (at most two, only if eventual coverage >=
0.85, HOTA within 5% of anchor, IDSW within 15%, and fewer prototypes /
novel ids).
""")
    print("COMPARE_DEV_DONE", frozen, status)


if __name__ == "__main__":
    main()
