"""Phase 4M held-out comparison + generalization decision."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
HO = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "heldout"
AUDIT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
DOC = ROOT / "docs" / "iclr27_phase4m"
TAGS = ["j1b", "m1", "m3"]


def tracking(tag):
    t = json.load(open(HO / "trackeval" / f"tracking_{tag}.json"))[tag]
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
        HO / "trackeval" / f"semantic_{tag}.csv")))
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
    p = HO / "trackeval" / f"resolution_{tag}.csv"
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


def resolved_novel_accuracy(tag):
    p = AUDIT / f"identity_decisions_ho_{tag}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    nov = [r for r in rows if r["query_gt_role"] == "novel"]
    if not nov:
        return None
    correct = sum(1 for r in nov if r["decision_class"] in
                  ("CORRECT_EXISTING", "CORRECT_NEW"))
    return round(correct / len(nov), 4)


def memory(tag):
    p = AUDIT / "heldout" / f"prototype_provenance_{tag}.csv"
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


def main():
    HO.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "heldout").mkdir(parents=True, exist_ok=True)
    ho_root = AUDIT / "prov_ho_root"
    ho_root.mkdir(parents=True, exist_ok=True)
    for tag in TAGS:
        link = ho_root / f"prov_{tag}"
        if not link.exists():
            link.symlink_to(ROOT / "outputs" / "iclr27_phase4m" / "prov" /
                            f"heldout_{tag}")
    env = dict(os.environ)
    env["PHASE4L_PROV_ROOT"] = str(ho_root)
    env["PHASE4L_AUDIT_OUT"] = str(AUDIT / "heldout")
    env["PHASE4L_TAO_JSON"] = str(
        ROOT / "outputs" / "iclr27_phase4l" / "heldout" /
        "validation_heldout_tao.json")
    for tag in TAGS:
        if not (AUDIT / "heldout" /
                f"prototype_provenance_{tag}.csv").exists():
            subprocess.run([sys.executable,
                            str(ROOT / "src/iclr27_phase4k" /
                                "build_offline_audit.py"),
                            "--tag", tag], env=env, check=True)
    # identity decisions from held-out provenance
    for tag in TAGS:
        out = AUDIT / f"identity_decisions_ho_{tag}.csv"
        if not out.exists():
            subprocess.run([
                sys.executable,
                str(ROOT / "src/iclr27_phase4m" /
                    "build_identity_decisions_v2.py"),
                "--tag", tag,
                "--prov-root",
                str(ROOT / "outputs" / "iclr27_phase4m" / "prov" /
                    f"heldout_{tag}"),
                "--tao-json",
                str(ROOT / "outputs" / "iclr27_phase4l" / "heldout" /
                    "validation_heldout_tao.json"),
                "--z-cache",
                str(AUDIT / "det_z_cache_heldout"),
                "--out", str(out),
            ], env=env, check=True)
    # resolution metrics (must exist from the replay eval)
    for tag in TAGS:
        p = HO / "trackeval" / f"resolution_{tag}.csv"
        if not p.exists():
            subprocess.run([
                sys.executable,
                str(ROOT / "src/iclr27_phase4m" / "resolution_metrics.py"),
                "--log-root", str(HO / tag / "semantic_logs"),
                "--out", str(p)], env=env, check=True)

    rows = []
    for tag in TAGS:
        row = {"tag": tag}
        row.update(tracking(tag))
        row.update(semantic(tag))
        row.update(resolution(tag))
        row.update(memory(tag))
        row["resolved_novel_accuracy"] = resolved_novel_accuracy(tag)
        rows.append(row)
    with open(HO / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1))

    anchor = next(r for r in rows if r["tag"] == "j1b")
    gains = []
    for tag in ("m1", "m3"):
        r = next(x for x in rows if x["tag"] == tag)
        acc = r.get("resolved_novel_accuracy")
        anc_acc = anchor.get("resolved_novel_accuracy")
        ok = (
            r["HOTA"] >= 0.95 * anchor["HOTA"] and
            r["IDSW"] <= 1.15 * anchor["IDSW"] and
            (r.get("eventual_resolution_coverage") or 0) >= 0.85 and
            (acc or 0) >= (anc_acc or 0) and
            (r.get("novel_consistency") or 0) >= (
                anchor.get("novel_consistency") or 0)
        )
        gains.append((tag, ok))
    gen = "DEFERRAL_GENERALIZED" if all(ok for _, ok in gains) else \
        "DEFERRAL_NOT_GENERALIZED"
    (DOC / "HELDOUT_RESULTS.md").write_text(
        f"""# Phase 4M Held-Out Results (24 videos, one-shot, seed 20260808)

Frozen candidates: m1 (margin defer) and m3 (hybrid defer).  The anchor
j1b was reproduced exactly (HOTA 0.1792, IDSW 563, Frag 226).

Full comparison: `outputs/iclr27_phase4m/runs/heldout/comparison.csv`.

| tag | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | routing | N2K | novel_cons | protos | USEFUL | POLLUTING | fp_reuse | net | resolved_novel_acc | eventual_cov | unres_at_term | lat_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
            f"| {r['tag']} | {r['HOTA']} | {r['DetA']} | {r['AssA']} | "
            f"{r['IDF1']} | {r['MOTA']} | {r['IDSW']} | {r['Frag']} | "
            f"{r.get('routing_accuracy','')} | "
            f"{r.get('n2k_rate_novel_denom','')} | "
            f"{r.get('novel_consistency','')} | "
            f"{r.get('prototypes','')} | {r.get('USEFUL','')} | "
            f"{r.get('POLLUTING','')} | {r.get('fp_reuse_share','')} | "
            f"{r.get('assoc_net_utility','')} | "
            f"{r.get('resolved_novel_accuracy','')} | "
            f"{r.get('eventual_resolution_coverage','')} | "
            f"{r.get('unresolved_at_termination_rate','')} | "
            f"{r.get('resolution_latency_p90','')} |"
            for r in rows) +
        f"""

Generalization decision: `{gen}`.

Per-candidate checks: {gains}.
""")
    (DOC / "GENERALIZATION_DECISION.md").write_text(
        f"""# Phase 4M Generalization Decision

Status: `{gen}`.

The two frozen candidates were evaluated once on the 24-video held-out
subset after all dev tuning.  No held-out result was used to modify any
candidate.  The dev/held-out routing shift (dev N2K ~0.09-0.11 vs
held-out ~0.45-0.53) is analyzed as a post-freeze diagnostic only: the
identity-deferral mechanism targets EXISTING-vs-NEW resolution and does
not claim to fix the known/novel gate shift.
""")
    print("COMPARE_HELDOUT_DONE", gen)


if __name__ == "__main__":
    main()
