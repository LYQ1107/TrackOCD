"""Phase 4N dev component comparison (N0=M3 anchor vs N2 validity)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
DEV = ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev"
AUDIT = ROOT / "outputs" / "iclr27_phase4n" / "audit"
DOC = ROOT / "docs" / "iclr27_phase4n"
OUT = ROOT / "outputs" / "iclr27_phase4n" / "dev"
TAGS = ["m3", "n2"]


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
              "novel_consistency", "commit_coverage_novel",
              "commit_coverage_fp", "fp_global_memory_admission_rate",
              "global_novel_memory_size", "novel_ids_created",
              "gt_novel_reuse_tracks", "known_tracks_committed_to_novel"):
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
              "resolution_latency_median", "resolution_latency_p90"):
        if r.get(k) not in ("", None):
            try:
                out[k] = round(float(r[k]), 4)
            except ValueError:
                out[k] = r[k]
    return out


def memory(tag):
    p = AUDIT / f"prototype_provenance_{tag}.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(open(p)))
    grp = Counter(r["outcome_group"] for r in rows)
    n_reuse = sum(int(r["n_reuses"]) for r in rows)
    fp_abs = sum(int(r["fp_absorptions"]) for r in rows)
    novel_abs = sum(int(r["novel_absorptions"]) for r in rows)
    helpful = sum(int(r["assoc_helpful"]) for r in rows)
    harmful = sum(int(r["assoc_harmful"]) for r in rows)
    return {
        "prototypes": len(rows),
        "USEFUL": grp.get("USEFUL", 0),
        "POLLUTING": grp.get("POLLUTING", 0),
        "LOW_EVIDENCE": grp.get("LOW_EVIDENCE", 0),
        "fp_reuse_share": round(fp_abs / max(n_reuse, 1), 4),
        "novel_reuse_share": round(novel_abs / max(n_reuse, 1), 4),
        "assoc_net_utility": helpful - harmful,
    }


def detection(tag):
    """FP semantic-entry and valid-novel coverage from semantic logs."""
    log_root = DEV / tag / "semantic_logs"
    gt = json.load(open(
        ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" /
        "validation_20.json"))
    known = set(json.loads((ROOT / "data" / "trackocd_v1" / "pure" /
                            "splits" /
                            "supported_known_ids.json").read_text()))
    gt_by_img = {}
    for a in gt["annotations"]:
        if a.get("iscrowd"):
            continue
        gt_by_img.setdefault(a["image_id"], []).append(a)

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
        return inter / ua if ua > 0 else 0.0

    stats = Counter()
    for p in sorted(log_root.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            best, bi = None, 0.5
            for g in gt_by_img.get(int(r["image_id"]), []):
                b = g["bbox"]
                v = iou(r["bbox"], [b[0], b[1], b[0] + b[2], b[1] + b[3]])
                if v >= bi:
                    bi, best = v, g
            role = "fp"
            if best is not None:
                role = "known" if int(best["category_id"]) in known \
                    else "novel"
            novel_like = r["p_known"] < r.get("decision_threshold", 0.3)
            entered = r.get("global_novel_id") is not None
            stats[(role, "det")] += 1
            if novel_like:
                stats[(role, "novel_like")] += 1
                if entered:
                    stats[(role, "sem_entry")] += 1
    return {
        "valid_known_dets": stats[("known", "det")],
        "valid_novel_dets": stats[("novel", "det")],
        "fp_dets": stats[("fp", "det")],
        "fp_semantic_entry_rate": round(
            stats[("fp", "sem_entry")] /
            max(stats[("fp", "novel_like")], 1), 4),
        "valid_novel_semantic_entry_cov": round(
            stats[("novel", "sem_entry")] /
            max(stats[("novel", "novel_like")], 1), 4),
        "valid_known_semantic_entry_cov": round(
            stats[("known", "sem_entry")] /
            max(stats[("known", "novel_like")], 1), 4),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PHASE4L_PROV_ROOT"] = str(
        ROOT / "outputs" / "iclr27_phase4m")
    env["PHASE4L_AUDIT_OUT"] = str(AUDIT)
    for tag in TAGS:
        if not (AUDIT / f"prototype_provenance_{tag}.csv").exists():
            subprocess.run([sys.executable,
                            str(ROOT / "src" / "iclr27_phase4k" /
                                "build_offline_audit.py"),
                            "--tag", tag], env=env, check=True)
    rows = []
    for tag in TAGS:
        row = {"tag": tag}
        row.update(tracking(tag))
        row.update(semantic(tag))
        row.update(resolution(tag))
        row.update(memory(tag))
        row.update(detection(tag))
        rows.append(row)
    with open(OUT / "component_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1))

    anchor = next(r for r in rows if r["tag"] == "m3")
    n2 = next(r for r in rows if r["tag"] == "n2")
    ok = (
        n2["HOTA"] >= 0.95 * anchor["HOTA"] and
        n2["IDSW"] <= 1.15 * anchor["IDSW"] and
        n2["valid_novel_semantic_entry_cov"] >= 0.7 and
        n2["fp_semantic_entry_rate"] <= anchor["fp_semantic_entry_rate"] and
        n2["eventual_resolution_coverage"] >= 0.85
    )
    status = "VALIDITY_AWARE_ROUTING_SUPPORTED" if ok else \
        "VALIDITY_AWARE_ROUTING_NOT_SUPPORTED"
    (DOC / "DEVELOPMENT_RESULTS.md").write_text(
        f"""# Phase 4N Development Results (dev)

Component comparison: N0 = frozen M3 deferral; N2 = N0 + validity gate.
Full rows: `outputs/iclr27_phase4n/dev/component_comparison.csv`.

| tag | HOTA | AssA | IDSW | routing | N2K | novel_cons | protos | USEFUL | POLLUTING | fp_reuse | net | event_cov | fp_sem_entry | novel_sem_cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
            f"| {r['tag']} | {r['HOTA']} | {r['AssA']} | {r['IDSW']} | "
            f"{r.get('routing_accuracy','')} | "
            f"{r.get('n2k_rate_novel_denom','')} | "
            f"{r.get('novel_consistency','')} | "
            f"{r.get('prototypes','')} | {r.get('USEFUL','')} | "
            f"{r.get('POLLUTING','')} | {r.get('fp_reuse_share','')} | "
            f"{r.get('assoc_net_utility','')} | "
            f"{r.get('eventual_resolution_coverage','')} | "
            f"{r.get('fp_semantic_entry_rate','')} | "
            f"{r.get('valid_novel_semantic_entry_cov','')} |"
            for r in rows) +
        f"""

Component decision: `{status}`.
""")
    print("COMPARE_DEV_DONE", status)


if __name__ == "__main__":
    main()
