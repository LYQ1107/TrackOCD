"""Generate data-heavy Phase 4L audit documents from audit CSVs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4l" / "audit"
DOC = ROOT / "docs" / "iclr27_phase4l"


def rows(name):
    p = OUT / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def table(headers, rs):
    return "| " + " | ".join(headers) + " |\n|" + \
        "|".join(["---"] * len(headers)) + "|\n" + "\n".join(
            "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |"
            for r in rs)


def write(name, text):
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / name).write_text(text)


def main():
    det = rows("admissibility_detection_features.csv")
    trk = rows("admissibility_tracklet_features.csv")
    pers = rows("persistent_fp_analysis.csv")
    pred = rows("admissibility_predictability.csv")
    pairs = rows("novel_matching_pairs.csv")
    dist = rows("novel_similarity_distributions.csv")
    radius = rows("prototype_radius.csv")
    margin = rows("margin_analysis.csv")
    mpred = rows("relative_matching_predictability.csv")

    # ---- SEMANTIC_ADMISSIBILITY_AUDIT.md ----
    n_tot = len(det)
    n_tp = sum(1 for r in det if r["gt_role"] in ("known", "novel"))
    n_fp = n_tot - n_tp
    lines = [
        "# Phase 4L Semantic Admissibility Audit (Root Cause A)",
        "",
        f"Detection-level rows: {n_tot} (valid object evidence {n_tp}, "
        f"FP {n_fp}); tracklet-level rows: {len(trk)}. GT is offline-only.",
        "",
        "## Detection feature availability",
        "",
        "Per-detection causal features: detector score, bbox area/aspect, "
        "mask area ratio, appearance/DINO embedding norms, p_known, "
        "best_known, known margin (recomputed with the frozen M2 "
        "aggregator), best novel cosine, association score, semantic "
        "delta, assigned flag, track age, novel support.",
        "",
        "## Tracklet feature availability",
        "",
        "Per-tracklet causal features: length, age, miss gaps, "
        "consecutive hits, detection score stats, p_known stats, "
        "appearance prefix cosine, inter-frame bbox IoU, scale/aspect "
        "change, semantic/gid switch rates, mean association score.",
        "",
        "## Predictability (simple models, 5-fold CV)",
        "",
    ]
    if pred:
        lines.append(table(["tag", "scope", "features", "n", "n_positive",
                            "auroc", "auprc", "tree_auroc"],
                           pred))
    else:
        lines.append("_pending_")
    write("SEMANTIC_ADMISSIBILITY_AUDIT.md", "\n".join(lines))

    # ---- PERSISTENT_FP_ADMISSIBILITY_AUDIT.md ----
    lines = ["# Phase 4L Persistent FP Admissibility Audit", "",
             "FP tracklets split by length buckets; the decisive question "
             "is whether persistent FPs (length >= 6) remain separable.",
             ""]
    if pers:
        lines.append(table(list(pers[0].keys()), pers))
    else:
        lines.append("_pending_")
    write("PERSISTENT_FP_ADMISSIBILITY_AUDIT.md", "\n".join(lines))

    # ---- NOVEL_MATCHING_GEOMETRY_AUDIT.md ----
    diag = {}
    dp = OUT / "matching_diagnostics.json"
    if dp.exists():
        diag = json.loads(dp.read_text())
    lines = ["# Phase 4L Novel Matching Geometry Audit (Root Cause B)",
             "",
             "All novel matching queries from the frozen J1b replay "
             "(tau=0.30, M0): geometry reconstructed causally from the "
             "provenance event stream (no future members, no GT).",
             "",
             "## Query composition",
             "",
             "| case | n | share |",
             "|---|---:|---:|"]
    for k, label in (("n_total", "total"), ("n_same_novel", "SAME_NOVEL"),
                     ("n_different_novel", "DIFFERENT_NOVEL"),
                     ("n_known_collision", "KNOWN_COLLISION"),
                     ("n_fp_query", "FP_QUERY")):
        v = diag.get(k)
        if v is not None:
            lines.append(f"| {label} | {v} | "
                         f"{v / max(diag.get('n_total', 1), 1):.4f} |")
    lines += ["", "## Similarity distributions by case", ""]
    if dist:
        lines.append(table(list(dist[0].keys()), dist))
    lines += ["", "## Margin analysis (true novel, cross-track)", ""]
    if margin:
        lines.append(table(list(margin[0].keys()), margin))
    write("NOVEL_MATCHING_GEOMETRY_AUDIT.md", "\n".join(lines))

    # ---- UMBRELLA_PROTOTYPE_AUDIT.md ----
    lines = ["# Phase 4L Umbrella Prototype Audit", "",
             "Per-prototype causal radius/support/absorption (rebuilt "
             "from the J1b event stream):", ""]
    if radius:
        lines.append(table(list(radius[0].keys()), radius[:15]))
        lines += ["", "Full table: `prototype_radius.csv`.", ""]
    else:
        lines.append("_pending_")
    write("UMBRELLA_PROTOTYPE_AUDIT.md", "\n".join(lines))

    # ---- RELATIVE_MATCHING_AUDIT.md ----
    lines = ["# Phase 4L Relative Matching Audit", "",
             "Can causal relative geometry separate SAME_NOVEL (correct "
             "EXISTING_NOVEL) from DIFFERENT_NOVEL (should create "
             "NEW_NOVEL) on true-novel cross-track queries?", "",
             "## Predictability (logistic / tree, 5-fold CV)", ""]
    if mpred:
        lines.append(table(list(mpred[0].keys()), mpred))
    else:
        lines.append("_pending_")
    lines += ["", "## Threshold trade-offs",
              "", "See `margin_analysis.csv` and the decision document "
              "for precision/recall of candidate rules.", ""]
    write("RELATIVE_MATCHING_AUDIT.md", "\n".join(lines))

    print("PHASE4L_AUDIT_DOCS_GENERATED")


if __name__ == "__main__":
    main()
