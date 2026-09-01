"""Generate the data-heavy Phase 4K audit documents from the audit CSVs."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4k" / "audit"
DOC = ROOT / "docs" / "iclr27_phase4k"
TAGS = ("j0", "j1b", "m1")


def csv_rows(name):
    p = OUT / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def pct(x, n):
    return f"{x / n:.3f}" if n else "n/a"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + \
        "|".join(["---"] * len(headers)) + "|\n" + "\n".join(
            "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |"
            for r in rows)


def write(path, text):
    DOC.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    utility = {r["tag"]: r for r in csv_rows("prototype_utility.csv")}
    prov = {t: csv_rows(f"prototype_provenance_{t}.csv") for t in TAGS}
    inter = csv_rows("association_interventions_j1b.csv") or \
        csv_rows("association_interventions.csv")
    cross = csv_rows("cross_track_support.csv")
    hubs = csv_rows("pollution_hubs.csv")
    pred = csv_rows("causal_predictability.csv")
    feat = csv_rows("causal_feature_auroc.csv")

    # ---- NOVEL_MEMORY_PROVENANCE_AUDIT.md ----
    lines = [
        "# Phase 4K Novel Memory Provenance Audit",
        "",
        "Provenance was instrumented strictly online (no GT, no future) in",
        "`SemanticStateManager` / `FrameOnlineTracker` and replayed with the",
        "frozen Phase 4J configurations J0 (τ=0.50, M0), J1b (τ=0.30, M0) and",
        "M1/J2b (τ=0.30, M1 age≥2 support≥2). GT is used only in this offline",
        "audit to label prototypes and association decisions.",
        "",
        "## Replay and byte equivalence",
    ]
    eq_rows = []
    for t in TAGS:
        eqp = OUT / f"prov_{t}" / "equivalence.json"
        eq = json.loads(eqp.read_text()) if eqp.exists() else {}
        eq_rows.append({
            "tag": t, "images": eq.get("images", ""),
            "byte_exact": eq.get("byte_exact", ""),
            "diff": len(eq.get("diff", [])),
            "missing": len(eq.get("missing", [])),
            "extra": len(eq.get("extra", [])),
        })
    lines.append(table(["tag", "images", "byte_exact", "diff", "missing",
                        "extra"], eq_rows))
    lines += [
        "",
        "## Event volume and prototype counts",
    ]
    ev_rows = []
    for t in TAGS:
        s = json.loads((OUT / f"offline_summary_{t}.json").read_text()) \
            if (OUT / f"offline_summary_{t}.json").exists() else {}
        ev_rows.append({
            "tag": t, "births": s.get("births", ""),
            "updates": s.get("updates", ""), "reuses": s.get("reuses", ""),
            "decisions": s.get("decisions", ""),
            "prototypes": s.get("prototypes", ""),
        })
    lines.append(table(["tag", "births", "updates", "reuses",
                        "decisions", "prototypes"], ev_rows))
    lines += ["", "## Prototype lifecycle fields",
              "", "Per-prototype rows (birth, updates, reuses, embedding",
              "dispersion/drift, contributor tracks/videos, association",
              "utility, outcome group) are in",
              "`outputs/iclr27_phase4k/audit/prototype_provenance_*.csv`.",
              ""]
    write(DOC / "NOVEL_MEMORY_PROVENANCE_AUDIT.md", "\n".join(lines))

    # ---- PROTOTYPE_UTILITY_AUDIT.md ----
    lines = ["# Phase 4K Prototype Utility Audit", "",
             "Outcome groups are fixed and transparent:",
             "",
             "- **USEFUL**: ≥2 reuses, majority of reuses match the same real",
             "  GT-novel category, known+FP absorption < 50%, helpful",
             "  association effects ≥ harmful.",
             "- **POLLUTING**: ≥2 reuses and (known+FP absorption ≥ 50% or a",
             "  different GT-novel category outnumbers the majority category).",
             "- **MIXED**: ≥2 reuses that are neither USEFUL nor POLLUTING.",
             "- **LOW_EVIDENCE**: <2 reuses.",
             "",
             "## Summary by tag",
             ""]
    urows = [{
        "tag": r["tag"], "prototypes": r["n_prototypes"],
        "useful": r["n_useful"], "polluting": r["n_polluting"],
        "mixed": r["n_mixed"], "low_evidence": r["n_low_evidence"],
        "useful_frac": r["useful_frac"], "polluting_frac": r["polluting_frac"],
        "mean_support_useful": r["mean_support_useful"],
        "mean_support_polluting": r["mean_support_polluting"],
        "mean_purity_useful": r["mean_purity_useful"],
        "mean_purity_polluting": r["mean_purity_polluting"],
        "net_assoc_utility": r["net_assoc_utility"],
    } for r in utility.values()]
    lines.append(table(list(urows[0].keys()), urows))
    lines += ["", "## Per-prototype rows",
              "", "`outputs/iclr27_phase4k/audit/prototype_utility.csv` and",
              "`prototype_provenance_j1b.csv` (main object) contain the raw",
              "dimensions; j0/m1 provide the same audit for the two controls.",
              ""]
    write(DOC / "PROTOTYPE_UTILITY_AUDIT.md", "\n".join(lines))

    # ---- ASSOCIATION_INTERVENTION_AUDIT.md ----
    eff = Counter(r["effect"] for r in inter)
    lines = ["# Phase 4K Association Intervention Audit", "",
             "For every semantic association decision we compute the",
             "appearance-only best candidate (ap) and the final best",
             "candidate (fn) with semantic consistency included. A",
             "decision effect exists when semantics change the argmax or",
             "cross the match threshold; GT then labels the effect",
             "helpful / harmful / neutral / no-effect.",
             "",
             "## Effect totals",
             ""]
    lines.append(table(["effect", "count"], [
        {"effect": k, "count": v} for k, v in
        sorted(eff.items(), key=lambda x: -x[1])]))
    lines += ["", "## Per-tag effect totals", ""]
    eff_rows = []
    for t in TAGS:
        rows = csv_rows(f"association_interventions_{t}.csv")
        c = Counter(r["effect"] for r in rows)
        eff_rows.append({
            "tag": t, "no_effect": c.get("no_effect", 0),
            "neutral_switch": c.get("neutral_switch", 0),
            "helpful": c.get("helpful", 0), "harmful": c.get("harmful", 0),
        })
    lines.append(table(["tag", "no_effect", "neutral_switch", "helpful",
                        "harmful"], eff_rows))
    lines += ["", "## Prototype-level net utility",
              "",
              "Per-prototype helpful/harmful/neutral counts and net utility",
              "are in `outputs/iclr27_phase4k/audit/prototype_provenance_*.csv`",
              "and every labeled decision is in",
              "`association_interventions.csv`.",
              ""]
    write(DOC / "ASSOCIATION_INTERVENTION_AUDIT.md", "\n".join(lines))

    # ---- CROSS_TRACK_SUPPORT_AUDIT.md ----
    lines = ["# Phase 4K Cross-Track Support Audit", "",
             "A prototype has cross-track support when at least two",
             "different (video, physical-track) keys contributed. The key",
             "question: does independent physical-track evidence separate",
             "USEFUL from POLLUTING memory better than age/support alone?",
             "",
             "## Cross-track vs same-track (J1b main)",
             ""]
    cr = [r for r in cross if r["tag"] == "j1b" and
          r["outcome_group"] in ("USEFUL", "POLLUTING", "MIXED")]
    rows = []
    for grp in ("USEFUL", "POLLUTING", "MIXED"):
        sub = [r for r in cr if r["outcome_group"] == grp]
        x = [r for r in sub if r["cross_track"] == "1"]
        s = [r for r in sub if r["cross_track"] == "0"]
        rows.append({
            "group": grp, "n": len(sub), "cross_track_n": len(x),
            "same_track_n": len(s),
            "cross_track_mean_purity": round(
                sum(float(r["semantic_purity"]) for r in x) / len(x), 3)
            if x else "n/a",
            "same_track_mean_purity": round(
                sum(float(r["semantic_purity"]) for r in s) / len(s), 3)
            if s else "n/a",
            "cross_track_net_assoc": sum(int(r["assoc_net_utility"])
                                         for r in x),
            "same_track_net_assoc": sum(int(r["assoc_net_utility"])
                                        for r in s),
        })
    lines.append(table(["group", "n", "cross_track_n", "same_track_n",
                        "cross_track_mean_purity", "same_track_mean_purity",
                        "cross_track_net_assoc", "same_track_net_assoc"],
                       rows))
    lines += ["", "Per-prototype rows: `cross_track_support.csv`.", ""]
    write(DOC / "CROSS_TRACK_SUPPORT_AUDIT.md", "\n".join(lines))

    # ---- POLLUTION_HUB_AUDIT.md ----
    harm = [r for r in hubs if r["hub_type"] == "harmful" and
            r["tag"] == "j1b"]
    useful = [r for r in hubs if r["hub_type"] == "useful" and
              r["tag"] == "j1b"]
    allp = prov["j1b"]
    total_abs = sum(int(r["known_absorptions"]) + int(r["fp_absorptions"])
                    for r in allp)
    top10_abs = sum(int(r["known_absorptions"]) + int(r["fp_absorptions"])
                    for r in sorted(allp, key=lambda r:
                    int(r["known_absorptions"]) + int(r["fp_absorptions"]),
                    reverse=True)[:max(1, len(allp) // 10)])
    total_harm = sum(int(r["assoc_harmful"]) for r in allp)
    top10_harm = sum(int(r["assoc_harmful"]) for r in sorted(
        allp, key=lambda r: int(r["assoc_harmful"]), reverse=True)[
            :max(1, len(allp) // 10)])
    lines = ["# Phase 4K Pollution Hub Audit", "",
             f"J1b: total known+FP absorptions = {total_abs}; top-10% of",
             f"prototypes account for {top10_abs} "
             f"({pct(top10_abs, total_abs)}). Total harmful association",
             f"effects = {total_harm}; top-10% account for {top10_harm} "
             f"({pct(top10_harm, total_harm)}).",
             "",
             "## Top harmful prototypes (J1b)", ""]
    lines.append(table(list(harm[0].keys()), harm[:12]))
    lines += ["", "## Top useful prototypes (J1b)", ""]
    lines.append(table(list(useful[0].keys()), useful[:12]))
    lines += ["", "Full top-20 lists: `pollution_hubs.csv`.", ""]
    write(DOC / "POLLUTION_HUB_AUDIT.md", "\n".join(lines))

    # ---- CAUSAL_PREDICTABILITY_AUDIT.md ----
    lines = ["# Phase 4K Causal Predictability Audit", "",
             "Can online-available evidence at a past checkpoint separate",
             "prototypes that ultimately become USEFUL from those that",
             "become POLLUTING? Features at each checkpoint use only events",
             "up to that time (no future, no GT); GT enters only through the",
             "retrospective label. Models: standardized logistic regression",
             "and a depth-3 tree, 5-fold stratified CV.",
             "",
             "## Time-conditioned predictability (J1b main)",
             ""]
    pred_rows = [r for r in pred if r["tag"] == "j1b"]
    lines.append(table(["checkpoint", "n", "n_useful", "n_polluting",
                        "auroc_logistic", "auprc_logistic", "auroc_tree",
                        "top_single_feature", "top_single_auroc"],
                       pred_rows))
    lines += ["", "## Best single features (J1b, all checkpoints)",
              ""]
    feat_rows = [r for r in feat if r["tag"] == "j1b"]
    best = sorted(feat_rows, key=lambda r: -float(r["auroc"]))[:15]
    lines.append(table(["checkpoint", "feature", "auroc", "n", "n_useful",
                        "n_polluting"], best))
    lines += ["", "Full tables: `causal_predictability.csv` and",
              "`causal_feature_auroc.csv`; per-prototype checkpoint features:",
              "`causal_checkpoint_features.csv`.", ""]
    write(DOC / "CAUSAL_PREDICTABILITY_AUDIT.md", "\n".join(lines))

    print("PHASE4K_DOCS_GENERATED")


if __name__ == "__main__":
    main()
