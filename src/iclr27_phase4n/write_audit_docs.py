"""Phase 4N audit docs + root decisions from the audit CSVs."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4n" / "audit"
DOC = ROOT / "docs" / "iclr27_phase4n"


def load(name):
    return list(csv.DictReader(open(AUDIT / name)))


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    pop = load("detection_population.csv")
    dist = load("detector_score_distributions.csv")
    thr = load("detector_threshold_curve.csv")
    pers = load("persistent_fp_features.csv")
    pred = load("validity_predictability.csv")
    shift = load("gate_shift_summary.csv")
    age = load("gate_shift_by_age.csv")
    vid = load("gate_shift_by_video.csv")
    inter = load("detector_gate_interaction.csv")

    # ---------- detection population doc ----------
    pop_rows = []
    for mode in ("dev", "heldout"):
        rows = [r for r in pop if r["mode"] == mode]
        c = Counter(r["gt_role"] for r in rows)
        pop_rows.append({
            "mode": mode, "n": len(rows),
            "VALID_KNOWN": c.get("known", 0),
            "VALID_NOVEL": c.get("novel", 0),
            "FP": c.get("fp", 0),
            "novel_frac": round(c.get("novel", 0) / max(len(rows), 1), 4),
            "fp_frac": round(c.get("fp", 0) / max(len(rows), 1), 4),
        })
    (DOC / "DETECTION_POPULATION_AUDIT.md").write_text(
        "# Phase 4N Detection Population Audit\n\n"
        "| mode | total | VALID_KNOWN | VALID_NOVEL | FP | novel_frac | fp_frac |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['mode']} | {r['n']} | {r['VALID_KNOWN']} | "
            f"{r['VALID_NOVEL']} | {r['FP']} | {r['novel_frac']} | "
            f"{r['fp_frac']} |" for r in pop_rows) +
        "\n\nFP dominates the detection stream; true novel detections are "
        "a tiny minority.  Full rows: `detection_population.csv`.\n")

    # ---------- detector fp score doc ----------
    (DOC / "DETECTOR_FP_SCORE_AUDIT.md").write_text(
        "# Phase 4N Detector FP Score Audit\n\n"
        "Score distributions per role and AUROC/AUPRC "
        "(valid-vs-FP and novel-vs-FP):\n\n"
        "| mode | role | n | mean | std | q25 | median | q75 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['mode']} | {r['role']} | {r['n']} | {r['mean']} | "
            f"{r['std']} | {r['q25']} | {r['median']} | {r['q75']} |"
            for r in dist) +
        "\n\nThreshold oracle (novel recall vs FP): "
        "`detector_threshold_curve.csv`.\n")

    # ---------- persistent fp doc ----------
    pers_stats = []
    for mode in ("dev", "heldout"):
        rows = [r for r in pers if r["mode"] == mode]
        for role in ("FP", "KNOWN", "NOVEL"):
            rs = [r for r in rows if r["role"] == role]
            if not rs:
                continue
            pers_stats.append({
                "mode": mode, "role": role, "n": len(rs),
                "mean_frames": round(sum(int(r["frames"]) for r in rs) /
                                     len(rs), 2),
                "mean_score": round(sum(float(r["mean_score"]) for r in rs) /
                                    len(rs), 4),
                "mean_p_known": round(
                    sum(float(r["mean_p_known"]) for r in rs) / len(rs), 4),
            })
    (DOC / "PERSISTENT_FP_FRONTEND_AUDIT.md").write_text(
        "# Phase 4N Persistent FP Frontend Audit\n\n"
        "Tracklet-level (majority GT role):\n\n"
        "| mode | role | n | mean_frames | mean_score | mean_p_known |\n"
        "|---|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['mode']} | {r['role']} | {r['n']} | "
            f"{r['mean_frames']} | {r['mean_score']} | "
            f"{r['mean_p_known']} |" for r in pers_stats) +
        "\n\nFull rows: `persistent_fp_features.csv`.\n")

    # ---------- validity predictability doc ----------
    (DOC / "OBJECT_VALIDITY_PREDICTABILITY_AUDIT.md").write_text(
        "# Phase 4N Object Validity Predictability Audit\n\n"
        "Logistic regression (trained on dev, evaluated on dev and "
        "held-out; valid vs FP):\n\n"
        "| features | eval_mode | AUROC | AUPRC | novel_recall@prec0.3 | "
        "fp_rej@novel_rec0.7 |\n"
        "|---|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['features']} | {r['mode']} | {r['AUROC']} | "
            f"{r['AUPRC']} | {r['valid_novel_recall_at_prec0.3']} | "
            f"{r['fp_rejection_at_novel_recall0.7']} |" for r in pred) +
        "\n")

    # ---------- gate shift doc ----------
    sep = {}
    for r in shift:
        if r.get("mode") and r["role"] == "known-vs-novel":
            sep[r["mode"]] = (r["feature"], r.get("AUROC"), r.get("AUPRC"))
    (DOC / "KNOWN_NOVEL_GATE_SHIFT_AUDIT.md").write_text(
        "# Phase 4N Known-Novel Gate Shift Audit\n\n"
        "Known-vs-novel separability (valid detections only):\n\n"
        "| mode | feature | AUROC | AUPRC |\n"
        "|---|---:|---:|---:|\n" +
        "\n".join(
            f"| {m} | {f} | {a} | {p} |" for m, (f, a, p) in sep.items()) +
        "\n\nDev-to-held-out shift per feature (known and novel rows): "
        "`gate_shift_summary.csv` (KS distance and Cohen's d).\n")

    # ---------- track-age doc ----------
    (DOC / "TRACK_AGE_GATE_SHIFT_AUDIT.md").write_text(
        "# Phase 4N Track-Age Gate Shift Audit\n\n"
        "| mode | age | known_n | novel_n | routing_acc | k2n | n2k | "
        "mean_p_known |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['mode']} | {r['age_bucket']} | {r['known_n']} | "
            f"{r['novel_n']} | {r['routing_acc']} | {r['k2n']} | "
            f"{r['n2k']} | {r['mean_p_known']} |" for r in age) +
        "\n")

    # ---------- video-level doc ----------
    vstats = defaultdict(list)
    for r in vid:
        vstats[r["mode"]].append(r)
    parts = []
    for mode in ("dev", "heldout"):
        rs = vstats.get(mode, [])
        n2k = [float(r["n2k"]) for r in rs if r["n2k"] != ""]
        rout = [float(r["routing_acc"]) for r in rs
                if r["routing_acc"] != ""]
        fp = [int(r["fp_count"]) for r in rs]
        import numpy as np
        parts.append(
            f"**{mode}** ({len(rs)} videos): N2K median "
            f"{np.median(n2k):.3f} (IQR {np.percentile(n2k,25):.3f}-"
            f"{np.percentile(n2k,75):.3f}); routing median "
            f"{np.median(rout):.3f}; FP/video median {np.median(fp):.0f}.")
    (DOC / "VIDEO_LEVEL_ROUTING_SHIFT_AUDIT.md").write_text(
        "# Phase 4N Video-Level Routing Shift Audit\n\n" +
        "\n\n".join(parts) +
        "\n\nPer-video rows: `gate_shift_by_video.csv`.  The N2K shift is "
        "not a single-video artifact if it appears across the median.\n")

    # ---------- detector x gate doc ----------
    (DOC / "DETECTOR_GATE_COUPLING_AUDIT.md").write_text(
        "# Phase 4N Detector x Gate Coupling Audit\n\n"
        "| mode | score bucket | n | known | novel | fp | "
        "frac_routed_novel | fp_routed_novel | novel_routed_known |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r['mode']} | {r['score_bucket']} | {r['n']} | "
            f"{r['known']} | {r['novel']} | {r['fp']} | "
            f"{r['fraction_routed_novel']} | {r['fp_routed_novel']} | "
            f"{r['novel_routed_known']} |" for r in inter) +
        "\n")

    # ---------- root decisions ----------
    def get_pred(feats, mode):
        for r in pred:
            if r["features"] == feats and r["mode"] == mode:
                return r
        return {}

    d0 = get_pred("D0_detector", "dev")
    d1 = get_pred("D1_tracking", "dev")
    d2 = get_pred("D2_semantic", "dev")
    d4 = get_pred("D4_all", "dev")
    d4ho = get_pred("D4_all", "heldout")
    a0 = float(d0.get("AUROC") or 0)
    a1 = float(d1.get("AUROC") or 0)
    a2 = float(d2.get("AUROC") or 0)
    a4 = float(d4.get("AUROC") or 0)
    a4ho = float(d4ho.get("AUROC") or 0)
    nvfp = next((r for r in dist if r["role"] == "novel-vs-FP_AUROC" and
                 r["mode"] == "dev"), {})
    nv_auprc = float(nvfp.get("AUPRC") or 0)
    if a0 >= 0.8:
        front = "FRONTEND_VALIDITY_SIGNAL_STRONG"
    elif a0 >= 0.65:
        front = "FRONTEND_VALIDITY_SIGNAL_PARTIAL"
    elif a0 >= 0.52:
        front = "FRONTEND_VALIDITY_SIGNAL_WEAK"
    else:
        front = "NO_FRONTEND_VALIDITY_SIGNAL"
    extra = []
    if a1 - a0 >= 0.03:
        extra.append("TRACKING_VALIDITY_ADDS_VALUE")
    if a2 - a0 >= 0.03:
        extra.append("SEMANTIC_VALIDITY_ADDS_VALUE")
    if a0 >= 0.85 and nv_auprc >= 0.5:
        extra.append("DETECTOR_SCORE_SUFFICIENT")
    elif nv_auprc < 0.2:
        extra.append("NEW_DETECTOR_REQUIRED")
    if a4ho < 0.6:
        extra.append("VALIDITY_TRANSFER_WEAK")
    if a4 - a0 >= 0.05:
        extra.append("MULTI_SOURCE_VALIDITY_ADDS_VALUE")

    # gate root
    sep_ho = {f: (a, p) for m, (f, a, p) in sep.items()
              if m == "heldout" and a not in ("", None)}
    ho_auroc = max((float(a) for f, (a, p) in sep_ho.items()), default=0)
    # dev/held-out routing gap on valid-matched detections
    gaps = {}
    for mode in ("dev", "heldout"):
        rows = [r for r in pop if r["mode"] == mode and
                r["gt_role"] in ("known", "novel") and
                r.get("p_known") not in ("", None)]
        known = [r for r in rows if r["gt_role"] == "known"]
        novel = [r for r in rows if r["gt_role"] == "novel"]
        k2n = sum(1 for r in known if float(r["p_known"]) < 0.30)
        n2k = sum(1 for r in novel if float(r["p_known"]) >= 0.30)
        correct = (len(known) - k2n) + (len(novel) - n2k)
        gaps[mode] = {
            "routing": correct / max(len(rows), 1),
            "n2k": n2k / max(len(novel), 1),
        }
    gap_r = abs(gaps["dev"]["routing"] - gaps["heldout"]["routing"])
    gap_n = abs(gaps["dev"]["n2k"] - gaps["heldout"]["n2k"])
    if ho_auroc >= 0.7 and gap_r < 0.15 and gap_n < 0.20:
        gate = "NO_CLEAR_ROUTING_SHIFT"
    elif ho_auroc >= 0.7:
        gate = "CALIBRATION_SHIFT_DOMINANT"
    elif ho_auroc >= 0.55:
        gate = "MIXED_ROUTING_SHIFT"
    else:
        gate = "REPRESENTATION_SHIFT_DOMINANT"

    (DOC / "FRONTEND_ROOT_DECISION.md").write_text(
        f"""# Phase 4N Frontend Root Decision

Status: `{front}`.

Detector-score validity AUROC (dev): {round(a0,3)}; detector+tracking+
semantic (D4): {round(a4,3)}; D4 on held-out: {round(a4ho,3)}.
Supplementary flags: {", ".join(extra) or "none"}.

Full evidence: OBJECT_VALIDITY_PREDICTABILITY_AUDIT.md and
`validity_predictability.csv`.
""")
    (DOC / "GATE_ROOT_DECISION.md").write_text(
        f"""# Phase 4N Gate Root Decision

Status: `{gate}`.

Held-out known-vs-novel separability (best feature): AUROC
{round(ho_auroc,3)}.  See KNOWN_NOVEL_GATE_SHIFT_AUDIT.md and
`gate_shift_summary.csv`.  Dev/held-out valid routing gap:
{round(gap_r,3)}; N2K gap: {round(gap_n,3)}.
""")
    print("AUDIT_DOCS_DONE", front, gate)


if __name__ == "__main__":
    main()
