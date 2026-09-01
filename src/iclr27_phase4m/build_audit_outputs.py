"""Phase 4M audit CSV assembly + geometry/Pareto summaries.

Consumes the corrected identity decision CSVs and the retrospective
deferral oracle CSVs and emits:
  - overbirth_events.csv / wrong_reuse_events.csv / ambiguity_features.csv
  - time_to_resolution.csv (already from the oracle; copied with tag)
  - deferral_pareto.csv
  - geometry_summary.json (four-class stats + AUROC)
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4m" / "audit"
TAGS = ["j1b", "b1", "b2"]


def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    pos = s[y]
    neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    u = ranks[y].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    all_overbirth = []
    all_wrong_reuse = []
    all_amb = []
    summary = {}
    pareto_rows = []
    for tag in TAGS:
        rows = list(csv.DictReader(open(AUDIT / f"identity_decisions_v2_{tag}.csv")))
        retr = {(r["video_id"], int(r["frame_id"]), r["track_id"]): r for r
                in csv.DictReader(
                    open(AUDIT / f"retrospective_{tag}_summary.csv"))}
        amb = [r for r in rows if r["query_gt_role"] == "novel"]
        for r in amb:
            out = {"tag": tag,
                   "video_id": r["video_id"], "frame_id": r["frame_id"],
                   "track_id": r["track_id"], "sem_id": r["sem_id"],
                   "branch": r["branch"], "actual_action": r["actual_action"],
                   "decision_class": r["decision_class"],
                   "query_gt_category": r["query_gt_category"],
                   "best_cos": r["best_cos"], "second_cos": r["second_cos"],
                   "margin": r["margin"], "entropy": r["entropy"],
                   "novel_minus_known": r["novel_minus_known"],
                   "query_zscore": r["query_zscore"],
                   "member_mean_cos": r["member_mean_cos"],
                   "member_std_cos": r["member_std_cos"],
                   "support_causal": r["support_causal"],
                   "near_count": r["near_count"],
                   "correct": int(r["decision_class"] in
                                  ("CORRECT_EXISTING", "CORRECT_NEW"))}
            rr = retr.get((r["video_id"], int(r["frame_id"]),
                           r["track_id"]))
            if rr is not None:
                out["oracle_resolved_by_t8"] = int(
                    rr["resolved_correctly_by_t8"])
                out["terminated_before_t8"] = int(rr["terminated_before_t8"])
            else:
                out["oracle_resolved_by_t8"] = ""
                out["terminated_before_t8"] = ""
            all_amb.append(out)
        for r in rows:
            if r["decision_class"] == "OVERBIRTH":
                all_overbirth.append(r)
            if r["decision_class"] == "WRONG_EXISTING" and \
                    r["query_gt_role"] == "novel":
                all_wrong_reuse.append(r)

        # geometry summary
        cls = Counter(r["decision_class"] for r in amb)
        geos = defaultdict(lambda: defaultdict(list))
        for r in amb:
            for f in ("best_cos", "margin", "entropy",
                      "novel_minus_known", "query_zscore"):
                try:
                    geos[r["decision_class"]][f].append(float(r[f]))
                except (TypeError, ValueError):
                    pass
        y = [1 if r["decision_class"] in ("CORRECT_EXISTING",
                                          "CORRECT_NEW") else 0
             for r in amb]
        aurocs = {}
        for f in ("best_cos", "margin", "entropy",
                  "novel_minus_known", "query_zscore"):
            vals = []
            for r, yy in zip(amb, y):
                try:
                    vals.append((yy, float(r[f])))
                except (TypeError, ValueError):
                    pass
            if len(vals) >= 8 and sum(a for a, _ in vals) >= 2 and \
                    sum(1 - a for a, _ in vals) >= 2:
                aurocs[f] = auroc([a for a, _ in vals],
                                  [b for _, b in vals])
            else:
                aurocs[f] = None
        summary[tag] = {
            "novel_decisions": len(amb),
            "classes": dict(cls),
            "geometry_median": {
                c: {f: float(np.median(geos[c][f]))
                    for f in geos[c]} for c in geos},
            "correct_auroc": aurocs,
        }

        # deferral Pareto on novel decisions with oracle outcomes
        rows_with_oracle = [r for r in all_amb if r["tag"] == tag and
                            r["oracle_resolved_by_t8"] != ""]
        candidates = [
            ("margin<0.05", lambda r: float(r["margin"]) < 0.05),
            ("margin<0.10", lambda r: float(r["margin"]) < 0.10),
            ("best<0.70", lambda r: float(r["best_cos"]) < 0.70),
            ("best<0.75", lambda r: float(r["best_cos"]) < 0.75),
            ("nK<0.20", lambda r: float(r["novel_minus_known"]) < 0.20),
            ("nK<0.30", lambda r: float(r["novel_minus_known"]) < 0.30),
            ("entropy>1.6", lambda r: float(r["entropy"]) > 1.6),
            ("best<.75 or mar<.05",
             lambda r: float(r["best_cos"]) < 0.75 or
             float(r["margin"]) < 0.05),
        ]
        n = len(rows_with_oracle)
        for name, cond in candidates:
            defer = [r for r in rows_with_oracle if cond(r)]
            resolve = [r for r in defer if r["oracle_resolved_by_t8"] == 1]
            decide = [r for r in rows_with_oracle if not cond(r)]
            decide_err = sum(1 for r in decide if not r["correct"])
            pareto_rows.append({
                "tag": tag, "rule": name,
                "defer_fraction": round(len(defer) / max(n, 1), 4),
                "immediate_decisions": len(decide),
                "immediate_error_rate": round(
                    decide_err / max(len(decide), 1), 4),
                "eventual_resolution_coverage": round(
                    len(resolve) / max(len(defer), 1), 4),
                "prevented_errors_if_resolved": len(resolve),
                "unresolved_at_termination": sum(
                    1 for r in defer
                    if r["terminated_before_t8"] == 1),
                "mean_delay_frames": 8,
            })

    # write CSVs
    def write(path, rows, extra=None):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(AUDIT / "overbirth_events.csv", all_overbirth)
    write(AUDIT / "wrong_reuse_events.csv", all_wrong_reuse)
    write(AUDIT / "ambiguity_features.csv", all_amb)
    write(AUDIT / "deferral_pareto.csv", pareto_rows)
    # error-cost summary: consequences of wrong reuse vs overbirth
    cost_rows = []
    for tag in TAGS:
        prov = AUDIT / f"prototype_provenance_{tag}.csv"
        if not prov.exists():
            prov = ROOT / "outputs" / "iclr27_phase4l" / "audit" / \
                f"prototype_provenance_{tag}.csv"
        if not prov.exists():
            continue
        rows = list(csv.DictReader(open(prov)))
        ob = [r for r in all_overbirth if r["tag"] == tag]
        wr = [r for r in all_wrong_reuse if r["tag"] == tag]
        cost_rows.append({
            "tag": tag,
            "overbirth_events": len(ob),
            "wrong_reuse_novel_events": len(wr),
            "prototypes": len(rows),
            "fp_reuse_share": round(sum(
                int(r["fp_absorptions"]) for r in rows) /
                max(sum(int(r["n_reuses"]) for r in rows), 1), 4),
            "net_assoc_utility": sum(
                int(r["assoc_helpful"]) - int(r["assoc_harmful"])
                for r in rows),
        })
    write(AUDIT / "error_cost_analysis.csv", cost_rows)
    with open(AUDIT / "geometry_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("AUDIT_OUTPUTS_DONE",
          len(all_overbirth), len(all_wrong_reuse), len(all_amb),
          len(pareto_rows), len(cost_rows))


if __name__ == "__main__":
    main()
