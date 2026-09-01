"""Phase 4G metric-definition reconciliation.

Reconciles the two known-origin counts (180/398 vs 259/398) and the hub
counts (178 vs 37+46) from Phase 4E/4F by computing both definitions from
the frozen Phase 4F audit logs.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def main():
    out = f"{ROOT}/outputs/iclr27_phase4g/audit"
    import pathlib
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    origin_rows = []
    hub_rows = []
    for method in ["c1", "iam", "m2"]:
        for stream in ["official", "long"]:
            logs_path = f"{ROOT}/outputs/iclr27_phase4f/audit/memory_trajectory_{method}_{stream}.csv"
            try:
                logs = list(csv.DictReader(open(logs_path)))
            except FileNotFoundError:
                continue
            by_vid = defaultdict(list)
            for l in logs:
                vid = l.get("predicted_virtual_novel_id")
                if vid not in (None, ""):
                    by_vid[int(vid)].append(l)
            for vid, ls in by_vid.items():
                ls_sorted = sorted(ls, key=lambda x: int(x["arrival_index"]))
                roles = [l["true_role"] for l in ls]
                novel_roles = [l["true_role"] for l in ls if l["true_role"] == "novel"]
                classes_all = set(int(l["true_class"]) for l in ls)
                classes_novel = set(int(l["true_class"]) for l in ls
                                    if l["true_role"] == "novel")
                primary = Counter(int(l["true_class"]) for l in ls
                                  if l["true_role"] == "novel")
                primary_c = primary.most_common(1)[0][0] if primary else None
                wrong_existing = sum(1 for l in ls
                                     if l["predicted_action"] == "EXISTING_NOVEL"
                                     and primary_c is not None
                                     and int(l["true_class"]) != primary_c)
                origin_rows.append({
                    "method": method, "stream": stream, "virtual_id": vid,
                    "birth_origin_role": ls_sorted[0]["true_role"],
                    "ever_contaminated_by_known": int("known" in roles),
                    "novel_free": int(len(novel_roles) == 0 and len(ls) > 0),
                    "novel_assignment_count": len(novel_roles),
                    "total_assignment_count": len(ls),
                })
                hub_rows.append({
                    "method": method, "stream": stream, "virtual_id": vid,
                    "birth_origin_role": ls_sorted[0]["true_role"],
                    "global_absorption_hub": int(len(classes_all) >= 2),
                    "novel_absorption_hub": int(len(classes_novel) >= 2),
                    "causal_wrong_existing": wrong_existing,
                    "novel_classes_absorbed": len(classes_novel),
                    "total_classes_absorbed": len(classes_all),
                })
    fn = list(origin_rows[0].keys())
    with open(f"{out}/prototype_origin_definition_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(origin_rows)
    fn2 = list(hub_rows[0].keys())
    with open(f"{out}/hub_definition_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn2)
        w.writeheader()
        w.writerows(hub_rows)

    # summary table
    summary = []
    for method in ["c1", "iam", "m2"]:
        for stream in ["official", "long"]:
            o = [r for r in origin_rows if r["method"] == method
                 and r["stream"] == stream]
            h = [r for r in hub_rows if r["method"] == method
                 and r["stream"] == stream]
            if not o:
                continue
            summary.append({
                "method": method, "stream": stream,
                "total_virtual_ids": len(o),
                "birth_origin_known": sum(1 for r in o
                                          if r["birth_origin_role"] == "known"),
                "ever_contaminated_by_known": sum(1 for r in o
                                                  if r["ever_contaminated_by_known"]),
                "novel_free": sum(1 for r in o if r["novel_free"]),
                "global_absorption_hub": sum(1 for r in h
                                             if r["global_absorption_hub"]),
                "novel_absorption_hub": sum(1 for r in h
                                            if r["novel_absorption_hub"]),
                "causal_wrong_hub_ge2": sum(1 for r in h
                                            if r["causal_wrong_existing"] >= 2),
                "causal_wrong_hub_ge5": sum(1 for r in h
                                            if r["causal_wrong_existing"] >= 5),
            })
    with open(f"{out}/definition_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
