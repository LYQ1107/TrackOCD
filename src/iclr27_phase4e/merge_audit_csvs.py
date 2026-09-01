"""Regenerate per-config and canonical Phase 4E audit CSVs from JSON logs.

The canonical shared CSV files (wrong_existing_assignments.csv etc.) are
rebuilt from the four per-track audit logs (C1/C2 x official/long), so both
official and long-stream rows are present with ``config``/``stream`` columns.
The error-source classification reuses ``audit_identity.aggregate_audit``.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4e" / "audit"

from src.orbit.protocol import load_gt
from src.iclr27_phase4d.long_stream import load_stream_cache
from src.iclr27_phase4e.audit_identity import aggregate_audit


CONFIGS = [
    ("c1", "runs/orbit_msr/msr_nr2/model.pth", 0.5, 0.45),
    ("c2", "runs/orbit_msr/msr_c2/model.pth", 0.5, 0.45),
]


def gt_rows_for(stream):
    if stream == "official":
        return load_gt("pure")
    _, gt_rows, _, _ = load_stream_cache()
    return gt_rows


def main():
    summaries = {}
    for cname, *_ in CONFIGS:
        for stream in ["official", "long"]:
            p = OUT / f"audit_log_{cname}_{stream}.json"
            if not p.exists():
                print("missing", p)
                continue
            logs = json.loads(p.read_text())
            gt = gt_rows_for(stream)
            # per-config files with explicit stream suffix to avoid clobbering
            name = f"{cname}_{stream}"
            s = aggregate_audit(logs, gt, OUT, name, stream)
            summaries[f"{cname}|{stream}"] = s
            print(s, flush=True)

    stems = ["wrong_existing_assignments", "first_occurrence_false_merge",
             "false_merge_by_hub_prototype", "prototype_confidence_analysis",
             "confidence_purity_correlation", "wrong_existing_by_memory_scale",
             "wrong_existing_by_prototype_support", "wrong_existing_by_margin"]
    for stem in stems:
        all_rows = []
        for cname, *_ in CONFIGS:
            for stream in ["official", "long"]:
                p = OUT / f"{stem}_{cname}_{stream}.csv"
                if not p.exists():
                    continue
                for r in csv.DictReader(open(p)):
                    r["config"] = cname
                    r["stream"] = stream
                    all_rows.append(r)
        if all_rows:
            fn = list(all_rows[0].keys())
            with open(OUT / f"{stem}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_rows)
            print(stem, len(all_rows))

    # summary with error-source counts
    final = {}
    for key, s in summaries.items():
        cname, stream = key.split("|")
        wrong_p = OUT / f"wrong_existing_assignments_{cname}_{stream}.csv"
        fm_p = OUT / f"first_occurrence_false_merge_{cname}_{stream}.csv"
        err = {}
        if wrong_p.exists():
            err["wrong"] = dict(Counter(
                r.get("primary_category", "?")
                for r in csv.DictReader(open(wrong_p))).most_common())
        if fm_p.exists():
            err["first_merge"] = dict(Counter(
                r.get("primary_category", "?")
                for r in csv.DictReader(open(fm_p))).most_common())
        final[key] = dict(s)
        final[key]["error_sources"] = err
    (OUT / "audit_summary_all.json").write_text(
        json.dumps(final, indent=1))
    print("wrote audit_summary_all.json")


if __name__ == "__main__":
    main()
