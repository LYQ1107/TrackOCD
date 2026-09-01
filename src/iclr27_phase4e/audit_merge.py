"""Merge Phase 4E audit logs (C1/C2 x official/long) into canonical CSVs.

Reads the per-track audit logs written by audit_identity.py and produces the
canonical audit artifacts required by the Phase 4E protocol:
  wrong_existing_assignments.csv
  wrong_existing_by_memory_scale.csv
  wrong_existing_by_prototype_support.csv
  wrong_existing_by_margin.csv
  first_occurrence_false_merge.csv
  false_merge_by_hub_prototype.csv
  prototype_confidence_analysis.csv
  confidence_purity_correlation.csv

All statistics are offline audit statistics.  GT labels are used only to
attribute errors after the decision is made; they never enter the causal
decision loop.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
AUDIT = ROOT / "outputs" / "iclr27_phase4e" / "audit"

CONFIGS = [
    ("c1", "official"),
    ("c2", "official"),
    ("c1", "long"),
    ("c2", "long"),
]


def load_logs():
    logs_by_cfg = {}
    for model, stream in CONFIGS:
        p = AUDIT / f"audit_log_{model}_{stream}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        logs_by_cfg[(model, stream)] = json.loads(p.read_text())
    return logs_by_cfg


def error_flags(l):
    f = {
        "same_class_proto_unselected": bool(l.get("true_class_prototype_exists")),
        "hub_pollution": bool(l.get("assigned_consistency", 1.0) is not None
                              and float(l.get("assigned_consistency", 1.0)) < 0.6)
                         or bool(l.get("assigned_distinct_classes", 0) is not None
                                 and int(l.get("assigned_distinct_classes", 0)) >= 2),
        "radius_wide": bool(l.get("assigned_radius", 0.0) is not None
                            and float(l.get("assigned_radius", 0.0)) >= 0.5
                            and float(l.get("best_novel_similarity", 1.0)) < 0.6),
        "support_reward": bool(l.get("assigned_support", 0) is not None
                               and int(l.get("assigned_support", 0)) >= 8
                               and float(l.get("best_novel_similarity", 1.0)) < 0.55),
        "low_margin_forced": bool(l.get("novel_margin_used", 1.0) is not None
                                  and float(l.get("novel_margin_used", 1.0)) < 0.02),
        "other": True,
    }
    return f


def primary_category(flags):
    order = ["same_class_proto_unselected", "hub_pollution", "radius_wide",
             "support_reward", "low_margin_forced", "other"]
    for k in order:
        if flags.get(k):
            return k
    return "other"


def _num(v, default=float("nan")):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_rows(logs_by_cfg):
    wrong_rows, first_rows, proto_rows = [], [], []
    corr_accum = defaultdict(list)

    for (model, stream), logs in logs_by_cfg.items():
        routed = [l for l in logs if l["role"] == "novel"
                  and l["predicted_action"] != "KNOWN"]
        exist = [l for l in routed if l["predicted_action"] == "EXISTING_NOVEL"]
        wrong = [l for l in exist
                 if int(l["class"]) != int(l.get("assigned_primary_class", -1))]
        first = [l for l in logs if l["role"] == "novel" and l["first_occurrence"]]
        first_merge = [l for l in first
                       if l["predicted_action"] == "EXISTING_NOVEL"]

        for l in wrong:
            fl = error_flags(l)
            row = dict(l)
            row.update({
                "model": model, "stream": stream,
                "primary_category": primary_category(fl),
            })
            for k, v in fl.items():
                row[f"flag_{k}"] = int(v)
            wrong_rows.append(row)
        for l in first_merge:
            fl = error_flags(l)
            row = dict(l)
            row.update({
                "model": model, "stream": stream,
                "primary_category": primary_category(fl),
            })
            for k, v in fl.items():
                row[f"flag_{k}"] = int(v)
            first_rows.append(row)

        # prototype-level confidence audit
        by_vid = defaultdict(list)
        for l in logs:
            vid = l.get("predicted_virtual_novel_id")
            if vid is not None:
                by_vid[int(vid)].append(l)
        for vid, ls in by_vid.items():
            last = ls[-1]
            classes = [int(l["class"]) for l in ls]
            cnt = Counter(classes)
            primary = cnt.most_common(1)[0][0]
            purity = cnt[primary] / max(len(classes), 1)
            n_wrong = sum(1 for l in ls
                          if l["role"] == "novel"
                          and l.get("assigned_primary_class") is not None
                          and int(l["class"]) != int(l["assigned_primary_class"]))
            n_first = sum(1 for l in ls if l["first_occurrence"]
                          and l["predicted_action"] == "EXISTING_NOVEL")
            distinct = len({int(l.get("assigned_primary_class", -1))
                            for l in ls
                            if l.get("assigned_primary_class") is not None})
            proto_rows.append({
                "model": model, "stream": stream,
                "virtual_id": vid,
                "support": last.get("assigned_support", 0),
                "radius": last.get("assigned_radius", float("nan")),
                "dispersion": last.get("assigned_dispersion", float("nan")),
                "mean_margin": last.get("assigned_mean_margin", float("nan")),
                "min_margin": last.get("assigned_min_margin", float("nan")),
                "low_margin_count": last.get("assigned_low_margin_count", 0),
                "recent_stability": last.get("assigned_recent_stability", float("nan")),
                "age": last.get("assigned_age", 0),
                "conf_legal": last.get("assigned_conf_legal", float("nan")),
                "distinct_classes": distinct,
                "purity_offline": purity,
                "wrong_existing_caused": n_wrong,
                "first_merge_caused": n_first,
                "primary_class_offline": primary,
            })

        # correlations per model/stream
        feats = ["support", "dispersion", "mean_margin", "min_margin",
                 "low_margin_count", "recent_stability", "age", "conf_legal"]
        targets = ["purity_offline", "wrong_existing_caused", "first_merge_caused"]
        for f in feats:
            for t in targets:
                pairs = [(r[f], r[t]) for r in proto_rows
                         if r["model"] == model and r["stream"] == stream
                         and not (isinstance(r[f], float) and math.isnan(r[f]))
                         and r[f] is not None]
                if len(pairs) < 5:
                    continue
                xs = np.asarray([p[0] for p in pairs], dtype=np.float64)
                ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
                if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
                    continue
                pr = np.corrcoef(xs, ys)[0, 1]
                # Spearman
                def rank(a):
                    order = np.argsort(a)
                    rk = np.empty_like(order, dtype=np.float64)
                    rk[order] = np.arange(1, len(a) + 1)
                    return rk
                sr = np.corrcoef(rank(xs), rank(ys))[0, 1]
                corr_accum[(model, stream)].append({
                    "feature": f, "target": t,
                    "pearson_r": pr, "spearman_r": sr, "n": len(pairs),
                })

        print(f"[{model}/{stream}] routed={len(routed)} exist={len(exist)} "
              f"wrong={len(wrong)} first_merge={len(first_merge)} "
              f"protos={len(by_vid)}", flush=True)
    return wrong_rows, first_rows, proto_rows, corr_accum


def bucket_tables(wrong_rows, routed_counts):
    def table(key, bins, label):
        rows = []
        for model, stream in CONFIGS:
            for b in bins:
                sel = [r for r in wrong_rows
                       if r["model"] == model and r["stream"] == stream
                       and b[0] <= _num(r.get(key), -1) < b[1]]
                den = routed_counts[(model, stream, key, tuple(b[:2]))]
                rows.append({
                    "model": model, "stream": stream, label: b[2],
                    "wrong_existing": len(sel), "routed_novel": den,
                    "wrong_existing_rate": len(sel) / max(den, 1),
                })
        return rows

    mem_bins = [(0, 33, "0-32"), (33, 129, "33-128"),
                (129, 257, "129-256"), (257, 10**9, "257+")]
    sup_bins = [(0, 1, "1"), (1, 5, "2-4"), (5, 10, "5-9"),
                (10, 20, "10-19"), (20, 10**9, "20+")]
    mar_bins = [(-1e-9, 0.01, "<0.01"), (0.01, 0.02, "0.01-0.02"),
                (0.02, 0.05, "0.02-0.05"), (0.05, 0.1, "0.05-0.10"),
                (0.1, 1.0, ">0.10")]
    return (table("active_novel_prototypes", mem_bins, "memory_bucket"),
            table("assigned_support", sup_bins, "support_bucket"),
            table("novel_margin_used", mar_bins, "margin_bucket"))


def main():
    logs_by_cfg = load_logs()
    wrong_rows, first_rows, proto_rows, corr_accum = build_rows(logs_by_cfg)

    routed_counts = {}
    for model, stream in CONFIGS:
        logs = logs_by_cfg[(model, stream)]
        routed = [l for l in logs if l["role"] == "novel"
                  and l["predicted_action"] != "KNOWN"]
        mem_bins = [(0, 33), (33, 129), (129, 257), (257, 10**9)]
        sup_bins = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 10**9)]
        mar_bins = [(-1e-9, 0.01), (0.01, 0.02), (0.02, 0.05),
                    (0.05, 0.1), (0.1, 1.0)]
        for key, bins in [("active_novel_prototypes", mem_bins),
                          ("assigned_support", sup_bins),
                          ("novel_margin_used", mar_bins)]:
            for b in bins:
                routed_counts[(model, stream, key, tuple(b))] = sum(
                    1 for l in routed if b[0] <= _num(l.get(key), -1) < b[1])

    write_csv(AUDIT / "wrong_existing_assignments.csv", wrong_rows)
    write_csv(AUDIT / "first_occurrence_false_merge.csv", first_rows)
    write_csv(AUDIT / "prototype_confidence_analysis.csv", proto_rows)

    ms, sp, mg = bucket_tables(wrong_rows, routed_counts)
    write_csv(AUDIT / "wrong_existing_by_memory_scale.csv", ms)
    write_csv(AUDIT / "wrong_existing_by_prototype_support.csv", sp)
    write_csv(AUDIT / "wrong_existing_by_margin.csv", mg)

    hubs = []
    for model, stream in CONFIGS:
        acc = {}
        for l in first_rows:
            if l["model"] != model or l["stream"] != stream:
                continue
            vid = l["predicted_virtual_novel_id"]
            h = acc.setdefault(vid, {"first_merge_count": 0, "classes": set(),
                                     "supports": [], "cons": [], "disps": [],
                                     "radii": []})
            h["first_merge_count"] += 1
            h["classes"].add(l["class"])
            h["supports"].append(_num(l.get("assigned_support"), 0))
            h["cons"].append(_num(l.get("assigned_consistency"), 0))
            h["disps"].append(_num(l.get("assigned_dispersion"), 0))
            h["radii"].append(_num(l.get("assigned_radius"), 0))
        for vid, h in acc.items():
            hubs.append({
                "model": model, "stream": stream, "prototype": vid,
                "first_merge_count": h["first_merge_count"],
                "absorbed_class_count": len(h["classes"]),
                "absorbed_classes": ",".join(str(c) for c in sorted(h["classes"])),
                "mean_support": float(np.mean(h["supports"])),
                "mean_consistency": float(np.mean(h["cons"])),
                "mean_dispersion": float(np.mean(h["disps"])),
                "mean_radius": float(np.mean(h["radii"])),
            })
    write_csv(AUDIT / "false_merge_by_hub_prototype.csv", hubs)

    corr_rows = []
    for (model, stream), rows in corr_accum.items():
        for r in rows:
            corr_rows.append({"model": model, "stream": stream, **r})
    write_csv(AUDIT / "confidence_purity_correlation.csv", corr_rows)

    summary = {
        "wrong_existing_rows": len(wrong_rows),
        "first_merge_rows": len(first_rows),
        "prototype_rows": len(proto_rows),
        "correlation_rows": len(corr_rows),
        "hub_rows": len(hubs),
    }
    (AUDIT / "audit_merge_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
