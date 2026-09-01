#!/usr/bin/env python3
"""Summarize per-class TrackEval files for Phase71 without rerunning metrics."""
from __future__ import annotations
import json, pathlib, statistics, tempfile, os

ROOT = pathlib.Path(__file__).resolve().parents[2]

def atomic_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def read_summary(path: pathlib.Path):
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2: return None
    header = lines[-2].split(); values = lines[-1].split()
    out = {}
    for key, value in zip(header, values):
        try: out[key] = float(value)
        except ValueError: out[key] = None
    return out

def one_fold(src: pathlib.Path):
    rows = []
    for path in sorted(src.glob("*_summary.txt")):
        row = read_summary(path)
        if row is not None: rows.append(row)
    metrics = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "LocA", "OWTA", "CLR_Re", "CLR_Pr", "IDSW", "Frag"]
    macro = {}
    for metric in metrics:
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        macro[metric] = statistics.fmean(vals) if vals else None
    sum_metrics = ["CLR_TP", "CLR_FN", "CLR_FP", "IDTP", "IDFN", "IDFP", "Dets", "GT_Dets", "IDs", "GT_IDs", "IDSW", "Frag"]
    sums = {m: sum((r.get(m) or 0.0) for r in rows) for m in sum_metrics}
    dets, gt = sums["Dets"], sums["GT_Dets"]
    weighted = {
        "CLR_Re": sums["CLR_TP"] / gt if gt else 0.0,
        "CLR_Pr": sums["CLR_TP"] / dets if dets else 0.0,
        "IDR": sums["IDTP"] / (sums["IDTP"] + sums["IDFN"]) if sums["IDTP"] + sums["IDFN"] else 0.0,
        "IDP": sums["IDTP"] / (sums["IDTP"] + sums["IDFP"]) if sums["IDTP"] + sums["IDFP"] else 0.0,
    }
    return {"class_summary_count": len(rows), "macro": macro, "count_sums": sums, "count_weighted": weighted, "source_dir": str(src.resolve())}

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="formal1_tco_serial"); args = ap.parse_args()
    root = ROOT / "outputs/iclr27_phase71/metrics" / args.tag / "trackeval"
    out = {"protocol": "trackocd_phase71_trackeval_per_class_aggregate", "tag": args.tag, "folds": {}}
    for fold in range(4): out["folds"][str(fold)] = one_fold(root / f"fold_{fold}")
    for metric in ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "LocA", "OWTA", "CLR_Re", "CLR_Pr", "IDSW", "Frag"]:
        vals = [out["folds"][str(f)]["macro"].get(metric) for f in range(4)]
        vals = [v for v in vals if v is not None]
        out.setdefault("aggregate", {}).setdefault("macro_mean_over_folds", {})[metric] = statistics.fmean(vals) if vals else None
    atomic_json(ROOT / "outputs/iclr27_phase71/metrics" / args.tag / "trackeval_aggregate.json", out)
    print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
