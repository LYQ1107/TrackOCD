#!/usr/bin/env python3
"""Summarize the pinned TrackEval per-class TAO output without rerunning it.

TrackEval's TAO adapter emits one summary per class.  This script preserves
those values and computes transparent macro and count-weighted aggregates;
the aggregate is a reporting aid, not a replacement for the official files.
"""
from __future__ import annotations
import csv, json, pathlib, statistics, tempfile, os

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval/OVTR_Q0"
OUT = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval_aggregate.json"

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def read_summary(path):
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2: return None
    h, v = lines[-2].split(), lines[-1].split()
    d = {}
    for k, x in zip(h, v):
        try: d[k] = float(x)
        except ValueError: d[k] = None
    return d

def main():
    rows=[]
    for p in sorted(SRC.glob("*_summary.txt")):
        d=read_summary(p)
        if d is not None:
            d["class_file"]=p.name
            rows.append(d)
    metrics=["HOTA","DetA","AssA","MOTA","IDF1","LocA","OWTA","CLR_Re","CLR_Pr","IDSW","Frag"]
    macro={}
    for m in metrics:
        vals=[r[m] for r in rows if r.get(m) is not None]
        macro[m]=statistics.fmean(vals) if vals else None
    # Count-weighted detection/identity quantities from the exact class files.
    sums={m:sum((r.get(m) or 0.0) for r in rows) for m in ["CLR_TP","CLR_FN","CLR_FP","IDTP","IDFN","IDFP","Dets","GT_Dets","IDs","GT_IDs","IDSW","Frag"]}
    gt=sums["GT_Dets"]
    det=sums["Dets"]
    count_weighted={
        "CLR_Re": sums["CLR_TP"]/gt if gt else 0.0,
        "CLR_Pr": sums["CLR_TP"]/det if det else 0.0,
        "IDR": sums["IDTP"]/(sums["IDTP"]+sums["IDFN"]) if (sums["IDTP"]+sums["IDFN"]) else 0.0,
        "IDP": sums["IDTP"]/(sums["IDTP"]+sums["IDFP"]) if (sums["IDTP"]+sums["IDFP"]) else 0.0,
    }
    obj={"protocol":"trackocd_phase68_trackeval_tao_per_class_aggregate",
         "source_dir":str(SRC),"class_summary_count":len(rows),"macro":macro,
         "count_sums":sums,"count_weighted_detection_identity":count_weighted,
         "note":"TrackEval TAO emits per-class files in this run; macro values are unweighted class means and count sums preserve exact class summaries. No rerun or label access."}
    atomic_json(OUT,obj)
    print(json.dumps(obj,indent=2))
if __name__ == '__main__': main()
