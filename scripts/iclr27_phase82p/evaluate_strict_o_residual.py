#!/usr/bin/env python3
"""Evaluate residual lineage under the frozen event-side contract.

This remains an event-only diagnostic: no held labels are fed to inference,
and the evaluator keeps the original 76 positive denominator.  The frozen
``assigned && transformed IoU`` row condition is combined with the residual
lineage's single-track temporal consistency; a remap can therefore improve a
fragmented track without changing any Q0 proposal box.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import os
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "scripts/iclr27_phase75b/run_observability.py"
EVENT_ROWS = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"); os.replace(tmp, path)


def iou(a: Any, b: Any) -> float:
    if a is None or b is None: return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]; bx0, by0, bx1, by1 = [float(v) for v in b]; ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1); inter=max(0.0,ix1-ix0)*max(0.0,iy1-iy0); aa=max(0.0,ax1-ax0)*max(0.0,ay1-ay0); ab=max(0.0,bx1-bx0)*max(0.0,by1-by0); den=aa+ab-inter; return inter/den if den>0 else 0.0


def load_old() -> Any:
    spec=importlib.util.spec_from_file_location("phase75b_readonly",OLD)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {OLD}")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def load_rows() -> dict[str, dict[str, Any]]:
    out={}
    with EVENT_ROWS.open(newline="",encoding="utf-8") as f:
        import csv
        for row in csv.DictReader(f):
            row["video_id"]=int(row["video_id"]); row["image_id"]=int(row["image_id"]); row["frame_id"]=int(row["frame_id"]); row["assigned"]=int(float(row.get("assigned") or 0)); row["row_iou"]=float(row.get("row_iou") or 0.0)
            row["gt_bbox_xyxy"]=ast.literal_eval(row["gt_bbox_xyxy"]) if row.get("gt_bbox_xyxy") else None
            out[str(row["row_key"])] = row
    return out


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--replay",type=Path,required=True); ap.add_argument("--tag",default="residual"); args=ap.parse_args()
    mod=load_old(); prepared,_=mod.load_event_rows(); csv_rows=load_rows(); by_image: dict[tuple[int,int],list[dict[str,Any]]] = collections.defaultdict(list)
    for line in args.replay.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line); by_image[(int(row["video_id"]),int(row["image_id"]))].append(row)
    event_records=[]
    for item in prepared:
        e=item["event"]
        if e["kind"] != "positive_existing": continue
        sides={}
        for name,source_rows in (("source",item["source_rows"]),("target",item["target_rows"])):
            prefix_rows=source_rows if name=="source" else source_rows[:16]
            details=[]; by_track: dict[int,list[float]]=collections.defaultdict(list); strict_rows=0
            for frozen in prefix_rows:
                gt=csv_rows[str(frozen["row_key"])]["gt_bbox_xyxy"]; candidates=by_image.get((int(frozen["video_id"]),int(frozen["image_id"])),[]); best=max(candidates,key=lambda r:iou(r.get("bbox_xyxy"),gt),default=None); score=iou(best.get("bbox_xyxy"),gt) if best else 0.0; row_ok=int(csv_rows[str(frozen["row_key"])].get("assigned",0))==1 and float(csv_rows[str(frozen["row_key"])].get("row_iou",0.0))>=0.5; strict_rows += int(row_ok and score>=0.5)
                if best is not None: by_track[int(best["physical_track_id"])].append(score)
                details.append({"row_key":frozen["row_key"],"best_physical_track_id":int(best["physical_track_id"]) if best else None,"residual_max_iou":score,"frozen_event_row_ok":row_ok})
            tracks=[{"track_id":tid,"rows":len(vals),"mean_iou":float(statistics.mean(vals)),"max_iou":max(vals),"coverage":len(vals)/max(1,len(prefix_rows))} for tid,vals in by_track.items()]
            best_track=max(tracks,key=lambda r:(r["coverage"]*r["mean_iou"],r["coverage"]),default={"track_id":None,"rows":0,"mean_iou":0.0,"max_iou":0.0,"coverage":0.0}); sides[name]={"rows":len(prefix_rows),"strict_reliable_rows":strict_rows,"tracks":tracks,"best_track":best_track,"reliable":bool(strict_rows>0 and best_track["mean_iou"]>=0.5),"details":details}
        event_records.append({"event_key":e["event_key"],"fold":int(e["fold"]),"source":sides["source"],"target":sides["target"],"both_reliable":sides["source"]["reliable"] and sides["target"]["reliable"]})
    by_fold={str(f):{"events":sum(r["fold"]==f for r in event_records),"both_reliable":sum(r["fold"]==f and r["both_reliable"] for r in event_records),"source_reliable":sum(r["fold"]==f and r["source"]["reliable"] for r in event_records),"target_reliable":sum(r["fold"]==f and r["target"]["reliable"] for r in event_records)} for f in range(4)}
    result={"schema_version":"trackocd.phase82p.strict_o_residual.v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"replay":str(args.replay),"replay_sha256":hashlib.sha256(args.replay.read_bytes()).hexdigest(),"event_count":len(event_records),"aggregate":{"both_reliable":sum(r["both_reliable"] for r in event_records),"source_reliable":sum(r["source"]["reliable"] for r in event_records),"target_reliable":sum(r["target"]["reliable"] for r in event_records),"by_fold":by_fold},"events":event_records,"contract":{"positive_denominator":76,"prefix":16,"frozen_row_rule":"assigned == 1 and transformed IoU >= 0.5","residual_track_rule":"best remapped track mean IoU >= 0.5 with at least one strict reliable row"},"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False}
    atomic_json(ROOT/"outputs/iclr27_phase82p/metrics"/f"strict_o_{args.tag}.json",result); print(json.dumps(result["aggregate"],indent=2,sort_keys=True))


if __name__=="__main__": main()
