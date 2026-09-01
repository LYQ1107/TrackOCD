"""Join frozen Phase19 raw predictions to fixed public labels for measurement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase19.evaluation.evaluate import event_metrics, load_events


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"); os.replace(tmp,path)


def score(raw: dict[str, Any]) -> dict[str, Any]:
    events={e["event_key"]:e for e in load_events()}; records=[]
    for base in raw["records"]:
        e=events[base["event_key"]]
        target_cat=int(e.get("category_gt_denominator_only",e.get("target_category_gt_denominator_only")))
        source_keys=set(e["source_tracklet_keys"]); target_key=e["target_tracklet_key"]
        state_cat={}
        for s in base.get("state_births",[]):
            state_cat[int(s["semantic_id"])] = target_cat if s.get("birth_track")==target_key else (int(e.get("category_gt_denominator_only",e.get("distractor_category_gt_denominator_only"))) if s.get("birth_track") in source_keys else None)
        def correct(d):
            if d["action"]!="EXISTING" or d.get("semantic_id") is None:return False
            sid=int(d["semantic_id"]); births={int(s["semantic_id"]):s for s in base.get("state_births",[])}; b=births.get(sid)
            return bool(state_cat.get(sid)==target_cat and b and int(b["birth_video"])!=int(e["target_video"]) and b["birth_track"]!=target_key)
        target=base["target_decisions"]; prefix=int(e["target_first_reliable_prefix_index_gt_only"]); post=target[prefix:]
        existing=[d for d in post if d["action"]=="EXISTING"]; first=next((d for d in post if d["action"]!="DEFER"),None)
        records.append({"event_key":base["event_key"],"kind":e["kind"],"fold":e.get("fold"),
                        "target_category_evaluator_only":target_cat,"source_decisions":base["source_decisions"],"target_decisions":target,
                        "first_commit_after_prefix":first,"first_commit_correct":bool(first and correct(first)),
                        "post_prefix_correct_rows":int(sum(correct(d) for d in post)),"post_prefix_rows":len(post),
                        "existing_correct_rows":int(sum(correct(d) for d in existing)),"existing_rows":len(existing),
                        "pre_prefix_rows":prefix,"pre_prefix_defer_rows":int(sum(d["action"]=="DEFER" for d in target[:prefix])),
                        "premature_commit":bool(any(d["action"]!="DEFER" for d in target[:prefix])),"unresolved":first is None,
                        "state_count":len(base.get("state_births",[])),"duplicate_target_births":int(sum(state_cat.get(int(s["semantic_id"]))==target_cat and int(s["birth_video"])==int(e["target_video"]) for s in base.get("state_births",[]))),
                        "target_video":int(e["target_video"]),"latency":next((i for i,d in enumerate(post) if correct(d)),None)})
    return {"protocol":"trackocd_iclr27_phase19_public_scored_after_freeze","candidate":raw["candidate"],"raw_prediction_sha256":raw.get("raw_prediction_sha256"),"metrics":event_metrics(records),"records":records}


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--raw",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    raw=json.loads(a.raw.read_text()); result=score(raw); atomic(a.out,result); print(json.dumps({"complete":True,"candidate":result["candidate"],"metrics":result["metrics"]},indent=2))

if __name__=="__main__":main()
