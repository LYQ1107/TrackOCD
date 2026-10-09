#!/usr/bin/env python3
"""Summarize measured interventions, lock a variant and audit full-run gates."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.baselines.pandas_bytetrack.diagnostics import atomic_json


def row(root, subset, name):
    folder = root/subset/name/"eval"
    d = json.loads((folder/"diagnostics.json").read_text())
    m = json.loads((folder/"trackeval_summary.json").read_text())
    if d["status"] != "PASS" or m["status"] != "PASS":
        raise ValueError("diagnostics or evaluator failed")
    return {"variant":name,"diagnostics":d,"metrics":m["metrics"],
            "clear_counts":m["raw_combined"]["CLEAR"],"trackeval_count":m["raw_combined"]["Count"]}


def recall_losses(value):
    recalls = value["diagnostics"]["proposal_recall_iou05"]
    return {k:recalls["before"][k]["recall"]-recalls["after"][k]["recall"] for k in ["all","base","novel"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--stage",choices=["calibration","confirmation","full"],required=True)
    args=p.parse_args(); root=args.output_root
    if args.stage == "calibration":
        rows=[row(root,"calibration",n) for n in ["Original-FG-0","Dedup-A","Dedup-B","Dedup-C"]]
        original=rows[0]
        for r in rows[1:]:
            if r["diagnostics"]["gt_export_sha256"]!=original["diagnostics"]["gt_export_sha256"]:
                raise ValueError("comparison GT differs")
            r["recall_losses"] = recall_losses(r)
            r["recall_eligible"] = all(v is not None and v<=.02 for v in r["recall_losses"].values())
            r["metric_delta"]={k:r["metrics"][k]-original["metrics"][k] for k in r["metrics"]}
        eligible=[r for r in rows[1:] if r["recall_eligible"]]
        choice=max(eligible,key=lambda r:r["metrics"]["HOTA_mean"]) if eligible else min(rows[1:],key=lambda r:r["recall_losses"]["all"])
        locked={"variant":choice["variant"],"nms_config":json.loads((root/"selection.json").read_text())["variants"][choice["variant"]],
                "selection_is_tao_val_development":True,"any_recall_eligible":bool(eligible),
                "calibration_hota":choice["metrics"]["HOTA_mean"],"calibration_recall_losses":choice["recall_losses"]}
        path=root/"locked_variant.json"
        if path.exists() and json.loads(path.read_text()) != locked:
            raise ValueError("locked variant changed")
        atomic_json(path,locked)
        atomic_json(root/"calibration_results.json",{"status":"PASS","rows":rows,"locked_variant":locked})
        print(json.dumps(locked),flush=True)
    elif args.stage == "confirmation":
        lock=json.loads((root/"locked_variant.json").read_text()); name=lock["variant"]
        old,new=row(root,"confirmation","Original-FG-0"),row(root,"confirmation",name)
        same_gt=old["diagnostics"]["gt_export_sha256"]==new["diagnostics"]["gt_export_sha256"]
        losses=recall_losses(new)
        gain_hota=new["metrics"]["HOTA_mean"]-old["metrics"]["HOTA_mean"]
        gain_deta=new["metrics"]["DetA_mean"]-old["metrics"]["DetA_mean"]
        audit=json.loads((root/"audit.json").read_text())
        gate={"duplicated_boxes_supported":audit["exact_geometry"]["duplicate_rows_beyond_first"]>0,
              "candidate_reduction":1-new["diagnostics"]["retained_fraction"]>=.2,
              "calibration_recall_safe":lock["any_recall_eligible"] and all(v<=.02 for v in lock["calibration_recall_losses"].values()),
              "confirmation_recall_safe":all(v is not None and v<=.02 for v in losses.values()),
              "hota_improved":gain_hota>=.005,"deta_improved":gain_deta>=.002,
              "same_gt_and_pinned_evaluator":same_gt}
        result={"status":"PASS","locked_variant":lock,"rows":[old,new],"recall_losses":losses,
                "hota_gain":gain_hota,"deta_gain":gain_deta,"gates":gate,"run_full":all(gate.values())}
        atomic_json(root/"confirmation_results.json",result)
        print(json.dumps({k:result[k] for k in ["recall_losses","hota_gain","deta_gain","gates","run_full"]}),flush=True)
    else:
        conf=json.loads((root/"confirmation_results.json").read_text())
        if not conf["run_full"]:
            raise ValueError("full-run gates did not pass")
        name=conf["locked_variant"]["variant"]
        new=row(root,"full",name)
        old=json.loads(Path("/data3/liuyeqiang/pandas_bytetrack_tao_val/outputs/score_fix_full/trackeval/BT-FG-0/trackeval_summary.json").read_text())
        atomic_json(root/"full_results.json",{"status":"PASS","new":new,"original_fg0":old,
                    "metric_delta":{k:new["metrics"][k]-old["metrics"][k] for k in new["metrics"]}})
        print(json.dumps(new["metrics"]),flush=True)


if __name__=="__main__":
    main()
