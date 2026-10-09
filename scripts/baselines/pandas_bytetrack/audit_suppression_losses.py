#!/usr/bin/env python3
"""Evaluator-only decomposition of fixed NMS losses versus the candidate cap.

No new tracker run, parameter setting, inference or selected operating point.
Inspect only the three preregistered postprocessors on annotated cached frames.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.baselines.pandas_bytetrack.diagnostics import atomic_json,box_iou,frame_rows,gt_arrays,load_evaluator,read_video
from src.baselines.pandas_bytetrack.tracking_postprocess import prepare_tracking_detections


def hit(iou):
    return iou.max(axis=1)>=.5 if iou.shape[1] else np.zeros(iou.shape[0],bool)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--asset-root",type=Path,required=True)
    p.add_argument("--annotation",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    args=p.parse_args(); torch.set_num_threads(1)
    selection=json.loads((args.output_root/"selection.json").read_text())
    lock=json.loads((args.output_root/"locked_variant.json").read_text())
    _,anns,roles=load_evaluator(args.annotation,args.asset_root/"audit/tao_val_protocol.json")
    results={}
    for subset,names in [("calibration",list(selection["variants"])),("confirmation",[lock["variant"]])]:
        stats={n:Counter() for n in names}; examples={n:[] for n in names}
        for vid in selection[subset+"_ids"]:
            d=read_video(args.asset_root/f"outputs/score_fix_full/anonymous/full/video_{vid:04d}.npz")
            for pos,image_id in enumerate(d["image_id"]):
                if image_id<0: continue
                a,b,boxes=frame_rows(d,pos)
                fg,sem,proto=d["foreground_scores"][a:b],d["scores"][a:b],d["prototype_id"][a:b]
                gt,groles=gt_arrays(anns[int(image_id)],roles)
                before=box_iou(gt,boxes); covered=hit(before)
                for name in names:
                    config=selection["variants"][name]
                    no_cap=prepare_tracking_detections(boxes,fg,sem,proto,nms_iou=config["nms_iou"],max_detections=None)
                    indices=no_cap["original_detection_index"]
                    after_nms=box_iou(gt,no_cap["boxes"])
                    after_cap=after_nms[:,:config["max_detections"]]
                    hn,hc=hit(after_nms),hit(after_cap)
                    stat=stats[name]
                    stat["gt_total"]+=len(gt)
                    stat["before_covered"]+=int(covered.sum())
                    stat["after_nms_covered"]+=int(hn.sum())
                    stat["after_cap_covered"]+=int(hc.sum())
                    for role in ["base","novel","distractor"]:
                        mask=groles==role
                        stat["nms_lost_"+role]+=int((covered & ~hn & mask).sum())
                        stat["cap_lost_"+role]+=int((hn & ~hc & mask).sum())
                    for j in np.where(covered & ~hn)[0]:
                        removed=np.where(before[j]>=.5)[0]
                        suppressors=box_iou(boxes[removed],no_cap["boxes"])
                        can_suppress=np.any(suppressors>config["nms_iou"],axis=0)
                        different_gt=np.any(after_nms[np.arange(len(gt))!=j]>=.5,axis=0) if len(gt)>1 else np.zeros(len(indices),bool)
                        collision=bool(np.any(can_suppress & different_gt))
                        stat["nms_loss_with_suppressor_covering_other_gt"]+=int(collision)
                        if len(examples[name])<12:
                            examples[name].append({"image_id":int(image_id),"annotation_id":anns[int(image_id)][j]["id"],
                                                   "role":str(groles[j]),"suppressor_covers_different_gt":collision,
                                                   "best_iou_before":float(before[j].max()),
                                                   "best_iou_after_nms":float(after_nms[j].max()) if len(indices) else 0.})
            print(json.dumps({"subset":subset,"video_id":vid}),flush=True)
        results[subset]={n:{"counts":dict(stats[n]),"examples":examples[n]} for n in names}
    atomic_json(args.output_root/"suppression_loss_audit.json",{"status":"PASS","results":results,
        "interpretation":"Only GT-covered candidate losses are decomposed. Suppressor covering a different GT supports adjacent-object collision; absence does not prove no unannotated-object collision. Exact duplicate removal alone cannot change geometric coverage."})
    print(json.dumps({s:{n:d["counts"] for n,d in v.items()} for s,v in results.items()}),flush=True)


if __name__=="__main__":main()
