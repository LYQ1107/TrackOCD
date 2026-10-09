#!/usr/bin/env python3
"""Evaluate sealed Val4 clips with the unchanged canonical TAO_OW adapter."""
from __future__ import annotations
from collections import defaultdict
import copy
import hashlib
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
from scripts.trackocd_core.audit_frontend import TRACK_ROOT,length_statistics
from scripts.trackocd_core.audit_frontend_coverage import match_video
from src.trackocd_core.physical_diagnostic import observed_purity,pairwise_iou

RUN=ROOT/"outputs/trackocd_core/physical/masa_val4_first16"
CANONICAL=Path("/data3/liuyeqiang/InterMOT/third_party/MOTIP/TrackEval")


def read_frames(path:Path,images:list[dict])->dict:
    selected={im["image_id"] for im in images}
    frames={}
    with np.load(path,allow_pickle=False) as arrays:
        for i,raw in enumerate(arrays["image_id"]):
            image_id=int(raw)
            if image_id not in selected:
                continue
            if image_id in frames:
                raise ValueError("Duplicate canonical image in physical stream")
            begin,end=map(int,arrays["frame_offsets"][i:i+2])
            ids=np.asarray(arrays["track_id"][begin:end],dtype=np.int64)
            boxes=np.asarray(arrays["boxes"][begin:end],dtype=np.float32)
            scores=np.asarray(arrays["score"][begin:end],dtype=np.float32)
            if (len(np.unique(ids))!=len(ids) or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
                    or np.any(boxes[:,2:]<=boxes[:,:2]) or np.any((scores<0)|(scores>1))):
                raise ValueError("Invalid sealed physical stream; do not repair/filter")
            frames[image_id]=(ids,boxes,scores)
    if set(frames)!=selected:
        raise ValueError("Missing selected canonical frame; do not shrink GT universe")
    return frames


def main()->int:
    started=time.monotonic()
    prediction_path=RUN/"prediction_manifest.json"
    prediction=json.loads(prediction_path.read_text())
    if prediction["status"]!="SEALED_BOUNDED_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION" or prediction["real_image_forwards_started"]!=64:
        raise ValueError("Complete sealed bounded stream required before GT evaluation")
    private=ROOT/"outputs/trackocd_core/audit/masa_physical_diagnostic_plan.json"
    config_path=ROOT/"configs/trackocd_core/masa_physical_diagnostic.json"
    config=json.loads(config_path.read_text()); plan=json.loads(private.read_text())
    if sha256_file(private)!=config["private_plan_sha256"] or sha256_file(config_path)!=prediction["config_sha256"]:
        raise ValueError("Fixed plans changed after prediction")
    for video in prediction["videos"]:
        if sha256_file(RUN/video["npz_filename"])!=video["npz_sha256"]:
            raise ValueError("Prediction payload changed before evaluation")
    # First GT/role access in this process occurs only after sealed-prediction checks.
    annotation=Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json")
    if sha256_file(annotation)!=config["annotation_sha256"]:
        raise ValueError("Registered GT source changed")
    gt=json.loads(annotation.read_text())
    roles_path=ROOT/"configs/trackocd_core/roles.json"
    roles=json.loads(roles_path.read_text())
    known,novel=set(roles["known_ids"]),set(roles["novel_ids"])
    selected_images={im["image_id"] for v in plan["videos"] for im in v["images"]}
    selected_videos=set(config["video_ids"])
    subset=copy.deepcopy(gt)
    subset["images"]=[im for im in subset["images"] if int(im["id"]) in selected_images]
    subset["videos"]=[v for v in subset["videos"] if int(v["id"]) in selected_videos]
    subset["annotations"]=[a for a in subset["annotations"] if int(a["image_id"]) in selected_images]
    track_ids={int(a["track_id"]) for a in subset["annotations"]}
    subset["tracks"]=[t for t in subset["tracks"] if int(t["id"]) in track_ids]
    gt_folder=RUN/"canonical_gt"; gt_folder.mkdir()
    atomic_json(gt_folder/"validation.json",subset)
    # Process-local compatibility aliases; never patch third-party files/base env.
    if not hasattr(np,"float"): np.float=float
    if not hasattr(np,"int"): np.int=int
    sys.path.insert(0,str(CANONICAL))
    import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(CANONICAL.resolve()):
        raise ValueError("Wrong TrackEval imported; use fresh evaluator process")
    all_results={}; source_records={}
    gt_by_image=defaultdict(list)
    for ann in subset["annotations"]:
        gt_by_image[int(ann["image_id"])].append(ann)
    metric=trackeval.metrics.HOTA()
    for name in ("MASA_NATIVE","PANDAS_BT_FG0"):
        tracker_root=RUN/("canonical_"+name); data_folder=tracker_root/name/"data"; data_folder.mkdir(parents=True)
        rows=[]; frame_streams={}; lengths=[]; coverage_matches={}; gt_targets=[]; purity_rows=[]; sources=[]
        for video in plan["videos"]:
            video_id=video["video_id"]
            path=(RUN if name=="MASA_NATIVE" else TRACK_ROOT)/f"video_{video_id:04d}.npz"
            frames=read_frames(path,video["images"]); frame_streams[video_id]=frames
            sources.append({"video_id":video_id,"bytes":path.stat().st_size,"sha256":sha256_file(path)})
            ids=np.concatenate([frames[im["image_id"]][0] for im in video["images"]])
            _,counts=np.unique(ids,return_counts=True); lengths.extend(counts.tolist())
            grouped={}
            for image in video["images"]:
                image_id=image["image_id"]; local_ids,boxes,scores=frames[image_id]
                for local_id,box,score in zip(local_ids,boxes,scores):
                    x,y,x2,y2=map(float,box)
                    rows.append({"image_id":image_id,"video_id":video_id,"track_id":int(local_id),"bbox":[x,y,x2-x,y2-y],"score":float(score),"category_id":1})
                annotations=gt_by_image[image_id]
                gt_boxes=[]
                for ann in annotations:
                    x,y,w,h=map(float,ann["bbox"]); gt_boxes.append([x,y,x+w,y+h])
                    category=int(ann["category_id"])
                    if category in known|novel:
                        key=f"{video_id}_{ann['track_id']}"
                        target=grouped.setdefault(key,{"key":key,"category":category,"role":"known" if category in known else "novel","boxes":{}})
                        if target["category"]!=category: raise ValueError("Inconsistent GT category")
                        target["boxes"][image_id]=[x,y,x+w,y+h]
                purity_rows.append({"pred_ids":local_ids,"pred_boxes":boxes,"gt_ids":[int(a["track_id"]) for a in annotations],
                                    "gt_categories":[int(a["category_id"]) for a in annotations],"gt_boxes":gt_boxes,"video_id":video_id})
            matched=match_video(list(grouped.values()),{i:(row[0],row[1]) for i,row in frames.items()})
            coverage_matches.update(matched["matches"]); gt_targets.extend(grouped.values())
        atomic_json(data_folder/"pred.json",rows)
        dataset_config=trackeval.datasets.TAO_OW.get_default_dataset_config()
        dataset_config.update(GT_FOLDER=str(gt_folder),TRACKERS_FOLDER=str(tracker_root),TRACKERS_TO_EVAL=[name],
                              TRACKER_SUB_FOLDER="data",SPLIT_TO_EVAL="val",SUBSET="all",MAX_DETECTIONS=300,PRINT_CONFIG=False)
        dataset=trackeval.datasets.TAO_OW(dataset_config)
        per_sequence={}; per_video=[]; raw_count=preprocessed_count=0
        for seq in dataset.seq_list:
            raw=dataset.get_raw_seq_data(name,seq); data=dataset.get_preprocessed_seq_data(raw,"object")
            score=metric.eval_sequence(data); per_sequence[seq]=score
            raw_count+=sum(len(v) for v in raw["tracker_ids"]); preprocessed_count+=data["num_tracker_dets"]
            per_video.append({"video_id":int(dataset.seq_name_to_seq_id[seq]),"gt_rows":data["num_gt_dets"],"evaluated_pred_rows":data["num_tracker_dets"],
                              **{k:float(np.mean(score[k])) for k in ("HOTA","AssA","DetA","DetRe")}})
        combined=metric.combine_sequences(per_sequence)
        by_role={}
        for role in ("known","novel"):
            targets=[r for r in gt_targets if r["role"]==role]
            covered=sum(coverage_matches.get(r["key"],{}).get("reliable",False) for r in targets)
            by_role[role]={"gt_clip_tracks":len(targets),"reliably_observed":covered,"missing_or_unreliable":len(targets)-covered,"coverage":covered/len(targets) if targets else None}
        # Namespace physical IDs by video before a global posthoc purity aggregate.
        namespaced=[]; id_map={}
        for row in purity_rows:
            row=dict(row)
            row["pred_ids"]=[id_map.setdefault((row["video_id"],int(i)),len(id_map)) for i in row["pred_ids"]]
            row["gt_ids"]=[(row["video_id"]<<32)+int(i) for i in row["gt_ids"]]
            namespaced.append(row)
        length_result=length_statistics(np.asarray(lengths,dtype=np.int64))
        length_result.update(counts_come_from_full_frame_stream_not_only_annotated_frames=False,
                             scope="Only selected annotated-cadence clip projection, not full-video lifetime")
        all_results[name]={"canonical_tracking":{k:float(np.mean(combined[k])) for k in ("HOTA","AssA","DetA","DetRe")},
                           "canonical_raw_combined":{k:v.tolist() if hasattr(v,"tolist") else v for k,v in combined.items()},
                           "per_video":sorted(per_video,key=lambda r:r["video_id"]),"coverage":by_role,
                           "purity":observed_purity(namespaced),"annotated_projection_lengths":length_result,
                           "raw_prediction_rows":raw_count,"canonical_evaluated_prediction_rows":preprocessed_count,
                           "canonical_preprocessing_removed_rows":raw_count-preprocessed_count,"source_npz":sources}
    result={"schema_version":"trackocd.core.masa_physical_diagnostic_result.v1","status":"BOUNDED_DIAGNOSTIC_COMPLETE_NOT_M1_QUALIFICATION",
            "scope":config["scope"],"video_ids":config["video_ids"],"selected_images":len(selected_images),"selected_gt_rows":len(subset["annotations"]),
            "results":all_results,"prediction_manifest_sha256":sha256_file(prediction_path),"config_sha256":sha256_file(config_path),"roles_sha256":sha256_file(roles_path),
            "annotation_sha256":sha256_file(annotation),"canonical_adapter_sha256":sha256_file(CANONICAL/"trackeval/datasets/tao_ow.py"),
            "canonical_hota_sha256":sha256_file(CANONICAL/"trackeval/metrics/hota.py"),"canonical_gt_subset_sha256":sha256_file(gt_folder/"validation.json"),
            "cadence_and_history_boundary":config["sampling_boundary"],"provenance_boundary":config["provenance_boundary"],
            "label_use":"Independent evaluator after sealed predictions only; no returned mapping, resampling, thresholds or model updates",
            "not_full_val_or_main_result":True,"primary_freeze_permitted":False,
            "resources":{"cpu_workers":1,"gpu_used":False,"wall_seconds":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}}
    atomic_json(ROOT/"outputs/trackocd_core/audit/masa_physical_diagnostic_result.json",result)
    print(json.dumps({"status":result["status"],"results":{k:{f:v[f] for f in ("canonical_tracking","coverage","purity")} for k,v in all_results.items()},"resources":result["resources"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
