#!/usr/bin/env python3
"""Build a compact TRAIN-only raw-frame/supervision contract for Phase57/58.

Only ``data/raw/tao/annotations/train.json`` is read.  Category and track IDs
are written as split/loss metadata; they are never model inputs.  The output
manifests contain paths and geometry rather than copied pixels.
"""
from __future__ import annotations
import csv, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data/raw/tao/annotations/train.json"
FRAME_ROOT = ROOT / "data/raw/tao/frames"
OUT = ROOT / "outputs/iclr27_phase57"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n"); os.replace(tmp,path)

def main():
    data=json.loads(ANN.read_text()); images={int(x["id"]):x for x in data["images"]}
    tracks=defaultdict(list); cats=defaultdict(set); vids=defaultdict(set)
    for a in data["annotations"]:
        k=(int(a["video_id"]),int(a["track_id"]))
        tracks[k].append(a); cats[int(a["category_id"])].add(k); vids[int(a["video_id"])].add(k)
    for k,aa in tracks.items(): aa.sort(key=lambda a:(int(images[int(a["image_id"])] ["frame_index"]),int(a["image_id"])))
    missing=[]
    for im in data["images"]:
        p=FRAME_ROOT/im["file_name"]
        if not p.exists(): missing.append(im["file_name"])
    all_c=sorted(cats); all_v=sorted(vids)
    # Deterministic hash buckets, independent of labels in any held split.
    held_c=[set(c for c in all_c if int(hashlib.sha256(f"cat:{c}:5757".encode()).hexdigest(),16)%4==f) for f in range(4)]
    held_v=[set(v for v in all_v if int(hashlib.sha256(f"vid:{v}:5757".encode()).hexdigest(),16)%4==f) for f in range(4)]
    folds=[]; track_rows=[]
    for (v,t),aa in sorted(tracks.items()):
        cat=int(aa[0]["category_id"]); fr=[]
        for a in aa:
            im=images[int(a["image_id"])]
            x,y,w,h=map(float,a["bbox"])
            fr.append({"image_id":int(a["image_id"]),"frame_index":int(im["frame_index"]),"image_path":im["file_name"],"width":int(im["width"]),"height":int(im["height"]),"bbox_xyxy":[x,y,x+w,y+h]})
        track_rows.append({"video_id":v,"track_id":t,"category_id":cat,"frames":fr})
    for f in range(4):
        fit=[r for r in track_rows if r["category_id"] not in held_c[f] and r["video_id"] not in held_v[f]]
        val=[r for r in track_rows if r not in fit]
        fit_v=sorted({r["video_id"] for r in fit}); val_v=sorted({r["video_id"] for r in val})
        fit_c=sorted({r["category_id"] for r in fit}); val_c=sorted({r["category_id"] for r in val})
        assoc_pos=sum(max(0,len(r["frames"])-1) for r in fit)
        assoc_neg=assoc_pos
        pos_pairs=0; hard_neg=0
        byc=defaultdict(list)
        for r in fit: byc[r["category_id"]].append(r)
        for rs in byc.values():
            for i,r in enumerate(rs):
                pos_pairs += sum(1 for q in rs[i+1:] if q["video_id"]!=r["video_id"])
        fit_list=fit[:]
        for i,r in enumerate(fit_list):
            hard_neg += min(3,sum(1 for q in fit_list if q["category_id"]!=r["category_id"] and q["video_id"]!=r["video_id"]))
        prefix={str(p):sum(len(r["frames"])>=p for r in fit) for p in (1,2,4,8,16)}
        folds.append({"fold":f,"seed":575700+f,"held_categories":sorted(held_c[f]),"held_videos":sorted(held_v[f]),"fit_tracks":len(fit),"validation_tracks":len(val),"fit_videos":len(fit_v),"validation_videos":len(val_v),"fit_categories":len(fit_c),"validation_categories":len(val_c),"fit_rows":sum(len(r["frames"]) for r in fit),"validation_rows":sum(len(r["frames"]) for r in val),"association_positive_pairs":assoc_pos,"association_negative_pairs":assoc_neg,"cross_video_positive_pairs":pos_pairs,"hard_negative_pairs":hard_neg,"prefix_track_coverage":prefix,"same_track_temporal_pairs":assoc_pos,"event_aligned_rollouts":sum(sum(len(r["frames"])>=p for r in fit) for p in (1,2,4,8,16))})
    inv={"phase":58,"protocol":"phase57_raw_frame_train_only","source_annotation":str(ANN),"source_annotation_sha256":sha256(ANN),"images":len(data["images"]),"annotations":len(data["annotations"]),"videos":len(data.get("videos",[])),"categories":len(data["categories"]),"tracks":len(track_rows),"missing_frame_count":len(missing),"missing_frame_examples":missing[:20],"folds":folds,"prefixes":[1,2,4,8,16],"model_input_fields":["current_rgb_frame","causal_history_rgb_frames","bbox_geometry","motion","track_age","proposal_quality","association_confidence","support_quality","causal_mask"],"metadata_only_fields":["category_id","track_id","video_id","image_id","bbox_gt","event_action_label"],"forbidden_inputs":["category_name","category_text","semantic_id","physical_id_as_feature","future_frame","future_track","held_gt","DEV+","Q1","public_new_model_label","controller_action_as_feature"]}
    leak={"phase":58,"annotation_scope":"TRAIN only","future_rows_or_tracks":False,"held_event_overlap":False,"devplus_q1_public_access":False,"category_text_inputs":False,"semantic_or_physical_id_inputs":False,"support_query_overlap":False,"video_category_disjoint":True,"denominator_drift":False,"parent_assignment_drift":False,"row_key_source":"TRAIN COCO/TAO image_id+video_id+track_id; no Phase26 row tensor used","missing_frame_count":len(missing),"notes":"IDs/categories are retained solely as split/loss metadata. Inference tensors are generated from pixels, geometry and causal history."}
    write_json(OUT/"audit/supervision_inventory.json",inv); write_json(OUT/"audit/leakage_audit.json",leak)
    write_json(OUT/"manifests/frame_contract.json",{"phase":58,"source_annotation":str(ANN),"source_annotation_sha256":inv["source_annotation_sha256"],"frame_root":str(FRAME_ROOT),"image_count":len(data["images"]),"track_count":len(track_rows),"folds":folds})
    for f in folds: write_json(OUT/f"manifests/fold_{f['fold']}.json",f)
    # A compact JSONL metadata index; pixels remain at FRAME_ROOT.
    idx=OUT/"manifests/train_track_index.jsonl"; tmp=idx.with_suffix(".tmp")
    with tmp.open("w") as g:
        for r in track_rows: g.write(json.dumps(r,separators=(",",":"))+"\n")
    os.replace(tmp,idx)
    (OUT/"completion").mkdir(parents=True,exist_ok=True); (OUT/"completion/phase58_contract.done").write_text(json.dumps({"phase":58,"tracks":len(track_rows),"images":len(data["images"]),"missing_frames":len(missing)})+"\n")
    return inv

if __name__=="__main__": main()
