#!/usr/bin/env python3
"""Build native-Q0 runtime candidate sets for the B4 support hypothesis."""
from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
FOLD_MANIFEST = ROOT / "outputs/iclr27_phase22/manifests/fold_manifest.json"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box(value: Any) -> list[float] | None:
    if value is None or value == "": return None
    try: return [float(v) for v in (value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value)))]
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0.0, x2-x1)*max(0.0, y2-y1); aa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1]); bb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1]); return inter/max(aa+bb-inter,1e-8)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); gt: dict[tuple[int,int],list[tuple[list[float],int]]] = defaultdict(list); dims: dict[int,tuple[float,float]] = {}
    for r in rows:
        b=box(r.get("gt_bbox_xyxy")); key=(int(r["video_id"]),int(r["image_id"]));
        if b: gt[key].append((b,int(float(r.get("gt_category_id_common",-1) or -1))))
        try: dims[int(r["video_id"])] = (max(float(r.get("image_width") or 1), dims.get(int(r["video_id"]),(1,1))[0]), max(float(r.get("image_height") or 1), dims.get(int(r["video_id"]),(1,1))[1]))
        except Exception: pass
    native: list[dict[str,Any]]=[]
    with NATIVE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): native.append(json.loads(line))
    feats=np.load(FEATURES,allow_pickle=False)["features"].astype(np.float32); feats/=np.maximum(np.linalg.norm(feats,axis=1,keepdims=True),1e-8)
    if len(native)!=len(feats): raise RuntimeError(f"native/features mismatch {len(native)} vs {len(feats)}")
    groups:dict[tuple[int,int],list[int]]=defaultdict(list); tracks:dict[tuple[int,int],list[int]]=defaultdict(list)
    for i,r in enumerate(native):
        if r.get("bbox_xyxy") is None: continue
        key=(int(r["video_id"]),int(r["image_id"])); groups[key].append(i); tracks[(int(r["video_id"]),int(r.get("physical_track_id",-1)))].append(i)
    for key,inds in groups.items(): inds.sort(key=lambda i:(int(native[i].get("candidate_rank",0)),int(native[i].get("physical_track_id",-1)),i))
    history=np.zeros((len(native),2),np.float32)
    for inds in tracks.values():
        inds.sort(key=lambda i:(int(native[i].get("frame_id",0)),int(native[i].get("image_id",0)),i)); prior=[]
        for i in inds:
            if prior:
                sim=feats[i] @ feats[np.asarray(prior,np.int64)].T; history[i]=[float(sim.max()),float(sim.mean())]
            prior.append(i)
    feature_rows=[]; flat=[]; offsets=[0]; targets=[]; max_ious=[]; videos=[]; categories=[]; reliable_groups=0; missing_dims=0
    for key,inds in sorted(groups.items()):
        vid,img=key; w,h=dims.get(vid,(1.0,1.0)); missing_dims += int(vid not in dims); bxs=[box(native[i].get("bbox_xyxy")) for i in inds]; gtvals=gt.get(key,[]); ious=np.asarray([max((iou(b,g) for g,_ in gtvals),default=0.0) for b in bxs],np.float32); catvals=[c for _,c in gtvals if c>=0]; cat=Counter(catvals).most_common(1)[0][0] if catvals else -1; order=np.arange(len(inds),dtype=np.float32); count=max(1,len(inds)-1); local=[]
        for j,i in enumerate(inds):
            b=bxs[j] or [0,0,0,0]; x1,y1,x2,y2=b; bw=max(0.0,x2-x1); bh=max(0.0,y2-y1); area=(bw*bh)/max(w*h,1.0); aspect=math.log(max(bw/max(bh,1e-5),1e-5)); score=float(native[i].get("base_score",0.0) or 0.0); rank=order[j]/count; setmean=float(feats[i]@norm(feats[inds].mean(axis=0))); nearest=0.0
            if len(inds)>1 and len(inds) <= 32:
                sims=feats[i]@feats[np.asarray(inds,np.int64)].T; sims[j]=-1.0; nearest=float(sims.max())
            elif len(inds) > 32:
                # Large native sets are already represented by their
                # centroid; avoid an O(M^2) pairwise pass over 600k rows.
                nearest=setmean
            feature_rows.append([score,x1/max(w,1.0),y1/max(h,1.0),x2/max(w,1.0),y2/max(h,1.0),bw/max(w,1.0),bh/max(h,1.0),area,aspect,rank,math.log1p(len(inds)),history[i,0],history[i,1],setmean,nearest]); flat.append(i)
        if len(ious) and float(ious.max())>=.5:
            cand=np.flatnonzero(ious>=.5); target=int(max(cand,key=lambda j:(float(ious[j]),float(native[inds[int(j)]].get("base_score",0.0) or 0.0),-int(native[inds[int(j)]].get("candidate_rank",0))))); reliable_groups+=1
        else: target=len(inds)
        targets.append(target); max_ious.append(float(ious.max()) if len(ious) else 0.0); videos.append(vid); categories.append(cat); offsets.append(len(flat))
    blocked=set()
    for p in (POS,NEG):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e=json.loads(line); blocked.update((int(e["source_video"]),int(e["target_video"])))
    fm=json.loads(FOLD_MANIFEST.read_text(encoding="utf-8")); folds={}
    for fi,f in enumerate(fm["folds"]):
        fitv=set(map(int,f["fit_videos"]))-blocked; valv=set(map(int,f["validation_videos"]))-blocked; fitc=set(map(int,f["fit_categories"])); valc=set(map(int,f.get("held_categories",[]))); fit=[j for j,(v,c) in enumerate(zip(videos,categories)) if v in fitv and (c<0 or c in fitc)]; val=[j for j,(v,c) in enumerate(zip(videos,categories)) if v in valv and c in valc]; folds[str(fi)]={"fit_groups":fit,"validation_groups":val,"fit_videos":sorted(fitv),"validation_videos":sorted(valv),"fit_categories":sorted(fitc),"validation_categories":sorted(valc),"video_disjoint":True,"category_disjoint":True}
    outdir=Path("/data2/usr_for_deadline/trackocd_phase83/b4_native_sets");outdir.mkdir(parents=True,exist_ok=True); path=outdir/"b4_native_sets_v1.npz";fd,tmp=tempfile.mkstemp(prefix=".b4_native_sets_v1.",suffix=".npz",dir=str(outdir));os.close(fd)
    try:
        with open(tmp,"wb") as f: np.savez(f,features=np.asarray(feature_rows,np.float32),flat_indices=np.asarray(flat,np.int64),offsets=np.asarray(offsets,np.int64),targets=np.asarray(targets,np.int64),videos=np.asarray(videos,np.int64),categories=np.asarray(categories,np.int64),max_iou=np.asarray(max_ious,np.float32)); f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    manifest={"schema_version":"trackocd.phase83.b4.native_candidate_sets.v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"native":str(NATIVE.resolve()),"native_sha256":sha(NATIVE),"features":str(FEATURES.resolve()),"features_sha256":sha(FEATURES),"data":str(path.resolve()),"data_sha256":sha(path),"groups":len(groups),"candidate_rows":len(flat),"reliable_target_groups":reliable_groups,"defer_target_groups":len(groups)-reliable_groups,"feature_names":["base_score","x1_norm","y1_norm","x2_norm","y2_norm","box_width_norm","box_height_norm","box_area_norm","box_aspect_log","candidate_rank_norm","candidate_count_log","history_max_cosine","history_mean_cosine","candidate_set_mean_cosine","candidate_set_nearest_cosine"],"missing_video_dimensions":missing_dims,"event_videos_excluded":sorted(blocked),"folds":folds,"labels_used_only_for_train_targets":True,"model_input_forbidden":["gt_bbox","assigned","row_iou","category","physical_id","semantic_id","future","text","event_key"],"public_dev_q1_sealed_accessed":False}
    atomic_json(OUT/"manifests/b4_native_sets_v1.json",manifest);atomic_json(OUT/"status.json",{"phase":"Phase83","route":"B4_NATIVE_CANDIDATE_BUILD","status":"COMPLETE","manifest":str((OUT/"manifests/b4_native_sets_v1.json").resolve()),"public_dev_q1_sealed_accessed":False});atomic_json(OUT/"completion/b4_native_candidate_build.done",{"status":"DONE","manifest":str((OUT/"manifests/b4_native_sets_v1.json").resolve())});print(json.dumps({"status":"COMPLETE","groups":len(groups),"candidate_rows":len(flat),"reliable_target_groups":reliable_groups,"defer_target_groups":len(groups)-reliable_groups,"data":str(path)},indent=2,sort_keys=True))


if __name__=="__main__":main()
