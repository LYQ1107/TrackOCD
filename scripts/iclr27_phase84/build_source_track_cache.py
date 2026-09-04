#!/usr/bin/env python3
"""Materialize compact native-DINO causal track vectors for B84S."""
from __future__ import annotations
import csv, datetime as dt, hashlib, json, os, sys, tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key
NATIVE=Path('/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl'); FEATURES=Path('/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz'); PUBLIC=ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv'; OUT=ROOT/'outputs/iclr27_phase84'; DATA=Path('/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/source_track_native_vectors.npz')
def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom_json(p:Path,v:Any):
 p.parent.mkdir(parents=True,exist_ok=True); fd,n=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));
 try:
  with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(n,p)
 finally:
  if os.path.exists(n): os.unlink(n)
def box(v):
 try:
  x=[float(z) for z in (json.loads(v) if isinstance(v,str) else v)]; return x if len(x)==4 else None
 except Exception:return None
def iou(a,b):
 if a is None or b is None:return 0.
 x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0.,x2-x1)*max(0.,y2-y1); aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]); bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1]); return inter/max(aa+bb-inter,1e-8)
def norm(v): v=np.asarray(v,np.float32); return v/max(float(np.linalg.norm(v)),1e-8)
def main():
 table=load_frozen_tracks(); public=list(csv.DictReader(PUBLIC.open(newline='',encoding='utf-8'))); native=[json.loads(l) for l in NATIVE.open(encoding='utf-8') if l.strip()]; feat=np.asarray(np.load(FEATURES,allow_pickle=False)['features'],np.float32)
 by_image=defaultdict(list)
 for i,r in enumerate(native):
  if box(r.get('bbox_xyxy')) is not None: by_image[(int(r['video_id']),int(r.get('image_id',-1)))].append(i)
 mapped=defaultdict(list); matched_rows=0
 for r in public:
  pb=box(r.get('bbox_xyxy')); cands=by_image.get((int(r['video_id']),int(r['image_id'])),[])
  if pb is None or not cands: continue
  best=max(cands,key=lambda j:(iou(pb,box(native[j].get('bbox_xyxy'))),float(native[j].get('base_score',0.) or 0.),-int(native[j].get('candidate_rank') or 0),-j)); sc=iou(pb,box(native[best].get('bbox_xyxy')))
  if sc>=.5: mapped[track_key(r)].append((order_key(r),best)); matched_rows+=1
 for k in mapped: mapped[k].sort(key=lambda z:z[0])
 keys=sorted(table.metadata); idx={k:i for i,k in enumerate(keys)}; vectors=np.zeros((len(PREFIXES),len(keys),768),np.float32); protos=np.zeros((3,len(keys),768),np.float32); avail={str(p):0 for p in PREFIXES}
 for pi,p in enumerate(PREFIXES):
  for k in keys:
   arr=np.asarray([feat[j] for _,j in mapped.get(k,[])[:p]],np.float32)
   if len(arr): vectors[pi,idx[k]]=norm(arr.mean(0)); avail[str(p)]+=1
   else: vectors[pi,idx[k]]=table.raw_vector(k,p)
   if p==16 and len(arr):
    for ci,ch in enumerate(np.array_split(arr,min(3,len(arr)))): protos[ci,idx[k]]=norm(ch.mean(0))
 raw=np.stack([[table.raw_vector(k,p) for k in keys] for p in PREFIXES]).astype(np.float32)
 DATA.parent.mkdir(parents=True,exist_ok=True); tmp=DATA.with_name('.'+DATA.name+'.tmp.npz'); np.savez(tmp,keys=np.asarray(keys),vectors=vectors,prototypes=protos,raw_vectors=raw); os.replace(tmp,DATA)
 m={'schema_version':'trackocd.phase84.source_track_native_cache.v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'public_csv':str(PUBLIC.resolve()),'public_csv_sha256':sha(PUBLIC),'native_lineage':str(NATIVE.resolve()),'native_sha256':sha(NATIVE),'native_features':str(FEATURES.resolve()),'native_features_sha256':sha(FEATURES),'track_count':len(keys),'matched_public_rows_iou_ge_0.5':matched_rows,'prefix_coverage':avail,'data':str(DATA.resolve()),'data_sha256':sha(DATA),'prototypes':'up to M=3 contiguous causal chunks of source prefix16','public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 atom_json(OUT/'manifests/source_track_native_cache.json',m); print(json.dumps(m,indent=2,sort_keys=True))
if __name__=='__main__': main()
