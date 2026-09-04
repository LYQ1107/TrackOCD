#!/usr/bin/env python3
"""Build the fixed TRAIN-only native candidate manifest for B84S.

The candidate universe is the native Q0 per-image set.  Source vectors are
looked up from the compact corrected-DINO cache; category/IoU fields are
retained only as training labels and never enter the feature tensor.
"""
from __future__ import annotations
import csv, datetime as dt, hashlib, json, os, sys, tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase23.protocol import track_key
NATIVE=Path('/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl'); FEATURES=Path('/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz'); B4=Path('/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz'); B4MAN=ROOT/'outputs/iclr27_phase83/manifests/b4_native_sets_v1.json'; EPISODES=ROOT/'outputs/iclr27_phase30/manifests'; OUT=ROOT/'outputs/iclr27_phase84'; DATA=Path('/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/b84s_candidate_features.npz')
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,n=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(n,p)
 finally:
  if os.path.exists(n):os.unlink(n)
def box(v):
 try:
  x=[float(z) for z in (json.loads(v) if isinstance(v,str) else v)];return x if len(x)==4 else None
 except Exception:return None
def norm(v):v=np.asarray(v,np.float32);return v/max(float(np.linalg.norm(v)),1e-8)
def main():
 table=load_frozen_tracks(); native=[json.loads(l) for l in NATIVE.open(encoding='utf-8') if l.strip()]; native_feat=np.asarray(np.load(FEATURES,allow_pickle=False)['features'],np.float32); b4=np.load(B4,allow_pickle=False); base=b4['features'].astype(np.float32); flat=b4['flat_indices'].astype(np.int64); offsets=b4['offsets'].astype(np.int64); targets=b4['targets'].astype(np.int64); videos=b4['videos'].astype(np.int64); cats=b4['categories'].astype(np.int64); max_iou=b4['max_iou'].astype(np.float32); b4m=json.loads(B4MAN.read_text(encoding='utf-8'))
 # Build causal native-DINO vectors and M=3 prototypes from public TRAIN keys.
 by_image=defaultdict(list)
 for i,r in enumerate(native):
  if box(r.get('bbox_xyxy')) is not None:by_image[(int(r['video_id']),int(r.get('image_id',-1)))].append(i)
 public=list(csv.DictReader((ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv').open(newline='',encoding='utf-8'))); mapped=defaultdict(list)
 def iou(a,b):
  if a is None or b is None:return 0.
  x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]);inter=max(0.,x2-x1)*max(0.,y2-y1);aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1]);return inter/max(aa+bb-inter,1e-8)
 for r in public:
  pb=box(r.get('bbox_xyxy')); cs=by_image.get((int(r['video_id']),int(r['image_id'])),[])
  if pb is None or not cs:continue
  j=max(cs,key=lambda q:(iou(pb,box(native[q].get('bbox_xyxy'))),float(native[q].get('base_score',0.) or 0.),-int(native[q].get('candidate_rank') or 0),-q)); sc=iou(pb,box(native[j].get('bbox_xyxy')))
  if sc>=.5:mapped[track_key(r)].append((int(r.get('event_rank',0)),j))
 for k in mapped:mapped[k].sort(key=lambda z:z[0])
 source_vec={}; source_proto={}
 for k in table.metadata:
  arr=np.asarray([native_feat[j] for _,j in mapped.get(k,[])[:16]],np.float32)
  if not len(arr):continue
  source_vec[k]=norm(arr.mean(0)); chunks=np.array_split(arr,min(3,len(arr))); source_proto[k]=[norm(c.mean(0)) for c in chunks]
 # Select deterministic source supports from TRAIN fit episodes by category.
 support_by_cat=defaultdict(list)
 for fi in range(4):
  d=json.loads((EPISODES/f'episode_manifest_f{fi}.json').read_text(encoding='utf-8'))
  for r in d['records']:
   if r.get('split')!='fit' or r.get('kind')!='multi_positive_cross_video':continue
   q=str(r.get('query_track_key')); cat=int(table.metadata[q]['category']) if q in table.metadata else -1
   for s in r.get('support_track_keys',[]):
    s=str(s)
    if s in source_vec:support_by_cat[cat].append(s)
 for c in support_by_cat:support_by_cat[c]=sorted(set(support_by_cat[c]))
 # Add four source-conditioned channels to frozen native candidate descriptors.
 feat_rows=np.zeros((len(base),base.shape[1]+4),np.float32); support_group=np.full(len(offsets)-1,-1,np.int64); support_keys=[]
 for g in range(len(offsets)-1):
  c=int(cats[g]); choices=support_by_cat.get(c,[]); gv=int(videos[g]); choices=[s for s in choices if int(table.metadata[s]['video'])!=gv]
  sk=choices[0] if choices else None; support_keys.append(sk or '');
  if sk is not None:support_group[g]=list(table.metadata).index(sk) if sk in table.metadata else -1
  sv=source_vec.get(sk) if sk else None; sp=source_proto.get(sk,[]) if sk else []
  inds=flat[offsets[g]:offsets[g+1]]
  if sv is not None:
   for pos,ni in enumerate(inds):
    z=norm(native_feat[int(ni)]); sims=[float(z@p) for p in sp] if sp else [float(z@sv)]; vals=[float(z@sv),max(sims),float(np.mean(sims)),float(np.min(sims))]; feat_rows[offsets[g]+pos]=np.concatenate([base[offsets[g]+pos],np.asarray(vals,np.float32)])
  else: feat_rows[offsets[g]:offsets[g+1],:base.shape[1]]=base[offsets[g]:offsets[g+1]]
 # Features for groups without a source are explicit zeros in source channels,
 # while their DEFER/candidate target remains in the listwise labels.
 DATA.parent.mkdir(parents=True,exist_ok=True); tmp=DATA.with_name('.'+DATA.name+'.tmp.npz'); np.savez(tmp,features=feat_rows,offsets=offsets,targets=targets,videos=videos,categories=cats,max_iou=max_iou,source_group=support_group); os.replace(tmp,DATA)
 folds={}
 for k,v in b4m['folds'].items(): folds[k]={**v,'fit_groups':[int(x) for x in v['fit_groups']],'validation_groups':[int(x) for x in v['validation_groups']]}
 manifest={'schema_version':'trackocd.phase84.b84s.native_source_conditioned_manifest.v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'native':str(NATIVE.resolve()),'native_sha256':sha(NATIVE),'native_features':str(FEATURES.resolve()),'native_features_sha256':sha(FEATURES),'b4_manifest':str(B4MAN.resolve()),'b4_manifest_sha256':sha(B4MAN),'data':str(DATA.resolve()),'data_sha256':sha(DATA),'groups':len(offsets)-1,'candidate_rows':len(base),'feature_dim':int(feat_rows.shape[1]),'feature_names':['frozen_native_candidate_descriptors','source_mean_cosine','source_proto_max_cosine','source_proto_mean_cosine','source_proto_min_cosine'],'source_support_rule':'deterministic first TRAIN-fit support track of same category and different video; at most one source track per group, source has up to M=3 causal prototypes','folds':folds,'support_groups':int(np.sum(support_group>=0)),'defer_target_groups':int(np.sum(targets>=np.diff(offsets))),'labels_used_only_for_train_targets':True,'model_input_forbidden':['category','gt_iou','gt_bbox','assigned','physical_id','semantic_id','future','text','event_key','StateMemory','controller_action'],'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 atom_json(OUT/'manifests/b84s_native_manifest.json',manifest); atom_json(OUT/'status.json',{'phase':'Phase84','route':'B84S_MANIFEST','status':'COMPLETE','manifest':str((OUT/'manifests/b84s_native_manifest.json').resolve()),'public_dev_q1_sealed_accessed':False}); print(json.dumps({'groups':manifest['groups'],'candidate_rows':manifest['candidate_rows'],'support_groups':manifest['support_groups'],'feature_dim':manifest['feature_dim']},indent=2,sort_keys=True))
if __name__=='__main__':main()
