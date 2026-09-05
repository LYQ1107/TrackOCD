#!/usr/bin/env python3
"""Build Phase85 prefix-matched raw-anchored support groups from TRAIN only."""
from __future__ import annotations
import ast, csv, datetime as dt, hashlib, json, os, tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import sys
if str(ROOT:=Path(__file__).resolve().parents[2]) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import stable_raw_topk
OUT=ROOT/"outputs/iclr27_phase85"
NATIVE=Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl"); FEAT=Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz"); DESC=Path("/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz"); SOURCE=Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/source_track_native_vectors.npz"); OBS=Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl"); PUBLIC=ROOT/"data/iclr27_phase19r/sources/public_rows_corrected.csv"; EP=ROOT/"outputs/iclr27_phase30/manifests"
OLD_FOLD_MANIFEST=ROOT/"outputs/iclr27_phase84/manifests/b84sq_balanced_v3_manifest.json"
PREFIXES=(1,2,4,8,16)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def box(v):
 try:
  x=[float(a) for a in (json.loads(v) if isinstance(v,str) else v)]; return x if len(x)==4 else None
 except Exception:return None
def iou(a,b):
 if not a or not b:return 0.
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0.,x2-x1)*max(0.,y2-y1); aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1]);return inter/max(aa+bb-inter,1e-8)
def norm(x):
 x=np.asarray(x,np.float32); return x/max(float(np.linalg.norm(x)),1e-8)
def tkey(r): return f"v{int(r['video_id'])}:p{int(r['track_id'])}"
def order(r): return (int(r.get('event_rank',0)),int(r.get('frame_id',0)),int(r.get('proposal_local_id',0)))
def main():
 native=[json.loads(x) for x in NATIVE.open() if x.strip()]; nf=np.asarray(np.load(FEAT,allow_pickle=False)['features'],np.float32); nf/=np.maximum(np.linalg.norm(nf,axis=1,keepdims=True),1e-8); dz=np.load(DESC,allow_pickle=False); desc=np.zeros((len(native),15),np.float32); desc[dz['flat_indices'].astype(np.int64)]=dz['features'].astype(np.float32)
 groups=defaultdict(list)
 for i,r in enumerate(native):
  if box(r.get('bbox_xyxy')) is not None: groups[(int(r['video_id']),int(r.get('image_id',-1)))].append(i)
 for k in groups: groups[k].sort(key=lambda i:(int(native[i].get('candidate_rank') or 0),int(native[i].get('proposal_local_id') or 0),i))
 public=list(csv.DictReader(PUBLIC.open(newline=''))); bytrack=defaultdict(list); gt_by_image=defaultdict(list)
 for r in public:
  bytrack[tkey(r)].append(r); gb=box(r.get('gt_bbox_xyxy'))
  if gb is not None: gt_by_image[(int(r['video_id']),int(r['image_id']))].append((gb,int(float(r.get('gt_category_id_common',-1) or -1))))
 for k in bytrack: bytrack[k].sort(key=order)
 source=np.load(SOURCE,allow_pickle=False); skeys=[str(x) for x in source['keys'].tolist()]; si={k:i for i,k in enumerate(skeys)}; sv=source['vectors'].astype(np.float32); sp=source['prototypes'].astype(np.float32)
 blocked=set()
 if OBS.exists():
  for line in OBS.open():
   if line.strip():
    e=json.loads(line); blocked|={int(e.get('source_video',-1)),int(e.get('target_video',-1))}
 # Candidate descriptors are mapped by native row index; source/query pairing
 # is taken from the legal TRAIN episode manifest, never from held events.
 raw=[]; seen=set()
 for fi in range(4):
  man=json.loads((EP/f'episode_manifest_f{fi}.json').read_text())
  for rec in man.get('records',[]):
   if rec.get('split') not in ('fit','val') or rec.get('kind') not in ('multi_positive_cross_video','null_no_match_hard_negative'):continue
   q=str(rec.get('query_track_key')); qm=bytrack.get(q,[])
   if not qm or q not in si: continue
   qv=int(qm[-1]['video_id']); qcat=int(float(qm[-1].get('gt_category_id_common',-1) or -1))
   if qv in blocked:continue
   for s in sorted(set(str(x) for x in rec.get('support_track_keys',[]))):
    sm=bytrack.get(s,[])
    if not sm or s not in si:continue
    svideo=int(sm[-1]['video_id']); scat=int(float(sm[-1].get('gt_category_id_common',-1) or -1))
    if svideo==qv or svideo in blocked or scat<0 or qcat<0:continue
    for prefix in PREFIXES:
     tr=qm[min(prefix-1,len(qm)-1)]; image=(qv,int(tr['image_id'])); inds=groups.get(image,[])
     if not inds:continue
     key=(s,q,prefix,image[1]);
     if key in seen:continue
     seen.add(key)
     # Fixed TRAIN-side support metadata.  It is based only on completed
     # source observations and frozen source vectors; held-event GT never
     # enters this value or the eventual inference tensor.
     source_len=len(sm)
     source_quality=min(1.0, source_len/16.0)
     source_anchor=norm(sv[4,si[s]])
     _var_terms=[1.0-float(np.dot(norm(pp),source_anchor)) for pp in sp[:,si[s]] if np.linalg.norm(pp)>1e-8]
     source_variance=float(np.mean(_var_terms)) if _var_terms else 0.0
     raw.append({'orig_fold':fi,'source_key':s,'query_key':q,'source_video':svideo,'target_video':qv,'source_category':scat,'query_category':qcat,'prefix':prefix,'target_image':image[1],'target_row_key':str(tr.get('row_key','')),'kind':'positive' if rec.get('kind')=='multi_positive_cross_video' else 'defer','episode_id':str(rec.get('episode_id')),'source_length':source_len,'support_quality':source_quality,'source_variance':source_variance,'native_indices':inds})
 raw.sort(key=lambda c:(c['source_key'],c['query_key'],c['prefix'],c['kind'],c['episode_id']))
 # Keep deterministic exposure bounded per source/kind/original provenance.
 picked=[]; cap=defaultdict(int)
 for c in raw:
  ck=(c['source_key'],c['kind'],c['orig_fold'],c['prefix'])
  if cap[ck]>=2:continue
  cap[ck]+=1;picked.append(c)
 # Materialize raw top-32 candidate actions and TRAIN-only IoU targets.
 feats=[]; offsets=[0]; targets=[]; raw_scores=[]; counts=[]; metas=[]
 for c in picked:
  inds=c.pop('native_indices'); src=norm(sv[4,si[c['source_key']]]); prot=[norm(x) for x in sp[:,si[c['source_key']]] if np.linalg.norm(x)>1e-8] or [src]; z=nf[np.asarray(inds,np.int64)]; sim=z@src; pm=np.stack([z@p for p in prot],axis=1); extra=np.stack([sim,pm.max(1),pm.mean(1),pm.min(1)],axis=1); x=np.concatenate([desc[np.asarray(inds,np.int64)],extra],axis=1).astype(np.float32); kidx=stable_raw_topk(sim,32); x=x[kidx]; rs=sim[kidx]; gtvals=gt_by_image[(c['target_video'],c['target_image'])]; cand_target=len(kidx)
  if c['kind']=='positive':
   vals=[]
   for j,ni in enumerate(np.asarray(inds)[kidx]):
    cb=box(native[int(ni)].get('bbox_xyxy')); bi,bc=max(((iou(cb,g),cat) for g,cat in gtvals),default=(0.,-1))
    if bi>=.5 and bc==c['source_category']:vals.append((j,bi,float(native[int(ni)].get('base_score',0.) or 0.),-int(native[int(ni)].get('candidate_rank') or 0)))
   if vals:cand_target=int(max(vals,key=lambda v:(v[1],v[2],v[3]))[0])
  feats.extend(x.tolist()); offsets.append(len(feats)); targets.append(cand_target); raw_scores.extend(rs.tolist()); counts.append(len(inds)); metas.append({k:c[k] for k in ('source_key','query_key','source_video','target_video','source_category','query_category','prefix','target_image','target_row_key','kind','episode_id','orig_fold','source_length','support_quality','source_variance')})
 if not feats:raise RuntimeError('no legal support groups')
 data=OUT/'manifests/phase85_support_prefix_features.npz'; data.parent.mkdir(parents=True,exist_ok=True); tmp=data.with_name('.'+data.name+'.tmp.npz'); np.savez(tmp,features=np.asarray(feats,np.float32),offsets=np.asarray(offsets,np.int64),targets=np.asarray(targets,np.int64),raw_scores=np.asarray(raw_scores,np.float32),candidate_counts=np.asarray(counts,np.int64)); os.replace(tmp,data)
 # Reuse the registered Phase84 category/video validation assignment where a
 # pair is present, then deterministically assign new pairs by category, video,
 # or source hash. Fit excludes validation categories, videos and source tracks.
 old=json.loads(OLD_FOLD_MANIFEST.read_text()); oz=np.load(old['data'],allow_pickle=False)
 pair_fold={}
 for f,fd in old['folds'].items():
  for g in fd['validation_groups']: pair_fold[(str(oz['source_keys'][g]),str(oz['query_keys'][g]))]=int(f)
 cat_fold={int(c):int(f) for f,fd in old['folds'].items() for c in fd.get('validation_categories',[])}
 video_fold={int(v):int(f) for f,fd in old['folds'].items() for v in fd.get('validation_videos',[])}
 for m in metas:
  fallback=cat_fold.get(int(m['query_category']),video_fold.get(int(m['target_video']),int(hashlib.sha256(m['source_key'].encode()).hexdigest(),16)%3))
  m['assigned_fold']=pair_fold.get((m['source_key'],m['query_key']),fallback)
 folds={}
 for f in range(3):
  val=[i for i,m in enumerate(metas) if m['assigned_fold']==f]; hs={metas[i]['source_key'] for i in val}; hc={metas[i]['query_category'] for i in val}; hv={metas[i]['target_video'] for i in val}; fit=[i for i,m in enumerate(metas) if i not in set(val) and m['source_key'] not in hs and m['query_category'] not in hc and m['target_video'] not in hv]; folds[str(f)]={'fit_groups':fit,'validation_groups':val,'fit_source_tracks':sorted({metas[i]['source_key'] for i in fit}),'validation_source_tracks':sorted(hs),'fit_query_categories':sorted({metas[i]['query_category'] for i in fit}),'validation_query_categories':sorted(hc),'fit_target_videos':sorted({metas[i]['target_video'] for i in fit}),'validation_target_videos':sorted(hv),'source_track_disjoint':not(hs & {metas[i]['source_key'] for i in fit}),'query_category_disjoint':not(hc & {metas[i]['query_category'] for i in fit}),'target_video_disjoint':not(hv & {metas[i]['target_video'] for i in fit})}
 manifest={'schema_version':'trackocd.phase85.support_prefix_manifest.v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'data':str(data.resolve()),'data_sha256':sha(data),'groups':len(metas),'candidate_rows':len(feats),'feature_dim':19,'max_k':32,'prefixes':list(PREFIXES),'groups_meta':metas,'fold_count':3,'folds':folds,'source_cache':str(SOURCE.resolve()),'source_cache_sha256':sha(SOURCE),'native':str(NATIVE.resolve()),'native_sha256':sha(NATIVE),'native_features':str(FEAT.resolve()),'native_features_sha256':sha(FEAT),'candidate_descriptor':str(DESC.resolve()),'candidate_descriptor_sha256':sha(DESC),'episode_dir':str(EP.resolve()),'event_videos_excluded':sorted(x for x in blocked if x>=0),'train_labels_posthoc_only':True,'model_input_forbidden':['category','gt_bbox','gt_iou','physical_id','semantic_id','future','text','event_key','StateMemory','controller_action'],'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 atom(OUT/'manifests/phase85_support_prefix_manifest.json',manifest); atom(OUT/'audit/support_manifest_build.json',{'schema_version':'trackocd.phase85.support_manifest_build.v1','manifest':str((OUT/'manifests/phase85_support_prefix_manifest.json').resolve()),'groups':len(metas),'candidate_rows':len(feats),'folds':folds,'public_dev_q1_sealed_accessed':False}); print(json.dumps({'groups':len(metas),'candidate_rows':len(feats),'folds':{f:{'fit':len(v['fit_groups']),'val':len(v['validation_groups']),'source_disjoint':v['source_track_disjoint'],'category_disjoint':v['query_category_disjoint'],'video_disjoint':v['target_video_disjoint']} for f,v in folds.items()}},indent=2,sort_keys=True))
if __name__=='__main__':main()
