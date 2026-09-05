#!/usr/bin/env python3
"""Frozen raw/reranker/defer replay on the registered 76+76 causal events."""
from __future__ import annotations
import argparse, ast, csv, datetime as dt, hashlib, json, os, sys, tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import stable_raw_topk,set_context
from src.iclr27_phase85.support_model import SupportReranker,numpy_predict
OUT=ROOT/'outputs/iclr27_phase85'; NATIVE=Path('/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl'); FEAT=Path('/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz'); DESC=Path('/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz'); SOURCE=Path('/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/source_track_native_vectors.npz'); OBS=Path('/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl'); PUBLIC=ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv'; PREFIXES=(1,2,4,8,16)
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def box(v):
 try:return [float(x) for x in (json.loads(v) if isinstance(v,str) else v)]
 except Exception:return None
def iou(a,b):
 if not a or not b or len(a)!=4 or len(b)!=4:return 0.
 x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(aa+bb-inter,1e-8)
def norm(v):
 v=np.asarray(v,np.float32);return v/max(float(np.linalg.norm(v)),1e-8)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--source-cache',type=Path,default=SOURCE)
 ap.add_argument('--output-tag',default='support_event_replay')
 ap.add_argument('--policy',choices=('rerank_defer','raw_defer'),default='rerank_defer')
 args=ap.parse_args()
 native=[json.loads(l) for l in NATIVE.open() if l.strip()]; nf=np.asarray(np.load(FEAT,allow_pickle=False)['features'],np.float32);nf/=np.maximum(np.linalg.norm(nf,axis=1,keepdims=True),1e-8); dz=np.load(DESC,allow_pickle=False); desc=np.zeros((len(native),15),np.float32);desc[dz['flat_indices'].astype(np.int64)]=dz['features'].astype(np.float32)
 groups=defaultdict(list)
 for i,r in enumerate(native):
  if box(r.get('bbox_xyxy')) is not None:groups[(int(r['video_id']),int(r.get('image_id',-1)))].append(i)
 for g in groups:groups[g].sort(key=lambda i:(int(native[i].get('candidate_rank') or 0),int(native[i].get('proposal_local_id') or 0),i))
 public=list(csv.DictReader(PUBLIC.open(newline='')));gt={str(r['row_key']):box(r.get('gt_bbox_xyxy')) for r in public}; lengths=defaultdict(int)
 for r in public:lengths[f"v{int(r['video_id'])}:p{int(r['track_id'])}"]+=1
 s=np.load(args.source_cache,allow_pickle=False);skeys=[str(x) for x in s['keys'].tolist()];si={k:i for i,k in enumerate(skeys)};sv=s['vectors'].astype(np.float32);sp=s['prototypes'].astype(np.float32)
 models={}
 for f in range(3):
  paths=sorted((Path('/data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints')).glob(f'support_reranker_formal_r1_f{f}_step*.pt'))
  if not paths:raise FileNotFoundError(f'missing model fold {f}')
  z=torch.load(str(paths[-1]),map_location='cpu');m=SupportReranker(19,10,64,.05);m.load_state_dict(z['model']);m.eval();models[f]={'model':m,'mean':np.asarray(z['mean'],np.float32),'std':np.asarray(z['std'],np.float32),'path':str(paths[-1].resolve()),'sha256':sha(paths[-1])}
 records=[]
 for e in [json.loads(l) for l in OBS.open() if l.strip()]:
  sk=str(e.get('source_tracklet_key')); fold=int(e.get('fold',0)); source_valid=sk in si
  target_rows=[]
  if source_valid:
   src=norm(sv[4,si[sk]]); prot=[norm(x) for x in sp[:,si[sk]] if np.linalg.norm(x)>1e-8] or [src]
   for detail in e.get('target_row_details',[]):
    key=(int(detail.get('video_id',-1)),int(detail.get('image_id',-1)));inds=groups.get(key,[]); raw=np.asarray(nf[np.asarray(inds,np.int64)]@src if inds else [],np.float32); idx=stable_raw_topk(raw,32); cand=np.asarray(inds,np.int64)[idx] if len(idx) else np.empty(0,np.int64); z=nf[cand] if len(cand) else np.empty((0,nf.shape[1]),np.float32); pm=np.stack([z@p for p in prot],axis=1) if len(z) else np.empty((0,1),np.float32); extra=np.stack([raw[idx],pm.max(1),pm.mean(1),pm.min(1)],axis=1) if len(z) else np.empty((0,4),np.float32); x=np.concatenate([desc[cand],extra],axis=1).astype(np.float32) if len(z) else np.empty((0,19),np.float32); context=set_context(raw[idx],len(inds),lengths.get(sk,0),float(np.mean([1-float(np.dot(p,src)) for p in prot])) if prot else 0.,float(np.clip((float(extra[:,1].mean())+1)/2,0,1)) if len(extra) else 0.); model=models[fold%3]; rscore,p,_=numpy_predict(model['model'],x,context,raw[idx],model['mean'],model['std'],'cpu') if len(x) else (np.empty(0),0.,np.empty(0)); raw_choice=int(np.argmax(raw[idx])) if len(idx) else None; rerank_choice=int(np.argmax(rscore)) if len(rscore) else None; use=p<.5; final_choice=((rerank_choice if args.policy=='rerank_defer' else raw_choice) if use else None); target_gt=gt.get(str(detail.get('row_key'))); raw_iou=iou(box(native[int(cand[raw_choice])].get('bbox_xyxy')),target_gt) if raw_choice is not None else 0.; rank_iou=iou(box(native[int(cand[rerank_choice])].get('bbox_xyxy')),target_gt) if rerank_choice is not None else 0.; final_iou=iou(box(native[int(cand[final_choice])].get('bbox_xyxy')),target_gt) if final_choice is not None else 0.; target_rows.append({'row_key':str(detail.get('row_key')),'video_id':key[0],'image_id':key[1],'candidate_count':len(inds),'topk_count':len(cand),'defer_probability':float(p),'defer':bool(not use),'raw_iou':float(raw_iou),'reranked_iou':float(rank_iou),'final_iou':float(final_iou),'raw_reliable':bool(raw_iou>=.5),'reranked_reliable':bool(rank_iou>=.5),'final_reliable':bool(final_iou>=.5),'raw_choice_rank':raw_choice,'reranked_choice_rank':rerank_choice,'final_choice_rank':final_choice})
  records.append({'event_key':str(e.get('event_key')),'model_event_uid':str(e.get('model_event_uid')),'fold':fold,'polarity':str(e.get('polarity')),'prefix':int(e.get('prefix',0)),'source_tracklet_key':sk,'target_tracklet_key':str(e.get('target_tracklet_key')),'source_valid':source_valid,'source_reliable_frozen':bool(e.get('source_reliable',False)),'target_reliable_frozen':bool(e.get('target_reliable',False)),'both_reliable_frozen':bool(e.get('both_reliable',False)),'target_rows':target_rows,'raw_reliable':bool(any(r['raw_reliable'] for r in target_rows)),'reranked_reliable':bool(any(r['reranked_reliable'] for r in target_rows)),'final_reliable':bool(any(r['final_reliable'] for r in target_rows)),'deferred':bool(any(r['defer'] for r in target_rows))})
 summary=[]
 for p in PREFIXES:
  for pol in ('positive','negative'):
   rs=[r for r in records if r['prefix']==p and r['polarity']==pol];summary.append({'prefix':p,'polarity':pol,'events':len(rs),'raw_reliable_events':sum(r['raw_reliable'] for r in rs),'reranked_reliable_events':sum(r['reranked_reliable'] for r in rs),'final_reliable_events':sum(r['final_reliable'] for r in rs),'deferred_events':sum(r['deferred'] for r in rs),'source_reliable_frozen':sum(r['source_reliable_frozen'] for r in rs),'target_reliable_frozen':sum(r['target_reliable_frozen'] for r in rs),'both_reliable_frozen':sum(r['both_reliable_frozen'] for r in rs)})
 strategy='raw source-mean top32; bounded residual reranker; separate TRAIN defer head (p>=0.5 -> DEFER)' if args.policy=='rerank_defer' else 'raw source-mean top32; TRAIN defer head only (p>=0.5 -> DEFER); reranker output ignored'
 out={'schema_version':'trackocd.phase85.support_event_replay.v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'strategy':strategy,'policy':args.policy,'source_cache':str(args.source_cache.resolve()),'source_cache_sha256':sha(args.source_cache),'records':records,'summary':summary,'model_checkpoints':{str(f):{k:v for k,v in z.items() if k in ('path','sha256')} for f,z in models.items()},'inputs':{'native':str(NATIVE.resolve()),'native_sha256':sha(NATIVE),'features':str(FEAT.resolve()),'features_sha256':sha(FEAT),'descriptor':str(DESC.resolve()),'descriptor_sha256':sha(DESC),'source_cache':str(args.source_cache.resolve()),'source_cache_sha256':sha(args.source_cache),'observability':str(OBS.resolve()),'observability_sha256':sha(OBS)},'denominators':{'positive_events':76,'negative_events':76,'prefixes':list(PREFIXES)},'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'event_labels_posthoc_only':True,'controller_run':False}
 stem=args.output_tag
 atom(OUT/f'metrics/{stem}.json',out); atom(OUT/f'audit/{stem}.json',out); atom(OUT/f'completion/{stem}.done',{'status':'DONE','metrics':str((OUT/f'metrics/{stem}.json').resolve()),'sha256':sha(OUT/f'metrics/{stem}.json')});print(json.dumps({'p16':[x for x in summary if x['prefix']==16]},indent=2,sort_keys=True))
if __name__=='__main__':main()
