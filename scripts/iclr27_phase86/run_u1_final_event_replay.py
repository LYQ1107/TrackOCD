#!/usr/bin/env python3
"""Replay all events with one all-TRAIN support expert and one all-OOF U1 gate."""
from __future__ import annotations
import argparse,csv,datetime as dt,hashlib,json,os,sys,tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import set_context,stable_raw_topk
from src.iclr27_phase85.support_model import SupportReranker,numpy_predict
from scripts.iclr27_phase86.run_u1_selective import UtilityGate
OUT=ROOT/'outputs/iclr27_phase86'; OBS=Path('/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl'); NATIVE=Path('/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl'); FEAT=Path('/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz'); DESC=Path('/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz'); SOURCE=Path('/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/source_track_native_vectors.npz'); PUBLIC=ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv'; PREFIXES=(1,2,4,8,16); CK=Path('/data2/usr_for_deadline/trackocd_phase86/project_outputs/checkpoints')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def box(v):
 try:return [float(x) for x in (json.loads(v) if isinstance(v,str) else v)]
 except Exception:return None
def iou(a,b):
 if not a or not b:return 0.
 x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/max(aa+bb-inter,1e-8)
def norm(v):
 v=np.asarray(v,np.float32); return v/max(float(np.linalg.norm(v)),1e-8)
def load_models():
 ep=CK/'support_expert_final_alltrain_v1.pt'; gp=CK/'u1_final_alltrain_v1.pt'; ez=torch.load(ep,map_location='cpu'); gz=torch.load(gp,map_location='cpu'); em=SupportReranker(19,10,64,.05); em.load_state_dict(ez['model']); em.eval(); gm=UtilityGate(); gm.load_state_dict(gz['model']); gm.eval(); return {'model':em,'mean':np.asarray(ez['mean'],np.float32),'std':np.asarray(ez['std'],np.float32),'path':str(ep.resolve()),'sha256':sha(ep)}, {'model':gm,'mean':np.asarray(gz['mean'],np.float32),'std':np.asarray(gz['std'],np.float32),'path':str(gp.resolve()),'sha256':sha(gp)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='u1_final_alltrain_event_v1'); a=ap.parse_args(); expert,gate=load_models()
 native=[json.loads(l) for l in NATIVE.open() if l.strip()]; nf=np.asarray(np.load(FEAT,allow_pickle=False)['features'],np.float32); nf/=np.maximum(np.linalg.norm(nf,axis=1,keepdims=True),1e-8); dz=np.load(DESC,allow_pickle=False); desc=np.zeros((len(native),15),np.float32); desc[dz['flat_indices'].astype(np.int64)]=dz['features'].astype(np.float32)
 groups=defaultdict(list)
 for i,r in enumerate(native):
  if box(r.get('bbox_xyxy')) is not None: groups[(int(r['video_id']),int(r.get('image_id',-1)))].append(i)
 for g in groups: groups[g].sort(key=lambda i:(int(native[i].get('candidate_rank') or 0),int(native[i].get('proposal_local_id') or 0),i))
 public=list(csv.DictReader(PUBLIC.open(newline=''))); gt={str(r['row_key']):box(r.get('gt_bbox_xyxy')) for r in public}; lengths=defaultdict(int)
 for r in public: lengths[f"v{int(r['video_id'])}:p{int(r['track_id'])}"]+=1
 s=np.load(SOURCE,allow_pickle=False); skeys=[str(x) for x in s['keys'].tolist()]; si={k:i for i,k in enumerate(skeys)}; sv=s['vectors'].astype(np.float32); sp=s['prototypes'].astype(np.float32); records=[]
 for event in (json.loads(l) for l in OBS.open() if l.strip()):
  source_key=str(event.get('source_tracklet_key')); source_valid=source_key in si; target_rows=[]
  if source_valid:
   src=norm(sv[4,si[source_key]]); prot=[norm(x) for x in sp[:,si[source_key]] if np.linalg.norm(x)>1e-8] or [src]
   for detail in event.get('target_row_details',[]):
    inds=groups.get((int(detail.get('video_id',-1)),int(detail.get('image_id',-1))),[]); raw=np.asarray(nf[np.asarray(inds,np.int64)]@src if inds else [],np.float32); idx=stable_raw_topk(raw,32); cand=np.asarray(inds,np.int64)[idx] if len(idx) else np.empty(0,np.int64); zc=nf[cand] if len(cand) else np.empty((0,nf.shape[1]),np.float32); pm=np.stack([zc@p for p in prot],axis=1) if len(zc) else np.empty((0,1),np.float32); extra=np.stack([raw[idx],pm.max(1),pm.mean(1),pm.min(1)],axis=1) if len(zc) else np.empty((0,4),np.float32); xx=np.concatenate([desc[cand],extra],axis=1).astype(np.float32) if len(cand) else np.empty((0,19),np.float32)
    context=set_context(raw[idx],len(inds),lengths.get(source_key,0),float(np.mean([1-float(np.dot(p,src)) for p in prot])) if prot else 0.,float(np.clip((float(extra[:,1].mean())+1)/2,0,1)) if len(zc) else 0.); rscore,_,_=numpy_predict(expert['model'],xx,context,raw[idx],expert['mean'],expert['std'],'cpu') if len(xx) else (np.empty(0),0.,np.empty(0)); ro=stable_raw_topk(raw[idx],len(raw[idx])); mo=stable_raw_topk(rscore,len(rscore)); fvec=np.concatenate([context,xx.mean(0) if len(xx) else np.zeros(19,np.float32),xx.std(0) if len(xx) else np.zeros(19,np.float32),xx[ro[0],:16] if len(ro) else np.zeros(16,np.float32)]).astype(np.float32); utility=float(gate['model'](torch.from_numpy(((fvec-gate['mean'])/gate['std'])[None,:]))[0].detach()); use=bool(utility>0); rc=int(ro[0]) if len(ro) else None; mc=int(mo[0]) if len(mo) else None; fc=mc if use else rc; target=gt.get(str(detail.get('row_key'))); ri=iou(box(native[int(cand[rc])].get('bbox_xyxy')),target) if rc is not None else 0.; mi=iou(box(native[int(cand[mc])].get('bbox_xyxy')),target) if mc is not None else 0.; fi=iou(box(native[int(cand[fc])].get('bbox_xyxy')),target) if fc is not None else 0.; target_rows.append({'row_key':str(detail.get('row_key')),'raw_iou':ri,'reranked_iou':mi,'final_iou':fi,'raw_reliable':ri>=.5,'reranked_reliable':mi>=.5,'final_reliable':fi>=.5,'utility':utility,'use_rerank':use,'candidate_count':len(inds),'topk_count':len(cand),'raw_choice_rank':rc,'reranked_choice_rank':mc,'final_choice_rank':fc})
  records.append({'event_key':str(event.get('event_key')),'model_event_uid':str(event.get('model_event_uid')),'fold':int(event.get('fold',0)),'polarity':str(event.get('polarity')),'prefix':int(event.get('prefix',0)),'source_tracklet_key':source_key,'source_valid':source_valid,'target_rows':target_rows,'raw_reliable':bool(any(x['raw_reliable'] for x in target_rows)),'reranked_reliable':bool(any(x['reranked_reliable'] for x in target_rows)),'final_reliable':bool(any(x['final_reliable'] for x in target_rows)),'use_rerank_rows':int(sum(x['use_rerank'] for x in target_rows))})
 summary=[]
 for p in PREFIXES:
  for pol in ('positive','negative'):
   rs=[r for r in records if r['prefix']==p and r['polarity']==pol]; summary.append({'prefix':p,'polarity':pol,'events':len(rs),'raw_reliable_events':sum(r['raw_reliable'] for r in rs),'reranked_reliable_events':sum(r['reranked_reliable'] for r in rs),'final_reliable_events':sum(r['final_reliable'] for r in rs),'events_using_reranker':sum(bool(r['use_rerank_rows']) for r in rs)})
 out={'schema_version':'trackocd.phase86.u1_final_event.v1','phase':86,'tag':a.tag,'route':'U1_FINAL_ALLTRAIN_DEPLOYMENT','records':records,'summary':summary,'expert':{k:v for k,v in expert.items() if k in ('path','sha256')},'gate':{k:v for k,v in gate.items() if k in ('path','sha256')},'denominators':{'positive_events':76,'negative_events':76,'prefixes':list(PREFIXES)},'model_selection':'one all-non-event-TRAIN expert and one all-OOF-TRAIN gate; event fold is reporting only','event_labels_posthoc_only':True,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False}
 atom(OUT/f'metrics/{a.tag}.json',out); atom(OUT/f'audit/{a.tag}.json',out); atom(OUT/f'completion/{a.tag}.done',{'status':'DONE','metrics':str((OUT/f'metrics/{a.tag}.json').resolve()),'sha256':sha(OUT/f'metrics/{a.tag}.json')}); print(json.dumps({'p16':[x for x in summary if x['prefix']==16]},indent=2,sort_keys=True))
if __name__=='__main__': main()
