#!/usr/bin/env python3
"""Build TRAIN-only reconnect-gate examples from the temporal physical stream."""
from __future__ import annotations
import csv,datetime as dt,hashlib,json,os,tempfile
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase85'; LINE=OUT/'physical/temporal_mean_full/full_temporal_lineage.jsonl'; PUB=ROOT/'data/iclr27_phase19r/sources/public_rows_corrected.csv'; OBS=Path('/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl')
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def iv(v,d=-1):
 try:return int(v) if v is not None else d
 except Exception:return d
def key(r):return (iv(r.get('video_id')),iv(r.get('frame_id')),iv(r.get('image_id')),iv(r.get('proposal_local_id')))
def box(v):
 try:return [float(x) for x in (json.loads(v) if isinstance(v,str) else v)]
 except Exception:return None
def geom(b,w,h):
 if not b:return (0.,0.,0.,0.)
 w=max(float(w),1.);h=max(float(h),1.);return ((b[2]-b[0])/w,(b[3]-b[1])/h,max(0.,(b[2]-b[0])*(b[3]-b[1])/(w*h)),((b[0]+b[2])*.5/w,(b[1]+b[3])*.5/h))
def main():
 pub=list(csv.DictReader(PUB.open(newline=''))); pmap={key(r):r for r in pub}; blocked=set()
 for l in OBS.open():
  if l.strip():
   e=json.loads(l);blocked|={int(e.get('source_video',-1)),int(e.get('target_video',-1))}
 rows=[json.loads(l) for l in LINE.open() if l.strip()]; rows.sort(key=lambda r:(iv(r.get('video_id')),iv(r.get('frame_id')),iv(r.get('image_id')),iv(r.get('proposal_local_id'))))
 gt_tracks=defaultdict(list);gt_cats=defaultdict(list)
 for r in rows:
  pr=pmap.get(key(r));
  if pr and str(pr.get('gt_track_id','')).strip() not in ('','-1','None'):
   k=(int(r['video_id']),int(r.get('original_physical_track_id',r.get('physical_track_id',-1))));gt_tracks[k].append(int(float(pr['gt_track_id'])));gt_cats[k].append(int(float(pr.get('gt_category_id_common',-1) or -1)))
 majority={k:Counter(v).most_common(1)[0][0] for k,v in gt_tracks.items() if v}; categories={k:Counter(v).most_common(1)[0][0] for k,v in gt_cats.items() if v}
 latest={};history=defaultdict(int); feats=[];labels=[];meta=[]; stats=Counter()
 for r in rows:
  v=int(r.get('video_id',-1));orig=int(r.get('original_physical_track_id',r.get('physical_track_id',-1)));frame=int(r.get('frame_id',-1));
  if v in blocked: continue
  cand=r.get('phase85_candidate_original_track_id'); cc=int(r.get('phase85_assignment_candidate_count') or 0)
  if str(r.get('phase85_parent_assignment_action','')) in ('KEEP_Q0','RECONNECT') and cand is not None and cc>0:
   cand=int(cand); parent=latest.get((v,cand)); child_box=box(r.get('bbox_xyxy')); pb=box(parent.get('bbox_xyxy')) if parent else None; pr=pmap.get(key(r)); w=float(pr.get('image_width',1) if pr else 1);h=float(pr.get('image_height',1) if pr else 1); cw,ch,ca,ccenter=geom(child_box,w,h); pw,ph,pa,pcenter=geom(pb,w,h); dist=float(np.hypot(ccenter[0]-pcenter[0],ccenter[1]-pcenter[1])) if parent else 1.; sized=float(np.hypot(cw-pw,ch-ph)); score=float(r.get('phase85_assignment_score') if r.get('phase85_assignment_score') is not None else -1.); gap=float(r.get('phase85_assignment_gap') if r.get('phase85_assignment_gap') is not None else 17.); x=[score,gap/16.,min(cc,256)/256.,float(r.get('base_score',0.) or 0.),cw,ch,ca,pa,min(dist,2.),min(sized,2.)]; a=majority.get((v,orig));b=majority.get((v,cand));
   if a is not None and b is not None: labels.append(int(a==b));feats.append(x);meta.append({'video_id':v,'frame_id':frame,'child_original':orig,'candidate_original':cand,'action':str(r.get('phase85_parent_assignment_action')),'child_gt_track':a,'candidate_gt_track':b,'child_category':categories.get((v,orig),-1),'source':'TRAIN_native_public_exact_key'});stats['labeled']+=1;stats['positive']+=int(a==b);stats['negative']+=int(a!=b)
  if r.get('bbox_xyxy') is not None:
   latest[(v,orig)]=r;history[(v,orig)]+=1
 arr=np.asarray(feats,np.float32);y=np.asarray(labels,np.float32);fold=np.asarray([int(hashlib.sha256(str(m['video_id']).encode()).hexdigest(),16)%3 for m in meta],np.int64);path=OUT/'manifests/physical_gate_examples.npz';path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.tmp.npz');np.savez(tmp,features=arr,labels=y,fold=fold);os.replace(tmp,path)
 manifest={'schema_version':'trackocd.phase85.physical_gate_manifest.v1','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'data':str(path.resolve()),'rows':len(y),'feature_dim':10,'feature_names':['assignment_score','gap_norm','candidate_count_norm','child_base_score','child_width_norm','child_height_norm','child_area_norm','parent_area_norm','center_distance_norm','size_distance_norm'],'folds':{str(f):{'rows':int((fold==f).sum()),'positive':int(((fold==f)&(y>0.5)).sum()),'negative':int(((fold==f)&(y<0.5)).sum()),'videos':sorted({m['video_id'] for i,m in enumerate(meta) if fold[i]==f})} for f in range(3)},'excluded_event_videos':sorted(x for x in blocked if x>=0),'labels_posthoc_only':True,'model_input_forbidden':['physical_id','semantic_id','category','future','text','gt_track_id','gt_bbox','event_key'],'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'stats':dict(stats)};atom(OUT/'manifests/physical_gate_examples.json',manifest);atom(OUT/'audit/physical_gate_build.json',{'manifest':str((OUT/'manifests/physical_gate_examples.json').resolve()),'rows':len(y),'stats':dict(stats),'public_dev_q1_sealed_accessed':False});print(json.dumps({'rows':len(y),'stats':dict(stats),'folds':manifest['folds']},indent=2,sort_keys=True))
if __name__=='__main__':main()
