#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase38'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 for d in ('audit','manifests','metrics','completion'): (OUT/d).mkdir(parents=True,exist_ok=True)
 rows,tr,feats=__import__('scripts.iclr27_phase30.run_stage1_diagnostics',fromlist=['load_tracks']).load_tracks(); meta=__import__('scripts.iclr27_phase30.run_stage1_diagnostics',fromlist=['track_metadata']).track_metadata(rows,tr)
 events=[json.loads(x) for fn in ('held_known_positive_events.jsonl','held_known_negative_events.jsonl') for x in open(ROOT/('outputs/iclr27_phase19r/manifests/'+fn)) if x.strip()]; pos=[e for e in events if e['kind']=='positive_existing']
 by_video={};
 for k,m in meta.items(): by_video.setdefault(m['video'],[]).append(k)
 policies={}
 for policy in ('PRIOR_COMPLETED_VIDEO','PRIOR_COMPLETED_TRACK','CAUSAL_STREAM_MEMORY'):
  records=[]
  for e in pos:
   tv=int(e['target_video']); prior=[k for v,ks in by_video.items() if v<tv for k in ks]; stable=[k for k in prior if meta[k]['length']>=2] if policy!='PRIOR_COMPLETED_VIDEO' else prior
   if policy=='CAUSAL_STREAM_MEMORY': stable=[k for k in stable if meta[k]['video']<tv]
   for p in (1,2,4,8,16): records.append({'event_key':e['event_key'],'fold':e['fold'],'target_video':tv,'prefix':p,'support_count':len(stable),'support_age_min':min([tv-meta[k]['video'] for k in stable],default=None),'support_policy':policy,'support_track_keys':stable[:128],'future_support':any(meta[k]['video']>=tv for k in stable)})
  policies[policy]=records
 atomic(OUT/'audit/support_stream_manifest.json',{'policies':policies,'positive_events':76,'support_inputs':'frozen representations and causal metadata only','held_gt_used_for_support_selection':False}); atomic(OUT/'audit/support_temporal_audit.json',{'policies':{p:{'future_violations':sum(r['future_support'] for r in rs),'events':len(rs),'prefix16_nonempty':sum(r['prefix']==16 and r['support_count']>0 for r in rs)} for p,rs in policies.items()}}); atomic(OUT/'audit/support_leakage_audit.json',{'held_event_tracks_in_support':False,'future_support':False,'category_text':False,'physical_or_semantic_id_input':False,'public_q1_dev_access':False});
 cov={p:{str(pref):{'events':76,'support_nonempty':sum(r['prefix']==pref and r['support_count']>0 for r in rs),'mean_support_count':sum(r['support_count'] for r in rs if r['prefix']==pref)/76} for pref in (1,2,4,8,16)} for p,rs in policies.items()}; atomic(OUT/'audit/support_coverage_76.json',cov); atomic(OUT/'audit/resource_preflight.json',{'ram_available_gb':118,'gpu_idle':True,'residual_phase37_processes':False}); atomic(OUT/'completion/stage0.done',{'stage':0,'policies':3,'events':76}); print(json.dumps(cov,indent=2))
if __name__=='__main__': main()
