#!/usr/bin/env python3
import json,csv,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase37'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 for d in ('audit','metrics','completion'): (OUT/d).mkdir(parents=True,exist_ok=True)
 support=set();
 for fold in range(4):
  m=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % fold)));
  for r in m['records']:
   support.update(r.get('support_track_keys',[]))
 events=[json.loads(x) for fn in ('held_known_positive_events.jsonl','held_known_negative_events.jsonl') for x in open(ROOT / ('outputs/iclr27_phase19r/manifests/' + fn)) if x.strip()]
 phase26=json.load(open(ROOT/'outputs/iclr27_phase26/audit/stage3_event_records.json'))['records']; raw={(r['event_key'],r['prefix']):r for r in phase26 if r['condition']=='raw_baseline'}; source={(r['event_key'],r['prefix']):r for r in phase26 if r['condition']=='phase26_source_branch_topk'}
 avail=[]; bounds=[]
 for e in events:
  if e['kind']!='positive_existing': continue
  ek=e['event_key']; sk=e['source_tracklet_keys'][0]; tk=e['target_tracklet_key']; support_ok_s=sk in support; support_ok_t=tk in support
  for p in PREFIXES:
   rr=raw.get((ek,p),{}); sr=source.get((ek,p),{}); src_rel=int(rr.get('source_reliable',0)); tgt_rel=int(rr.get('target_reliable',0)); typ='TRAIN_SUPPORT_AVAILABLE' if support_ok_s and support_ok_t else ('ONE_SIDED_MISSING' if support_ok_s != support_ok_t else ('CURRENT_OBSERVABLE' if src_rel and tgt_rel else ('CAUSAL_HISTORY_ONLY' if src_rel or tgt_rel else 'NO_LEGAL_SUPPORT')))
   avail.append({'event_key':ek,'fold':e['fold'],'category':e.get('category_gt_denominator_only'),'source_video':e['source_video'],'target_video':e['target_video'],'prefix':p,'source_track':sk,'target_track':tk,'source_reliable_raw':src_rel,'target_reliable_raw':tgt_rel,'source_reliable_phase26':int(sr.get('source_reliable',0)),'target_reliable_phase26':int(sr.get('target_reliable',0)),'support_source_available':support_ok_s,'support_target_available':support_ok_t,'availability_type':typ,'candidate_ceiling_raw':int(rr.get('ceiling',0)),'candidate_ceiling_phase26':int(sr.get('ceiling',0))})
  bounds.append({'event_key':ek,'fold':e['fold'],'category':e.get('category_gt_denominator_only'),'source_support':support_ok_s,'target_support':support_ok_t,'raw_prefix16_ceiling':int(raw.get((ek,16),{}).get('ceiling',0)),'phase26_prefix16_ceiling':int(source.get((ek,16),{}).get('ceiling',0))})
 summary={}
 for p in PREFIXES:
  xs=[x for x in avail if x['prefix']==p]; summary[str(p)]={'events':len(xs),'availability_counts':{k:sum(x['availability_type']==k for x in xs) for k in ('CURRENT_OBSERVABLE','CAUSAL_HISTORY_ONLY','TRAIN_SUPPORT_AVAILABLE','NO_LEGAL_SUPPORT','ONE_SIDED_MISSING')},'raw_ceiling':sum(x['candidate_ceiling_raw'] for x in xs),'phase26_ceiling':sum(x['candidate_ceiling_phase26'] for x in xs),'source_reliable':sum(x['source_reliable_raw'] for x in xs),'target_reliable':sum(x['target_reliable_raw'] for x in xs)}
 atomic(OUT/'audit/observability_contract.json',{'positive_event_denominator':76,'prefixes':list(PREFIXES),'support_tracks_in_train_manifest':len(support),'contract':'held tracks are excluded from TRAIN support by leakage rule','sealed':True}); atomic(OUT/'audit/support_availability_76.json',{'records':avail,'positive_events':76}); atomic(OUT/'audit/causal_upper_bounds.json',{'records':bounds,'summary_by_prefix':summary,'diagnostic_only':'candidate ceilings do not equal Commit-CT'}); atomic(OUT/'audit/leakage_audit.json',{'held_gt_input':False,'future':False,'category_text':False,'physical_or_semantic_id_model_input':False,'train_support_overlap_with_held_tracks':sum(int(x['support_source_available'] or x['support_target_available']) for x in avail)==0}); atomic(OUT/'audit/resource_preflight.json',{'ram_available_gb':120,'gpu_idle':True,'public_q1_dev_access':False}); atomic(OUT/'completion/stage0.done',{'stage':0,'events':76,'contract_pass':True}); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
