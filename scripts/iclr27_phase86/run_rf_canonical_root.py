#!/usr/bin/env python3
"""Evaluate the registered RF score on causal canonical physical roots."""
from __future__ import annotations
import argparse,csv,datetime as dt,hashlib,json,os,tempfile,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES,load_frozen_tracks
from src.iclr27_phase75d.retrieval_metrics import score_records
from src.iclr27_phase86.causal_root_fragments import make_builder
from src.iclr27_phase23.protocol import load_rows
OUT=ROOT/'outputs/iclr27_phase86'; EP=ROOT/'outputs/iclr27_phase30/manifests'; LINEAGE=Path('/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/full_temporal_lineage.jsonl'); UNIONS=Path('/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/temporal_mean_full/union_events.jsonl')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def chamfer(q,c):
 q=np.asarray(q,np.float32); c=np.asarray(c,np.float32); q=q if len(q) else np.zeros((1,768),np.float32); c=c if len(c) else np.zeros((1,768),np.float32); sim=q@c.T; return float(.5*(np.max(sim,axis=1).mean()+np.max(sim,axis=0).mean()))
def fold_keys(f):
 m=json.loads((EP/f'episode_manifest_f{f}.json').read_text()); return sorted({str(r['query_track_key']) for r in m['records'] if r.get('split')=='val'})
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tag',default='rf_canonical_root_v1'); a=ap.parse_args()
 table=load_frozen_tracks(); public=load_rows(); builder,join=make_builder(table,public,LINEAGE,UNIONS); fold_rows=[]; p16_rows=[]; root_stats=[]
 for f in range(4):
  keys=[k for k in fold_keys(f) if k in table.metadata]; vids=np.asarray([table.metadata[k]['video'] for k in keys]); cats=np.asarray([table.metadata[k]['category'] for k in keys])
  for p in PREFIXES:
   rec=[]
   cache={(k,p):builder.build(k,p) for k in keys}; cache.update({(k,16):builder.build(k,16) for k in keys})
   for i,qk in enumerate(keys):
    cand_idx=np.where((np.arange(len(keys))!=i)&(vids!=vids[i]))[0]; cands=[keys[int(j)] for j in cand_idx]; pos=[keys[int(j)] for j in cand_idx if cats[int(j)]==cats[i]]; neg=[keys[int(j)] for j in cand_idx if cats[int(j)]!=cats[i]]; q=cache[(qk,p)]; scores=[]; raws=[]
    for ck in cands:
     c=cache[(ck,16)]; raw=float(q['raw_anchor_vector']@c['raw_anchor_vector']); raws.append(raw); scores.append(raw+.05*np.tanh(chamfer(q['fragment_vectors'],c['fragment_vectors'])-raw)); root_stats.append({'query':qk,'candidate':ck,'q_fragments':len(q['fragment_ids']),'c_fragments':len(c['fragment_ids'])})
    rec.append({'query_key':qk,'category':int(cats[i]),'video':int(vids[i]),'candidates':cands,'positives':pos,'negatives':neg,'scores':scores,'raw_scores':raws})
   met=score_records(rec); fold_rows.append({'fold':f,'prefix':p,'metrics':met}); p16_rows.append({'fold':f,'prefix':p,'r1':met['r1'],'raw_r1':met['raw_r1'],'map':met['map'],'raw_map':met['raw_map'],'hard_gap':met['hard_negative_gap'],'raw_hard_gap':met['raw_hard_negative_gap'],'unsafe':met['unsafe_flip_count']}); print(json.dumps({'fold':f,'prefix':p,'r1':met['r1'],'raw_r1':met['raw_r1'],'map':met['map'],'unsafe':met['unsafe_flip_count']},sort_keys=True),flush=True)
 aggregate=[]
 for p in PREFIXES:
  fs=[x['metrics'] for x in fold_rows if x['prefix']==p]; aggregate.append({'prefix':p,'aggregate':{'r1':float(np.mean([x['r1'] for x in fs])),'raw_r1':float(np.mean([x['raw_r1'] for x in fs])),'map':float(np.mean([x['map'] for x in fs])),'raw_map':float(np.mean([x['raw_map'] for x in fs])),'hard_negative_gap':float(np.mean([x['hard_negative_gap'] for x in fs])),'raw_hard_negative_gap':float(np.mean([x['raw_hard_negative_gap'] for x in fs])),'queries':sum(x['queries'] for x in fs),'unsafe_flip_count':sum(x['unsafe_flip_count'] for x in fs),'unsafe_flip_rate':sum(x['unsafe_flip_count'] for x in fs)/max(1,sum(x['queries'] for x in fs))}})
 p16=[x for x in fold_rows if x['prefix']==16]; rf=next(x['aggregate'] for x in aggregate if x['prefix']==16); nondec=sum(int(x['metrics']['r1']>=x['metrics']['raw_r1'] and x['metrics']['map']>=x['metrics']['raw_map']) for x in p16); mean_frag=float(np.mean([max(x['q_fragments'],x['c_fragments']) for x in root_stats])) if root_stats else 0.; frac_multi=float(np.mean([int(max(x['q_fragments'],x['c_fragments'])>1) for x in root_stats])) if root_stats else 0.; decision='RF_CANONICAL_ROOT_V1_PASS_TRAIN_VALIDATION' if nondec>=3 and rf['unsafe_flip_count']==0 else 'RF_CANONICAL_ROOT_V1_NEGATIVE'; out={'schema_version':'trackocd.phase86.rf_canonical_metrics.v1','phase':86,'tag':a.tag,'route':'RF_CANONICAL_ROOT_V1','aggregate':aggregate,'fold_rows':fold_rows,'prefix_rows':p16_rows,'p16_non_decreasing_folds':nondec,'decision':decision,'root_reconstruction':{'lineage':str(LINEAGE.resolve()),'lineage_sha256':sha(LINEAGE),'union_events':str(UNIONS.resolve()),'union_events_sha256':sha(UNIONS),'join':join,'mean_fragments_per_root_comparison':mean_frag,'fraction_roots_gt1_fragment':frac_multi,'max_fragments_per_root':max([max(x['q_fragments'],x['c_fragments']) for x in root_stats],default=0),'root_comparisons':len(root_stats),'fallback_count':sum(int(x['q_fragments']==0 or x['c_fragments']==0) for x in root_stats)},'config':{'residual_scale':.05,'query_prefix':'p','candidate_prefix':16,'score':'raw_cosine + 0.05*tanh(symmetric_fragment_score - raw_cosine)'},'training':False,'diagnostic_only':True,'controller_run':False,'sealed_run':False,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False}
 atom(OUT/f'metrics/{a.tag}.json',out); atom(OUT/'audit/rf_canonical_decision.json',out); atom(OUT/f'completion/{a.tag}.done',{'status':'DONE','metrics':str((OUT/f'metrics/{a.tag}.json').resolve()),'sha256':sha(OUT/f'metrics/{a.tag}.json')}); print(json.dumps({'decision':decision,'p16':rf,'nondec':nondec,'root_stats':out['root_reconstruction']},indent=2,sort_keys=True))
if __name__=='__main__': main()
