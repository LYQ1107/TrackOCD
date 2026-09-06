#!/usr/bin/env python3
"""Parameter-free Root-of-Fragments diagnostic on the frozen Phase75D table."""
from __future__ import annotations
import hashlib, json, os, tempfile, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, l2_normalize
from src.iclr27_phase75d.retrieval_metrics import score_records
OUT=ROOT/'outputs/iclr27_phase86'; CFG=ROOT/'configs/iclr27_phase86/rf_v1.json'; EP=ROOT/'outputs/iclr27_phase30/manifests'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:
   json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def frags(seq,size=4):
 if len(seq)==0: return [np.zeros((768,),np.float32)]
 return [l2_normalize(np.asarray(seq[i:i+size],np.float32).mean(0)) for i in range(0,len(seq),size)]
def chamfer(q,c):
 q=np.asarray(q,np.float32); c=np.asarray(c,np.float32); sim=q@c.T
 return float(.5*(np.max(sim,axis=1).mean()+np.max(sim,axis=0).mean()))
def keys_for_fold(f):
 d=json.loads((EP/f'episode_manifest_f{f}.json').read_text())
 return sorted({str(r['query_track_key']) for r in d['records'] if r.get('split')=='val'})
def main():
 table=load_frozen_tracks(); fold_rows=[]; prefix_rows=[]; frag_cache={}
 def fs(k,p):
  key=(k,p)
  if key not in frag_cache: frag_cache[key]=frags(table.get_frame_sequence(k,p))
  return frag_cache[key]
 for f in range(4):
  keys=[k for k in keys_for_fold(f) if k in table.metadata]; vids=np.asarray([table.metadata[k]['video'] for k in keys]); cats=np.asarray([table.metadata[k]['category'] for k in keys])
  for p in PREFIXES:
   rec=[]
   for i,qk in enumerate(keys):
    cand_idx=np.where((np.arange(len(keys))!=i)&(vids!=vids[i]))[0]; cands=[keys[int(j)] for j in cand_idx]; positives=[keys[int(j)] for j in cand_idx if cats[int(j)]==cats[i]]; negatives=[keys[int(j)] for j in cand_idx if cats[int(j)]!=cats[i]]; qfr=fs(qk,p); scores=[]; raws=[]
    for ck in cands:
     raw=float(table.raw_vector(qk,p)@table.raw_vector(ck,16)); raws.append(raw); scores.append(raw+.05*np.tanh(chamfer(qfr,fs(ck,16))-raw))
    rec.append({'query_key':qk,'category':int(cats[i]),'video':int(vids[i]),'candidates':cands,'positives':positives,'negatives':negatives,'scores':scores,'raw_scores':raws})
   m=score_records(rec);m.pop('per_query',None);fold_rows.append({'fold':f,'prefix':p,'queries':len(rec),'metrics':m});prefix_rows.append({'fold':f,'prefix':p,'r1':m['r1'],'raw_r1':m['raw_r1'],'map':m['map'],'raw_map':m['raw_map'],'hard_gap':m['hard_negative_gap'],'raw_hard_gap':m['raw_hard_negative_gap'],'unsafe':m['unsafe_flip_count']});print(json.dumps({'fold':f,'prefix':p,'r1':m['r1'],'raw_r1':m['raw_r1'],'map':m['map'],'unsafe':m['unsafe_flip_count']},sort_keys=True),flush=True)
 aggregate=[]
 for p in PREFIXES:
  rows=[x['metrics'] for x in fold_rows if x['prefix']==p]; n=sum(m['queries'] for m in rows); aggregate.append({'prefix':p,'aggregate':{'r1':float(np.mean([m['r1'] for m in rows])),'raw_r1':float(np.mean([m['raw_r1'] for m in rows])),'map':float(np.mean([m['map'] for m in rows])),'raw_map':float(np.mean([m['raw_map'] for m in rows])),'hard_negative_gap':float(np.mean([m['hard_negative_gap'] for m in rows])),'raw_hard_negative_gap':float(np.mean([m['raw_hard_negative_gap'] for m in rows])),'queries':n,'unsafe_flip_count':sum(m['unsafe_flip_count'] for m in rows),'unsafe_flip_rate':sum(m['unsafe_flip_count'] for m in rows)/max(n,1)}})
 p16=[x for x in fold_rows if x['prefix']==16]; temporal=json.loads((ROOT/'outputs/iclr27_phase85/metrics/physical_r_comparison.json').read_text()) if (ROOT/'outputs/iclr27_phase85/metrics/physical_r_comparison.json').exists() else {}; baseline_unsafe=temporal.get('p16',{}).get('unsafe_flip_count');rf=next(x['aggregate'] for x in aggregate if x['prefix']==16); nondec=sum(int(x['metrics']['r1']>=x['metrics']['raw_r1'] and x['metrics']['map']>=x['metrics']['raw_map']) for x in p16);decision='ROOT_OF_FRAGMENTS_V1_PASS_TRAIN_VALIDATION' if nondec>=3 and (baseline_unsafe is None or rf['unsafe_flip_count']<baseline_unsafe) else 'ROOT_OF_FRAGMENTS_V1_NEGATIVE';out={'schema_version':'trackocd.phase86.rf_metrics.v1','phase':86,'route':'H86-RF_ROOT_OF_FRAGMENTS','config':str(CFG.resolve()),'config_sha256':sha(CFG),'fragment_size':4,'fold_rows':fold_rows,'prefix_rows':prefix_rows,'aggregate':aggregate,'p16_non_decreasing_folds':nondec,'temporal_root_unsafe_reference':baseline_unsafe,'decision':decision,'training':False,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False,'diagnostic_only':True,'controller_run':False,'sealed_run':False}
 atom(OUT/'metrics/rf_metrics.json',out);atom(OUT/'audit/rf_decision.json',out);atom(OUT/'completion/rf_diagnostic.done',{'status':'DONE','metrics':str((OUT/'metrics/rf_metrics.json').resolve()),'sha256':sha(OUT/'metrics/rf_metrics.json')});print(json.dumps({'decision':decision,'p16':rf,'nondec':nondec},indent=2,sort_keys=True))
if __name__=='__main__': main()
