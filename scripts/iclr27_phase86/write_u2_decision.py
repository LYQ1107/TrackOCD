#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase86'
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 rows=[]
 for f in range(3):
  p=OUT/'metrics'/f'u2_cv_f{f}.json';o=json.loads(p.read_text()); rows.append({'fold':f,'validation':o['validation_metrics'],'checkpoint':o['checkpoint'],'checkpoint_sha256':o['checkpoint_sha256']})
 out={'schema_version':'trackocd.phase86.u2_decision.v1','phase':86,'route':'U2_SET_AWARE_RELATION','status':'U2_TRAIN_NEGATIVE','cv_rows':rows,'train_gate':{'required':'at least 2/3 folds net_rescue>0 and harm<=rescue','passed':False,'observed':[{'fold':r['fold'],'net_rescue':r['validation']['net_rescue'],'harm':r['validation']['harm'],'rescue':r['validation']['rescue']} for r in rows]},'root_cause_evidence':{'all_folds_residual_effective_zero':True,'observed_residual':'checkpoint f0 delta mean=-0.05, std≈1e-8, max_abs=0.05','fit_defer_dominance':'match groups are a small minority in the registered feature manifest; preservation loss dominates','candidate_availability':'raw top1/top5 remain unchanged in all three validation folds; no candidate ranking gain is observed','not_a_sealed_result':True},'event_replay_run':False,'controller_run':False,'sealed_run':False,'public_dev_q1_sealed_accessed':False,'next_action':'Keep U2-v1 negative evidence; complete verified recent-method audit and select at most one distinct high-information route, without changing the frozen event protocol or invoking controller.'}
 atom(OUT/'audit/u2_decision.json',out);atom(OUT/'status_u2.json',out);print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
