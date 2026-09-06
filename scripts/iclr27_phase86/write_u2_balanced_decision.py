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
  o=json.loads((OUT/'metrics'/f'u2_balanced_cv_f{f}.json').read_text()); rows.append({'fold':f,'validation':o['validation_metrics'],'checkpoint':o['checkpoint'],'checkpoint_sha256':o['checkpoint_sha256']})
 out={'schema_version':'trackocd.phase86.u2_balanced_decision.v1','phase':86,'route':'U2_SET_AWARE_RELATION_BALANCED','status':'U2_BALANCED_TRAIN_NEGATIVE','cv_rows':rows,'repair':'one registered minimal repair: balanced match/defer group exposure; architecture, raw anchor, seed, candidate universe and protocol unchanged','gate':{'required':'at least 2/3 folds net_rescue>0 and harm<=rescue','passed':False,'observed':[{'fold':r['fold'],'net_rescue':r['validation']['net_rescue'],'harm':r['validation']['harm'],'rescue':r['validation']['rescue']} for r in rows]},'root_cause_after_repair':'all three validation folds remained exactly raw top1/top5; the set-aware Transformer emitted no ranking changes, so the issue is not only defer imbalance and this route is closed after one repair cycle','event_replay_run':False,'controller_run':False,'sealed_run':False,'public_dev_q1_sealed_accessed':False,'next_action':'Do not launch a third U2 variant. Preserve U1/RF/U2 negative evidence and use the verified method audit to document the remaining legal support/representation gap.'}
 atom(OUT/'audit/u2_decision.json',out);atom(OUT/'status_u2.json',out);print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
