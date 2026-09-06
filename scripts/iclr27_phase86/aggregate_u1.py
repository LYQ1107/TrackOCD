#!/usr/bin/env python3
"""Aggregate U1 OOF validation and one event replay into a fixed decision."""
from __future__ import annotations
import hashlib,json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase86'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 fs=[]
 for f in range(3):
  p=OUT/f'metrics/u1_r1_formal_f{f}.json'; fs.append(json.loads(p.read_text()))
 evp=OUT/'metrics/u1_formal_replay_r1.json'; ev=json.loads(evp.read_text()); p16={x['polarity']:x for x in ev['summary'] if x['prefix']==16}
 gate=[]
 for d in fs:
  v=d['validation_metrics']; gate.append({'fold':d['fold'],'rescue':v['rescue'],'harm':v['harm'],'net_rescue':v['net_rescue'],'pass':bool(v['net_rescue']>0 and v['harm']<=v['rescue'])})
 train_pass=sum(int(x['pass']) for x in gate)>=2
 event_pass=bool(p16['positive']['u1_reliable_events']>=22 and p16['negative']['u1_reliable_events']<=8)
 obj={'schema_version':'trackocd.phase86.u1_decision.v1','phase':86,'route':'U1_OOF_UTILITY_GATE','train_validation_folds':gate,'train_gate_pass':train_pass,'formal_event_replay':{'positive_p16':p16['positive'],'negative_p16':p16['negative'],'registered_status_criteria':{'positive_min':22,'negative_max':8},'status_pass':event_pass,'metrics':str(evp.resolve()),'metrics_sha256':sha(evp)},'decision':'U1_SAFE_TRAIN_BUT_EVENT_SAFETY_FAIL' if train_pass and not event_pass else 'U1_GATE_FAIL' if not train_pass else 'U1_EVENT_STATUS_PASS','controller_run':False,'sealed_run':False,'diagnostic_only':True,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False,'next_action':'Do not run controller from this score-only stream; execute the preregistered RF diagnostic. U2 remains conditional on TRAIN gate failure, not opened by event score alone.'}
 atom(OUT/'audit/u1_decision.json',obj);atom(OUT/'metrics/u1_summary.json',obj);print(json.dumps(obj,indent=2,sort_keys=True))
if __name__=='__main__':main()
