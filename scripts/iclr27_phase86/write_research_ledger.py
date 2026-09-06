#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt,json,os,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase86/audit/research_ledger.json'
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip(); now=dt.datetime.now(dt.timezone.utc).isoformat()
 entries=[
 {'stage':'PREMATURE_FINALIZATION_REPAIR','hypothesis':'report generation must be locked to the original deadline window','start_time':now,'end_time':now,'code_head':head,'result':'guard implemented; early invocation raises FINALIZATION_TOO_EARLY_RESEARCH_MUST_CONTINUE','decision':'REPAIR_KEPT','next_action':'continue until unlock','remaining_seconds':None},
 {'stage':'U1_FINAL_ALLTRAIN_DEPLOYMENT','hypothesis':'one all-TRAIN expert plus one all-OOF gate removes event-fold modulo deployment bias','code_head':head,'inputs':['Phase85 support manifest','Phase85 OOF metadata'],'result':'p16 positive 20/76, negative 12/76','decision':'U1_FINAL_EVENT_SAFETY_FAIL','next_action':'do not connect controller'},
 {'stage':'RF_CANONICAL_ROOT_V1','hypothesis':'causal union-root fragment prototypes improve physical-to-R correspondence','code_head':head,'inputs':['Phase85 temporal lineage','Phase85 union events'],'result':'p16 R@1 .893219, mAP .849139, unsafe 1, 2/4 folds non-decreasing','decision':'RF_CANONICAL_ROOT_V1_NEGATIVE','next_action':'close RF-v1'},
 {'stage':'U2_SET_AWARE_RELATION','hypothesis':'set-aware candidate relation improves raw ranking','code_head':head,'result':'three-fold TRAIN rescue/harm all 0/0; constant residual','decision':'U2_TRAIN_NEGATIVE','next_action':'one balanced-exposure repair only'},
 {'stage':'U2_SET_AWARE_RELATION_BALANCED','hypothesis':'match/defer exposure imbalance is the first U2 root cause','code_head':head,'result':'three-fold TRAIN rescue/harm still 0/0; raw top-1/top-5 unchanged','decision':'U2_BALANCED_TRAIN_NEGATIVE','next_action':'no third U2 variant; use recent-method audit'},
 {'stage':'RECENT_METHOD_AUDIT','hypothesis':'an audited 2025/26 method may provide a legal causal support selector','code_head':head,'result':'no drop-in text-free causal support-set interface; no external code/weights imported','decision':'NO_VERIFIED_DROP_IN_METHOD','next_action':'preserve negative evidence and finalize only in unlocked interval'},
 ]
 atom(OUT,{'schema_version':'trackocd.phase86.research_ledger.v2','phase':86,'window_registration':'outputs/iclr27_phase86/audit/window_registration.json','entries':entries,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_or_text_as_model_input':False,'external_processes_touched':[],'resource_note':'CPU routes; GPU0 PID33785 and external intermot workers untouched'})
 print(str(OUT.resolve()))
if __name__=='__main__':main()
