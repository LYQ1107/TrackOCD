#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase34'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 raw=json.load(open(ROOT/'outputs/iclr27_phase31/metrics/compatibility_diagnostic.json')); agg=raw['raw_comparator']; out={'protocol':'trackocd_phase34_reranker_weighted_prototype_bridge_compatibility','proposal':'Phase26 frozen 41/76','controller':'Phase19R RC-MS-OCD unchanged','bridge':{'top_k':5,'temperature':0.1,'beta':0.25,'held_event_supports':'none in TRAIN-only manifest; exact no-support identity fallback'},'main_aggregate':agg,'folds':raw['folds'],'gate_c34':{'pass':False,'decision':'P34_GATE_C34_FAIL','reason':'All held event tracks use no legal TRAIN support, so bridge exactly degenerates to raw vectors; Commit-CT 1/76 does not exceed frozen 3/76.'},'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT input']}; atomic(OUT/'metrics/bridge_compatibility.json',out); atomic(OUT/'audit/decision.json',out['gate_c34']); atomic(OUT/'completion/stage2.done',{'stage':2,'gate_c34':'FAIL','commit_ct':'1/76'}); print(json.dumps(out['gate_c34'],indent=2))
if __name__=='__main__': main()
