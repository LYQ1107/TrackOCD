#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase32'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 raw=json.load(open(ROOT/'outputs/iclr27_phase31/metrics/compatibility_diagnostic.json')); agg=raw['raw_comparator']; result={'protocol':'trackocd_iclr27_phase32_routed_unchanged_controller','proposal':'Phase26 frozen 41/76','controller':'Phase19R RC-MS-OCD unchanged','routing':'Phase31 pair order; physical stream has one selected raw vector per row, so routed raw-vector sequence is identical to comparator','main_aggregate':agg,'routed_aggregate':agg,'folds':raw['folds'],'gate_c32':{'pass':False,'decision':'P32_GATE_C32_FAIL','reason':'Routed sequence is identical to raw under the one-candidate-per-physical-row stream; Commit-CT 1/76 does not strictly exceed frozen Phase28 3/76 and coverage is narrow.'},'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','future','held GT input','IDs/text']}
 atomic(OUT/'metrics/routed_compatibility.json',result); atomic(OUT/'audit/decision.json',result['gate_c32']); atomic(OUT/'completion/stage2.done',{'stage':2,'gate_c32':'FAIL','commit_ct':'1/76'}); print(json.dumps({'gate_c32':'FAIL','commit_ct':'1/76'},indent=2))
if __name__=='__main__': main()
