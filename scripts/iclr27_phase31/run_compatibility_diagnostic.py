#!/usr/bin/env python3
"""Read-only compatibility audit using the frozen Phase19R controller comparator.

The Phase31 reranker emits pair scores, while RC-MS-OCD consumes raw vectors and
state features.  Injecting pair scores would alter the registered controller
interface, so this script records the unchanged raw comparator and explicitly
marks reranker compatibility as not run rather than silently changing semantics.
"""
import json, tempfile, os
from pathlib import Path
import torch, numpy as np
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase31'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 torch.set_num_threads(1); folds=[]
 for fold in range(4):
  d=Phase19RData(fold); r=evaluate_candidate('raw',d,None,torch.device('cpu')); folds.append({'fold':fold,'metrics':r['metrics'],'known_metrics':r['known_metrics'],'events':r['events']})
 agg={'commit_ct_correct':sum(f['metrics']['commit_ct']['correct'] for f in folds),'commit_ct_eligible':sum(f['metrics']['commit_ct']['eligible'] for f in folds),'category_coverage_sum':sum(f['metrics']['category_coverage'] for f in folds),'video_coverage_sum':sum(f['metrics']['video_coverage'] for f in folds),'existing_precision_mean':float(np.mean([f['metrics']['existing_precision'] for f in folds])),'existing_recall_mean':float(np.mean([f['metrics']['existing_recall'] for f in folds])),'negative_false_merge_mean':float(np.mean([f['metrics']['negative_false_merge_rate'] for f in folds])),'duplicate_births':sum(f['metrics']['duplicate_births'] for f in folds),'premature_rate_mean':float(np.mean([f['metrics']['premature_rate'] for f in folds])),'unresolved_rate_mean':float(np.mean([f['metrics']['unresolved_rate'] for f in folds]))}
 out={'protocol':'trackocd_iclr27_phase31_unchanged_controller_compatibility_diagnostic','proposal_frozen':'Phase26','controller_frozen':'Phase19R RC-MS-OCD','raw_comparator':agg,'folds':folds,'reranker_compatibility':'NOT_RUN_INTERFACE_INCOMPATIBLE_WITHOUT_CONTROLLER_CHANGE','gate_c31':'NOT_RUN','reason':'Reranker pair scores require query-state metadata not exposed by the frozen controller; injecting them would alter action semantics, which Phase31 forbids. Raw comparator is retained for safety/CT context only.','sealed_inputs_not_read':['DEV+','Q1','public new-model labels','future','held GT input','IDs/text']}
 atomic(OUT/'metrics/compatibility_diagnostic.json',out); atomic(OUT/'audit/compatibility_decision.json',{'gate_c31':'NOT_RUN','raw_commit_ct':agg['commit_ct_correct'],'protocol_preserved':True}); atomic(OUT/'completion/compatibility.done',{'gate_c31':'NOT_RUN','raw_commit_ct':agg['commit_ct_correct']}); print(json.dumps({'raw':agg,'gate_c31':'NOT_RUN'},indent=2))
if __name__=='__main__': main()
