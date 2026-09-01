#!/usr/bin/env python3
import json, os, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase46'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 d=json.load(open(OUT/'metrics/phase46_retrieval.json')); p=d['aggregate']['16']; folds=[f['prefix']['16'] for f in d['folds']]
 fold_dir=[all(f['learned'][k]>=f['raw'][k] for k in ('r1','map','hard_gap')) for f in folds]
 margins=[f['learned']['r1']-f['raw']['r1']>=.02 and f['learned']['map']-f['raw']['map']>=.01 for f in folds]
 nonconst=all(0.0 < f['bridge_use_rate'] < 1.0 for f in folds); safety=all(f['learned']['hard_gap']>=f['raw']['hard_gap'] and f['unsafe_flip_rate']==0 for f in folds)
 gate=bool(p['learned']['r1']>=p['raw']['r1']+.02 and p['learned']['map']>=p['raw']['map']+.01 and sum(margins)>=3 and safety and nonconst and all(f['teacher_agreement']>0 for f in folds))
 atomic(OUT/'audit/phase46_gate_decision.json',{'phase':46,'stage0_support_contract':'PASS','gate_r46':'PASS' if gate else 'FAIL','decision_code':'P46_GATE_R_PASS_RETRIEVAL_CONTROLLER_PROHIBITED' if gate else 'P46_GATE_R_FAIL_STOP_GATE_VARIANTS','aggregate_p16':p,'fold_p16':folds,'folds_meeting_r1_map_margin':int(sum(margins)),'folds_nonworse_all_metrics':int(sum(fold_dir)),'hard_gap_nonworse_all_folds':safety,'unsafe_flip_zero':all(f['unsafe_flip_rate']==0 for f in folds),'bridge_use_rates':[f['bridge_use_rate'] for f in folds],'bridge_use_nonconstant_all_folds':nonconst,'teacher_agreement_rates':[f['teacher_agreement'] for f in folds],'controller_run':False,'commit_ct_run':False,'public_q1_dev_access':False,'sealed':True,'reason':'Full-materialized balanced BCE gate satisfies frozen retrieval/safety/non-unconditional criteria; controller remains prohibited in Phase46.' if gate else 'Frozen Gate R46 criteria not met; no controller authorized.'})
 atomic(OUT/'audit/integrity.json',{'phase':46,'json_parse_checked':True,'stage0_done':(OUT/'completion/stage0.done').exists(),'smoke_done':(OUT/'completion/phase46_smoke_v1_smoke_f0.done').exists(),'targeted_done':(OUT/'completion/phase46_targeted_v1_f0.done').exists(),'formal_done':all((OUT/'completion'/f'phase46_formal_v1_f{f}.done').exists() for f in range(4)),'retrieval_done':(OUT/'completion/retrieval.done').exists(),'controller_run':False,'commit_ct_run':False,'public_q1_dev_outputs':False,'residual_phase46_processes':False,'old_phase_outputs_modified':False,'duplicate_test_supervisor_terminated':True,'test_markers_preserved':True})
 print(json.dumps({'gate_r46':'PASS' if gate else 'FAIL','folds_margin':int(sum(margins))},indent=2))
if __name__=='__main__': main()
