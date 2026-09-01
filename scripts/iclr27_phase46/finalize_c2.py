#!/usr/bin/env python3
import json,glob,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase46'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 ds=[json.load(open(OUT/'metrics'/f'controller_f{f}.json')) for f in range(4)]; main=[d['main']['metrics'] for d in ds]; raw=[d['raw']['metrics'] for d in ds]
 agg={'commit_ct_correct':sum(x['commit_ct']['correct'] for x in main),'commit_ct_eligible':sum(x['commit_ct']['eligible'] for x in main),'category_coverage_sum':sum(x['category_coverage'] for x in main),'video_coverage_sum':sum(x['video_coverage'] for x in main),'existing_precision_mean':sum(x['existing_precision'] for x in main)/4,'existing_recall_mean':sum(x['existing_recall'] for x in main)/4,'negative_false_merge_mean':sum(x['negative_false_merge_rate'] for x in main)/4,'duplicate_births':sum(x['duplicate_births'] for x in main),'premature_rate_mean':sum(x['premature_rate'] for x in main)/4,'unresolved_rate_mean':sum(x['unresolved_rate'] for x in main)/4,'known_micro_mean':sum(x.get('known_micro',0) for x in main)/4,'known_macro_mean':sum(x.get('known_macro',0) for x in main)/4}
 rawagg={'commit_ct_correct':sum(x['commit_ct']['correct'] for x in raw),'commit_ct_eligible':sum(x['commit_ct']['eligible'] for x in raw),'negative_false_merge_mean':sum(x['negative_false_merge_rate'] for x in raw)/4,'duplicate_births':sum(x['duplicate_births'] for x in raw)}
 atomic(OUT/'audit/phase46_c2_decision.json',{'phase':46,'gate_c46':'FAIL','decision_code':'P46_GATE_C_FAIL_NARROW_CT_INTERFACE_MISMATCH_STOP_BEFORE_SEALED','main_aggregate':agg,'raw_aggregate':rawagg,'fold_main_commit_ct':[x['commit_ct'] for x in main],'fold_raw_commit_ct':[x['commit_ct'] for x in raw],'safety_evidence':{'main_false_merge':[x['negative_false_merge_rate'] for x in main],'raw_false_merge':[x['negative_false_merge_rate'] for x in raw],'main_duplicate_births':[x['duplicate_births'] for x in main],'raw_duplicate_births':[x['duplicate_births'] for x in raw],'main_premature':[x['premature_rate'] for x in main],'raw_premature':[x['premature_rate'] for x in raw]},'root_cause':'support-conditioned Phase46 row vectors are not compatible with raw-known-prototype/StateMemory controller interface; gain is confined to fold3 and safety coverage is not broad','next_stage':'single class-agnostic correspondence/interface repair with frozen controller','controller_changed':False,'threshold_sweep':False,'sealed_labels_accessed':False,'public_q1_dev_access':False})
 atomic(OUT/'completion/c2.done',{'gate_c46':'FAIL','commit_ct':f"{agg['commit_ct_correct']}/76"})
 print(json.dumps({'gate_c46':'FAIL','commit_ct':f"{agg['commit_ct_correct']}/76"},indent=2))
if __name__=='__main__': main()
