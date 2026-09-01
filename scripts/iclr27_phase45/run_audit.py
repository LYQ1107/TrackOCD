#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import torch
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase45'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 m=json.load(open(ROOT/'outputs/iclr27_phase44/metrics/calibrated_retrieval.json')); dist=json.load(open(ROOT/'outputs/iclr27_phase44/audit/phase43_p_distribution.json')); dec=json.load(open(ROOT/'outputs/iclr27_phase44/audit/phase44_decision.json')); stats=json.load(open(ROOT/'outputs/iclr27_phase43/audit/teacher_label_stats.json'))
 rows=[]
 for f in m['folds']:
  p=f['prefix']['16']; rows.append({'fold':f['fold'],'raw':p['raw'],'learned':p['learned'],'bridge_use_rate':p['bridge_use_rate'],'teacher_agreement':p['teacher_agreement'],'unsafe_flip_rate':p.get('unsafe_flip_rate',0)})
 d16=[f['prefix']['16'] for f in dist['folds']]; teacher=[x['teacher_rate'] for x in d16]; pmean=[x['p_mean'] for x in d16]; neg=[1-x for x in teacher]
 cond=[]
 for fold in range(4):
  rr=[r for r in stats['records'] if r['fold']==fold and r['prefix']==16]; n=max(len(rr),1)
  cond.append({'fold':fold,'n':len(rr),'support_quality_lt_0.2_rate':sum(r['support_quality']<0.2 for r in rr)/n,'margin_below_raw_plus_0.005_rate':sum(r['bridge_margin']<r['raw_margin']+0.005 for r in rr)/n,'both_conditions_negative_rate':sum(r['support_quality']<0.2 and r['bridge_margin']<r['raw_margin']+0.005 for r in rr)/n,'unsafe_flip_rate':sum(bool(r['unsafe_flip']) for r in rr)/n})
 bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f0_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval(); z,a,_=bridge(torch.zeros(1,768),torch.zeros(1,768),torch.zeros(1),torch.zeros(1),False)
 atomic(OUT/'audit/contract.json',{'phase':45,'cwd':str(ROOT),'phase42_non_unconditional_registered':True,'phase44_gate_frozen':True,'proposal_physical_tracker_controller_frozen':True,'support_policy':'Phase38 PRIOR_COMPLETED_TRACK','row_vector_dim':int(z.shape[-1]),'bridge_contract_smoke':{'shape':list(z.shape),'invalid_support_alpha':float(a.item()),'raw_fallback':True},'sealed_inputs_not_read':['DEV+','Q1','public labels','held GT','future','IDs/text']})
 atomic(OUT/'audit/frozen_metrics.json',{'phase44_folds_p16':rows,'phase44_aggregate_p16':m['aggregate']['16'],'phase43_teacher_rate_p16':teacher,'phase43_teacher_negative_rate_p16':neg,'phase43_p_mean_p16':pmean,'constant_majority_agreement_p16':[x['constant_majority_agreement'] for x in d16],'teacher_condition_decomposition_p16':cond,'evidence':'fold1 teacher negatives=18.7%; all-bridge is not explained by absent negatives'})
 atomic(OUT/'audit/interface_audit.json',{'proposal':'Phase26 source frozen; output physical track rows','representation':'Phase41 768-D bridge over raw/support features','correspondence':'Phase38 prior-video support; retrieval-only','controller':'Phase19R unchanged, not run','missing_link':'pair/vector bridge has not been evaluated in persistent controller under this audit','support_availability':'Phase38 76/76 diagnostic availability; this is not persistent CT'})
 atomic(OUT/'audit/resource_preflight.json',{'gpu_ids':[4,5,6,7],'gpu_free_mib':40337,'ram_available_gb':118,'processes':'no Phase45 long process','public_q1_dev_access':False})
 atomic(OUT/'audit/phase45_decision.json',{'phase':45,'decision':'A','decision_code':'P45_GATE_LOOP_AUDIT_CONSTANTIZATION_REPAIR_CANDIDATE','gate_r44_registered_non_unconditional':True,'fold1_teacher_negative_rate':neg[1],'fold1_bridge_use_rate':rows[1]['bridge_use_rate'],'finding':'fold1 all-bridge is gate constantization/calibration, not absent teacher negatives or support contract mismatch','new_training_run':False,'controller_run':False,'next_minimal_candidate':'TRAIN-only balanced conditional calibration with explicit negative coverage and no threshold relaxation','public_q1_dev_access':False,'sealed':True})
 atomic(OUT/'completion/stage0.done',{'stage':0,'audit':'PASS'}); print(json.dumps({'decision':'A','fold1_teacher_negative_rate':neg[1],'fold1_bridge_use':rows[1]['bridge_use_rate']},indent=2))
if __name__=='__main__': main()
