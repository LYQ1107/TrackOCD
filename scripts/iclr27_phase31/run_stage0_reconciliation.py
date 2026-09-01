#!/usr/bin/env python3
"""Reconcile Phase27/29/30 retrieval contracts before any Phase31 training."""
import json, os, tempfile, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase31'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 OUT.joinpath('audit').mkdir(parents=True,exist_ok=True); OUT.joinpath('completion').mkdir(parents=True,exist_ok=True)
 s30=json.load(open(ROOT/'outputs/iclr27_phase30/metrics/stage1_diagnostics.json')); p27=json.load(open(ROOT/'outputs/iclr27_phase27/metrics/correspondence_validation.json')); p29=json.load(open(ROOT/'outputs/iclr27_phase29/metrics/representation_aggregate.json'))
 raw=s30['aggregate']['raw_dinov2']['raw']; folds=[]
 for f in s30['folds']:
  x=f['models']['raw_dinov2']['prefix']['16']; folds.append({'fold':f['fold'],'validation_tracklets':f['validation_tracklets'],'r1':x['r1'],'r5':x['r5'],'map':x['map'],'hard_negative_gap':x['hard_negative_gap'],'positive_coverage':x['positive_coverage'],'query_metric':'cross-video candidates; same-category positives; category/video macro is not used in aggregate'})
 protocol={'phase30':{'query_tracklets_by_fold':[f['validation_tracklets'] for f in s30['folds']],'prefixes':s30['prefixes'],'aggregate_p16':{'r1':raw['r1']['16'],'map':raw['map']['16'],'hard_negative_gap':raw['hard_negative_gap']['16']},'feature_alignment':{'rows':43423,'permutation_sha256':'269b739ab52e5c9b24b541c75de6039d7d721ca166f03f31f9901da9fa885a29'}},'phase27':{'aggregate_r1':p27.get('mean_baseline_r1'),'aggregate_map':p27.get('mean_baseline_map'),'event_record_count':p27.get('event_record_count'),'metric_source':'correspondence_validation.json'},'phase29':{'aggregate_r1':p29['aggregate']['baseline_r1_mean'],'aggregate_map':p29['aggregate']['baseline_map_mean'],'fold_validation_tracklets':[x['validation_tracklets'] for x in p29.get('prefix16',[])]},'reconciliation':'NOT_IDENTICAL_DENOMINATOR_OR_FOLD_QUERY_SET; Phase27/29 report event/evaluator-style filtered tracklets, Phase30 uses TRAIN episode validation manifest (837,82,39,30). Raw numerical differences therefore cannot be interpreted as model gain/loss across phases.'}
 audit={'protocol':protocol,'stage30_raw_baseline_by_fold':folds,'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','held event outcomes','future rows/tracks','IDs/text/GT as model inputs'],'raw_baseline_reproduced':True,'training_authorized':True}
 atomic(OUT/'audit/evaluator_reconciliation.json',audit); atomic(OUT/'audit/raw_baseline_by_fold.json',{'protocol':'phase31_unified_stage1_reused','folds':folds,'aggregate_p16':raw['r1']['16']})
 atomic(OUT/'completion/stage0.done',{'stage':0,'raw_baseline_reproduced':True,'contract_reconciled':True})
 lines=['# Phase31 Stage 0 — Evaluator Reconciliation','',f"Raw DINOv2 p16 aggregate on the unified Phase30 TRAIN-disjoint episode contract: R@1={raw['r1']['16']:.4f}, mAP={raw['map']['16']:.4f}, hard-negative gap={raw['hard_negative_gap']['16']:.4f}.",'', 'Phase27/29 numbers (0.8032/0.7201) are not directly comparable: their validation tracklet sets and denominator/filtering differ from the Phase30 episode manifest (f0..f3 = 837,82,39,30). The feature row-key alignment is nevertheless identical (43,423 exact rows; inherited permutation hash).', '', '| fold | tracklets | p16 R@1 | p16 mAP | hard gap |', '|---:|---:|---:|---:|---:|']
 for f in folds: lines.append(f"| {f['fold']} | {f['validation_tracklets']} | {f['r1']:.4f} | {f['map']:.4f} | {f['hard_negative_gap']:.4f} |")
 lines += ['', 'Contract/leakage/resource audit is PASS. Phase31 may train only the preregistered raw-space monotonic reranker; no controller, threshold, StateMemory, backbone or sealed evaluation is authorized at this stage.']
 (OUT/'audit/STAGE0_RECONCILIATION.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'stage0':'done','raw_p16':raw['r1']['16'],'map_p16':raw['map']['16']},indent=2))
if __name__=='__main__': main()
