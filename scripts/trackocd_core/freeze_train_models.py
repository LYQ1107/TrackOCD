#!/usr/bin/env python3
"""Freeze actual completed Train artifacts, not research PASS or final report."""
from __future__ import annotations
import csv
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,atomic_write_text,sha256_file


def main():
    base=ROOT/'outputs/trackocd_core/core_training/train_first_v1';output=base/'model_freeze.json'
    if output.exists():raise ValueError('Frozen models/operating points cannot be overwritten')
    paths={'representation':base/'representation/training_receipt.json','evidence':base/'evidence/training_receipt.json',
        'policy':base/'policy/training_receipt.json','representation_heldout':base/'representation_evaluation/results.json',
        'policy_heldout':base/'policy_evaluation/results.json',
        'role_audit':ROOT/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS.json'}
    results={k:json.loads(v.read_text()) for k,v in paths.items()}
    if len(results['representation']['fits'])!=9 or len(results['evidence']['fits'])!=6 or len(results['policy']['fits'])!=18:raise ValueError('All actual registered fits required')
    if len(results['representation_heldout']['cases'])!=420 or len(results['policy_heldout']['cases'])!=1800:raise ValueError('All actual heldout cases required')
    protected={str(p.relative_to(ROOT)):sha256_file(p) for p in paths.values()}
    configs=('representation_training','evidence_training','persistent_policy_training','gt_main_evaluation','training_split','common_features','SCOPE_AMENDMENT_TRAIN_FIRST')
    for name in configs:
        p=ROOT/f'configs/trackocd_core/{name}.json';protected[str(p.relative_to(ROOT))]=sha256_file(p)
    for role in ('representation','evidence','policy'):
        for source,sha in results[role]['source_sha256'].items():
            if sha256_file(ROOT/source)!=sha:raise ValueError('Completed training source differs')
            protected[source]=sha
        for fit in results[role]['fits']:
            cp=fit['selected_checkpoint']['checkpoint'];path=ROOT/cp['path']
            if path.stat().st_size!=cp['bytes'] or sha256_file(path)!=cp['sha256']:raise ValueError('Selected checkpoint changed')
            protected[cp['path']]=cp['sha256']
    result={'schema_version':'trackocd.core.train-main-frozen.v1','status':'FROZEN_TRAIN_ONLY_MODELS_AND_OPERATING_POINTS',
        'scientific_PASS_claimed':False,'final_goal_complete':False,'protected_sha256':protected,
        'selection':'Previously registered Train development checkpoint/family/operating points; no Val tuning or heldout redesign',
        'representation_family':results['representation']['shared_geometry_family'],
        'policy_operating_points':[{'representation':f['representation'],'policy':f['model'],'seed':f['seed'],'points':f['operating_points']} for f in results['policy']['fits']],
        'representation_thresholds':[{'representation':c['representation'],'seed':c['seed'],'backend':c['backend'],'thresholds':c['selected']['thresholds']} for c in results['representation_heldout']['calibrations']],
        'val_prototypes':'All48available inheritedKnown classes from151legal Train prototype tracks,same prefix16 mean procedure;30missing Known IDs stay in GT denominator; no Val supplementation',
        'prototype_count_shift':'Train episodes15 to final48: explicit distribution shift,no retuning on Val',
        'native_MASA_status':'LIMITED_COVERAGE_PROVENANCE_INCOMPLETE_FRONTEND','cadence':'ANNOTATED-CADENCE CAUSAL REPLAY',
        'historical_M1_status':'BLOCKED_FRONTEND_QUALITY','TAO_Test_access':False,'Val_Novel_training_or_tuning':False}
    atomic_json(output,result)
    atomic_json(ROOT/'outputs/trackocd_core/TRAIN_FIRST_MODEL_FREEZE_RESULT.json',{**result,'private_freeze_sha256':sha256_file(output)})
    # Required per-order table combines actual completed controlled stages only.
    metrics={'old_acc':'standard','new_acc':'standard','h_score':'standard','all_acc':'standard','correct_commit_ct':'persistent',
        'false_merge_rate':'persistent','false_split_new_rate':'persistent','wrong_known_assignment_rate':'persistent','wait_unresolved_rate':'persistent','effective_commit_coverage':'persistent'}
    rows=[];errors=[]
    for kind in ('representation_heldout','policy_heldout'):
        for case in results[kind]['cases']:
            identity={'scope':'TAO_TRAIN_GT_CONTROLLED_HELDOUT','stage':kind,'method':case.get('method',case.get('representation')),
                'backend':case.get('backend','frozen_MLP'),'seed':case['seed'],'order':case['order'],'prefix':case['prefix'],
                'coverage_target':case.get('coverage_target',1.)}
            rows.append({**identity,**{m:case[s][m] for m,s in metrics.items()}})
            errors.append({**identity,'wrong_known_all_novel':case['errors']['all_novel_wrong_known_count'],
                'novel_gt_denominator':case['errors']['all_novel_wrong_known_denominator'],
                'memory_contamination_writes':case['errors']['memory_contamination_write_events'],'contaminated_states':case['errors']['contaminated_states'],
                **case['persistent']['outcome_counts']})
    for name,data in (('PER_ORDER_RESULTS.csv',rows),('ERROR_BREAKDOWN.csv',errors)):
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
        atomic_write_text(ROOT/'outputs/trackocd_core'/name,buffer.getvalue())
    print(json.dumps({'status':result['status'],'protected_assets':len(protected),'per_order_rows':len(rows),'final_goal_complete':False}),flush=True)


if __name__=='__main__':main()
