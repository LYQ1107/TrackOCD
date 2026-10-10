#!/usr/bin/env python3
"""Publish the predetermined RawB0/main/p1 fullstream case, not best-case pick."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from src.trackocd_v2.io import atomic_json,sha256_file
    out=ROOT/'outputs/trackocd_core/core_training/train_first_v1/limited_masa_evaluation'
    plan=json.loads((out/'plan.json').read_text())
    logical=next(r for r in plan['logical_cases'] if (r['method'],r['order'],r['prefix'])==('B0_frame_snapshot_vote','main',1))
    p=out/'cases'/logical['execution_id']/'case_metrics.json';d=json.loads(p.read_text());seal=p.parent/'sealed_complete.json';s=json.loads(seal.read_text())
    if d['status']!='COMPLETE_FULL_GT_POSTHOC_SEMANTIC_METRICS' or s['status']!='SEALED_COMPLETE_ALL_PREDICTED_IDS' or s['physical_tracks']!=304561:raise ValueError('Actually completed all-ID seal and posthoc metrics required')
    if sha256_file(p.parent/s['ledger']['filename'])!=d['sealed_ledger_sha256']:raise ValueError('Actual completed prediction bytes differ')
    assert (d['standard']['old_denominator'],d['standard']['new_denominator'],d['persistent']['commit_ct_denominator'])==(4413,819,527)
    result={'status':'PARTIAL_METHOD_MATRIX_TRUE_COMPLETE_FULL_STREAM_CASE','method':'B0_frame_snapshot_vote','seed':None,'order':'main','prefix':1,
        'reported_case_predefined_not_best_outcome_selected':True,'full988videos':True,'all304561physical_IDs':True,'full_matrix_complete':False,
        'execution_id':d['execution_id'],'preregistration_commit':d['evaluator_preregistration_commit'],'input_identity':d['input_identity'],
        'private_metrics_sha256':sha256_file(p),'sealed_predictions_sha256':d['sealed_ledger_sha256'],'source_sha256':sha256_file(Path(__file__).resolve()),
        'standard':d['standard'],'persistent':d['persistent'],'runtime':d['runtime'],
        'errors':{k:v for k,v in d['errors'].items() if k!='per_video_conditional_fixed_mapping_counts'},
        'inference_resources':s['resources'],'evaluation_resources':d['resources'],
        'GT_unresolved_includes_physically_missing_not_only_policy_WAIT':True,'GT_or_metrics_in_inference':False,'Val_tuning':False,'TAO_Test_access':False}
    atomic_json(ROOT/'outputs/trackocd_core/LIMITED_MASA_FIRST_FULLSTREAM_CASE_RESULT.json',result)
    print(json.dumps({'status':result['status'],'Old':d['standard']['old_acc'],'New':d['standard']['new_acc'],'H':d['standard']['h_score'],'CT':d['persistent']['correct_commit_ct']}))


if __name__=='__main__':main()
