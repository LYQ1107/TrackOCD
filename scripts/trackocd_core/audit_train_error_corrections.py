#!/usr/bin/env python3
"""Posthoc paired fixes/new errors of unchanged actual Train heldout ledgers.

No fit, threshold changes or rerunning live policies. Audited Standard uses
one global mapping; CT remains absorbing purity without Hungarian repair.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,atomic_write_text,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);a=p.parse_args()
    sources=('scripts/trackocd_core/audit_train_error_corrections.py','src/trackocd_core/posthoc_diagnostics.py')
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=a.preregistration_commit:raise ValueError('Exact remote correction audit registration required')
    for n in sources:
        if subprocess.check_output(['git','show',f'{a.preregistration_commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Audit source changed')
    import numpy as np
    import pyarrow.parquet as pq
    from src.trackocd_core.train_first_experiment import load_train
    from src.trackocd_core.evaluation import DecisionEvent,TrackKey,Target,seal_decisions,join_evaluation
    from src.trackocd_core.posthoc_diagnostics import evaluate_with_flags,paired_corrections
    from scripts.trackocd_core.extract_limited_masa_features import inputs
    inputs();started=time.monotonic();base=ROOT/'outputs/trackocd_core/core_training/train_first_v1';destination=ROOT/'outputs/trackocd_core/TRAIN_FIRST_ERROR_CORRECTION_RESULT.json'
    if destination.exists():raise ValueError('Preserve actual error audit')
    cache,labels,known=load_train(ROOT);by={r['key']:r for r in labels};routes=[r for r in cache.rows if by[r['key']]['partition']=='heldout_selection'];route_by={(r['video_id'],str(r['physical_track_id'])):r for r in routes}
    targets=[Target(TrackKey(r['video_id'],str(r['physical_track_id'])),by[r['key']]['category_id'],'known' if by[r['key']]['simulation_role']=='known' else 'novel') for r in routes]
    if len(targets)!=223:raise ValueError('Full223heldout target universe required')
    flags={};source_records={};case_verifications=[]
    for phase in ('representation','policy'):
        directory=base/f'{phase}_evaluation';receipt=json.loads((directory/'results.json').read_text());path=directory/'sealed_predictions.parquet'
        if sha256_file(path)!=receipt['private_ledger']['sha256']:raise ValueError('Actual retained ledger changed')
        source_records[phase]={'receipt_sha256':sha256_file(directory/'results.json'),'ledger_sha256':sha256_file(path),'rows':receipt['private_ledger']['rows']}
        groups=defaultdict(list)
        group_fields=('representation','seed','backend','order','prefix') if phase=='representation' else ('method','seed','order','prefix')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768):
            for row in batch.to_pylist():groups[tuple(row[k] for k in group_fields)].append(row)
        for key,rows in groups.items():
            rows.sort(key=lambda r:r['sequence']);cap=rows[0]['prefix'];order_name=rows[0]['order'];order=receipt['orders'][order_name]
            events=[DecisionEvent(r['sequence'],TrackKey(r['video_id'],r['physical_track_id']),min(cap,route_by[(r['video_id'],r['physical_track_id'])]['observation_count']),r['kind'],known_category_id=r['known_category_id'],token=r['token']) for r in rows]
            sealed=seal_decisions(events,video_order=order,prefix_cap=cap,known_ids=known);join=join_evaluation(sealed,targets,{t.key:t.key for t in targets});metrics,f=evaluate_with_flags(join)
            expected=next(r for r in receipt['cases'] if tuple(r[k] for k in group_fields)==key and (phase=='representation' or r['coverage_target']==1.))
            if metrics['standard']!=expected['standard'] or metrics['persistent']!=expected['persistent']:raise AssertionError('Posthoc flags must exactly reproduce immutable audited metrics')
            identity=(phase,)+(key if phase=='representation' else (key[0],key[1],'B1_track_nearest',key[2],key[3]))
            flags[identity]=f;case_verifications.append({'phase':phase,'identity':list(key),'exact_metrics_equal':True,'tracks':len(rows)})
        del groups
    comparisons=[]
    for seed in (1027,1028,1029):
        for order in ('main','seed1027','seed1028','seed1029'):
            for cap in (1,2,4,8,16):
                pairs=[(('representation','A0_RAW',None,'B1_track_nearest',order,cap),('representation',r,seed,'B1_track_nearest',order,cap)) for r in ('A1_SELECTED','A2_EVIDENCE','A2_CAPACITY_CONTROL')]
                pairs +=[(('representation','A1_SELECTED',seed,'B1_track_nearest',order,cap),('representation',r,seed,'B1_track_nearest',order,cap)) for r in ('A2_EVIDENCE','A2_CAPACITY_CONTROL')]
                pairs +=[(('policy',left,seed,'B1_track_nearest',order,cap),('policy',right,seed,'B1_track_nearest',order,cap)) for left,right in (('D1_SIMPLE_MLP','D2_RISK_AWARE'),('WITHOUT_RISK','FULL'),('WITHOUT_TEMPORAL','FULL'))]
                for left,right in pairs:comparisons.append({'scope':'TAO_TRAIN_GT_CONTROLLED_HELDOUT','left':left[1],'right':right[1],'seed':seed,'order':order,'prefix':cap,'phase':left[0],'counts':paired_corrections(flags[left],flags[right])})
    csv_rows=[]
    for r in comparisons:
        for metric,counts in r['counts'].items():csv_rows.append({**{k:v for k,v in r.items() if k!='counts'},'metric_and_role':metric,**counts})
    result={'status':'COMPLETE_POSTHOC_PAIRED_FIXES_AND_NEW_ERRORS_NOT_RETRAINING','preregistration_commit':a.preregistration_commit,
        'source_sha256':{n:sha256_file(ROOT/n) for n in sources},'actual_inputs':source_records,'verified_sealed_cases':case_verifications,'comparisons':comparisons,
        'fixed_known_gt':207,'fixed_pseudo_novel_gt':16,'one_global_mapping_only_standard':True,'no_hungarian_CT_repair':True,
        'policy_coverage_scope':'Only1.0point retained ledgers; othercoveragepoints not invented','no_fit_or_tuning':True,'Val_or_Test_access':False,
        'raw_shared_control_not_independent_three_repeats':True,'posthoc_diagnostic_not_architecture_selection':True,
        'resources':{'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_workers':1}}
    atomic_json(destination,result);buffer=io.StringIO();w=csv.DictWriter(buffer,fieldnames=list(csv_rows[0]),lineterminator='\n');w.writeheader();w.writerows(csv_rows)
    atomic_write_text(ROOT/'outputs/trackocd_core/ERROR_CORRECTION_COMPARISON.csv',buffer.getvalue())
    print(json.dumps({'status':result['status'],'verified_cases':len(case_verifications),'paired_comparisons':len(comparisons),'resources':result['resources']}),flush=True)


if __name__=='__main__':main()
