#!/usr/bin/env python3
"""Posthoc full-GT metrics of immutable all-ID semantic ledgers only.

May consume completed cases while other frozen inference runs. No GT or
metrics ever sent to inference workers; all models/points already frozen.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,atomic_write_text,sha256_file
SOURCES=('scripts/trackocd_core/evaluate_limited_masa.py','src/trackocd_core/posthoc_diagnostics.py',
    'src/trackocd_core/sealed_parquet.py','src/trackocd_core/scientific_statistics.py','src/trackocd_core/limited_masa_pairs.py','configs/trackocd_core/limited_masa_evaluation.json')


def worker(job_id,assignment_path):
    import numpy as np
    from scripts.trackocd_core.run_limited_masa_inference import inputs,completed_case
    from src.trackocd_core.sealed_parquet import read_sealed
    from src.trackocd_core.evaluation import Target,TrackKey,join_evaluation
    from src.trackocd_core.posthoc_diagnostics import evaluate_with_flags
    cfg,fc,manifest,identity=inputs();out=ROOT/cfg['output_directory'];assignment=json.loads(Path(assignment_path).read_text())
    if assignment['source_sha256']!={n:sha256_file(ROOT/n) for n in SOURCES}:raise ValueError('Registered evaluator source changed')
    plan=json.loads((out/'plan.json').read_text());job=plan['jobs'][job_id];d=completed_case(out,job,identity)
    if d is None:raise ValueError('GT evaluator may only read actually sealed predictions')
    directory=out/'cases'/job_id;destination=directory/'case_metrics.json'
    if destination.exists():raise ValueError('Preserve actual metrics; no favorable overwrite')
    known=json.loads((ROOT/'configs/trackocd_core/roles.json').read_text())['known_ids'];started=time.monotonic()
    sealed=read_sealed(directory/d['ledger']['filename'],plan['orders'][job['order']],known)
    if len(sealed.events)!=304561:raise ValueError('All physical IDs must reach posthoc evaluator')
    # Existing geometry/roles first opened ONLY AFTER actual immutable sealing.
    support=json.loads((ROOT/'outputs/trackocd_core/audit/physical_cross_video_support.json').read_text())
    gt_path=ROOT/'outputs/trackocd_core/audit/physical_cross_video_support_private/evaluator_only_geometry_join.json'
    physical_path=ROOT/'outputs/trackocd_core/audit/masa_full_val_physical_result.json'
    if sha256_file(gt_path)!=support['private_join']['sha256'] or sha256_file(physical_path)!=support['full_physical_receipt_sha256']:raise ValueError('Completed evaluator-only GT/physical identity differs')
    gt=json.loads(gt_path.read_text())['frontends']['MASA_NATIVE'];targets=[Target(TrackKey(g['video_id'],str(g['gt_local_id'])),g['category_id'],g['role']) for g in gt]
    matches={e.physical_key:None for e in sealed.events}
    for g in gt:
        if g['reliable_predicted_local_id'] is not None:
            key=TrackKey(g['video_id'],str(g['reliable_predicted_local_id']))
            if key not in matches:raise ValueError('Full predicted features/ledger lost a reliablephysicalID')
            matches[key]=TrackKey(g['video_id'],str(g['gt_local_id']))
    metrics,flags=evaluate_with_flags(join_evaluation(sealed,targets,matches))
    if (metrics['standard']['old_denominator'],metrics['standard']['new_denominator'])!=(4413,819):raise AssertionError('Never shrink fullGT denominators to frontend matches')
    if metrics['persistent']['commit_ct_denominator']!=cfg['fixed_full_GT']['reuse_opportunities'][job['order']]:raise AssertionError('Fixed fullGT reuse opportunity universe changed')
    if (metrics['errors']['matched_only_diagnostic']['known_denominator'],metrics['errors']['matched_only_diagnostic']['novel_denominator'])!=(1400,189):raise AssertionError('Canonical Native physical coverage changed')
    flag_path=directory/f"case_flags_{assignment['attempt']}.npz"
    with flag_path.open('xb') as f:np.savez_compressed(f,**flags)
    result={'status':'COMPLETE_FULL_GT_POSTHOC_SEMANTIC_METRICS','execution_id':job_id,'job':job,'input_identity':identity,
        'sealed_ledger_sha256':d['ledger']['sha256'],'runtime':d['runtime'],**metrics,
        'private_flags':{'filename':flag_path.name,'bytes':flag_path.stat().st_size,'sha256':sha256_file(flag_path)},
        'evaluator_source_sha256':assignment['source_sha256'],'evaluator_preregistration_commit':assignment['preregistration_commit'],
        'GT_join_sha256':sha256_file(gt_path),'physical_reference_sha256':sha256_file(physical_path),
        'unmatched_predicted_IDs_retained':sum(v is None for v in matches.values()),
        'GT_in_inference':False,'Val_tuning':False,'TAO_Test_access':False,
        'resources':{'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_used':False}}
    if result['resources']['peak_RSS_bytes']>4*2**30:raise RuntimeError('Evaluator4GiB RSS guard')
    atomic_json(destination,result)
    print(json.dumps({'case':job_id,'status':result['status'],'resources':result['resources']}),flush=True)


def metrics_record(out,job_id):
    p=out/'cases'/job_id/'case_metrics.json'
    if not p.exists():return None
    d=json.loads(p.read_text())
    if d['status']!='COMPLETE_FULL_GT_POSTHOC_SEMANTIC_METRICS' or d['execution_id']!=job_id:raise ValueError('Unexpected existing metric artifact')
    seal=json.loads((p.parent/'sealed_complete.json').read_text())
    if d['input_identity']!=seal['input_identity'] or d['job']!=seal['job'] or d['sealed_ledger_sha256']!=seal['ledger']['sha256']:raise ValueError('Metrics/seal identity differs')
    if d['evaluator_source_sha256']!={n:sha256_file(ROOT/n) for n in SOURCES}:raise ValueError('Existing metrics bound to different evaluator source')
    return d


def csv_write(path,rows):
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);atomic_write_text(path,s.getvalue())


def collect(cfg,out,plan,commit,started):
    import numpy as np
    from src.trackocd_core.posthoc_diagnostics import paired_corrections
    from src.trackocd_core.scientific_statistics import paired_video_bootstrap
    from src.trackocd_core.limited_masa_pairs import registered_pairs
    unique={key:metrics_record(out,key) for key in plan['jobs']}
    if any(d is None for d in unique.values()):raise ValueError('Every registered unique execution required')
    cases=[];aggregate=[];pairs=[]
    for r in plan['logical_cases']:
        d=unique[r['execution_id']];cases.append({**r,'standard':d['standard'],'persistent':d['persistent'],
            'errors':{k:v for k,v in d['errors'].items() if k!='per_video_conditional_fixed_mapping_counts'},'runtime':d['runtime'],
            'unmatched_predicted_IDs_retained':d['unmatched_predicted_IDs_retained']})
    fields={'old_acc':'standard','new_acc':'standard','h_score':'standard','all_acc':'standard','correct_commit_ct':'persistent','false_merge_rate':'persistent',
        'false_split_new_rate':'persistent','wrong_known_assignment_rate':'persistent','wait_unresolved_rate':'persistent','effective_commit_coverage':'persistent',
        'all_novel_wrong_known_rate':'errors','memory_contamination_write_events':'errors','contaminated_states':'errors'}
    for method,target,cap in sorted({(r['method'],r['coverage_target'],r['prefix']) for r in cases},key=lambda a:(a[0],str(a[1]),a[2])):
        rows=[r for r in cases if (r['method'],r['coverage_target'],r['prefix'])==(method,target,cap)];seeds=sorted({r['seed'] for r in rows},key=str)
        record={'scope':'LIMITED_MASA_FULL_GT_ANNOTATED_CADENCE','method':method,'coverage_target':target,'prefix':cap,'training_seeds':len(seeds),'video_orders':4}
        for metric,section in fields.items():
            seed_means=[float(np.mean([r[section][metric] for r in rows if r['seed']==seed])) for seed in seeds]
            record[metric+'_mean']=float(np.mean(seed_means));record[metric+'_seed_std_ddof0']=float(np.std(seed_means))
            record[metric+'_order_and_seed_std_ddof0']=float(np.std([r[section][metric] for r in rows]))
        record['predicted_ID_commitment_coverage_mean']=float(np.mean([r['runtime']['commitment_coverage'] for r in rows]));aggregate.append(record)
    def flags(r):
        d=unique[r['execution_id']];path=out/'cases'/r['execution_id']/d['private_flags']['filename']
        if sha256_file(path)!=d['private_flags']['sha256']:raise ValueError('Posthocflags changed')
        with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}
    for left,right,seed,order,cap,target,a,b in registered_pairs(cases,cfg):
        reuse_gap=abs(a['persistent']['effective_commit_coverage']-b['persistent']['effective_commit_coverage']);total_gap=abs(a['runtime']['commitment_coverage']-b['runtime']['commitment_coverage'])
        record={'left':left,'right':right,'seed':seed,'order':order,'prefix':cap,'coverage_target':target,'actual_reuse_coverage_gap':reuse_gap,
            'actual_predicted_ID_commitment_coverage_gap':total_gap,'actual_full_GT_non_wait_coverage_gap':abs(a['standard']['non_wait_coverage']-b['standard']['non_wait_coverage']),
            'comparable_within_tolerance':max(reuse_gap,total_gap)<=cfg['coverage_comparability_tolerance'],
            'h_score_delta':b['standard']['h_score']-a['standard']['h_score'],'CT_delta':b['persistent']['correct_commit_ct']-a['persistent']['correct_commit_ct'],
            'false_merge_delta':b['persistent']['false_merge_rate']-a['persistent']['false_merge_rate'],'wrong_known_all_novel_delta':b['errors']['all_novel_wrong_known_rate']-a['errors']['all_novel_wrong_known_rate']}
        if cap==16:
            record['paired_error_corrections']=paired_corrections(flags(a),flags(b))
            if target==1.:record['conditional_video_bootstrap']=paired_video_bootstrap(unique[a['execution_id']]['errors']['per_video_conditional_fixed_mapping_counts'],unique[b['execution_id']]['errors']['per_video_conditional_fixed_mapping_counts'],cfg['bootstrap']['samples'],cfg['bootstrap']['seed'])
        pairs.append(record)
    physical_path=ROOT/'outputs/trackocd_core/audit/masa_full_val_physical_result.json';physical=json.loads(physical_path.read_text())['results']['MASA_NATIVE']['canonical_tracking']
    result={'status':'COMPLETE_REAL_FULL_LIMITED_MASA_FROZEN_SEMANTIC_EVALUATION','preregistration_commit':commit,
        'input_identity':next(iter(unique.values()))['input_identity'],'full_inference_manifest_sha256':sha256_file(out/'full_inference_manifest.json'),
        'config_sha256':sha256_file(ROOT/'configs/trackocd_core/limited_masa_evaluation.json'),'source_sha256':{n:sha256_file(ROOT/n) for n in SOURCES},
        'unique_sealed_executions':len(unique),'logical_cases_including_explicit_aliases':len(cases),'cases':cases,'aggregate':aggregate,'paired_comparisons':pairs,
        'fixed_full_GT':cfg['fixed_full_GT'],'physical_native_reference':physical,'physical_reference_sha256':sha256_file(physical_path),
        'identical_frozen_physical_scores_for_every_method':True,'physical_metric_gain_from_semantic_postprocessing_claimed':False,
        'prototype_coverage':{'available':48,'inherited_known':78,'missing':30,'ValGTfill':False,'deployment_shift':cfg['prototype_episode_to_deployment_shift']},
        'PHE':cfg['PHE'],'scope':cfg['scope'],'M1_status_preserved':'BLOCKED_FRONTEND_QUALITY','new_optimization':False,'Val_tuning':False,'TAO_Test_access':False,
        'all_short_unmatched_and_unknown_IDs_kept':True,'all_models_and_points_frozen_before_Val_metrics':True,
        'named_coverage_targets_are_Train_development_not_achieved_Val_coverage_claims':True,
        'resource_summary':{'evaluator_supervisor_wall_seconds':time.monotonic()-started,'max_case_evaluator_RSS_bytes':max(d['resources']['peak_RSS_bytes'] for d in unique.values()),
            'all_private_ledgers_bytes':sum(json.loads((out/'cases'/k/'sealed_complete.json').read_text())['ledger']['bytes'] for k in unique)}}
    order_rows=[];error_rows=[]
    for r in cases:
        meta={'scope':'LIMITED_MASA_FULL_GT_ANNOTATED_CADENCE','stage':'T5','method':r['method'],'backend':'B1_track_nearest' if not r['method'].startswith('B') else r['method'],
            'seed':r['seed'],'order':r['order'],'prefix':r['prefix'],'coverage_target':r['coverage_target']}
        order_rows.append({**meta,**{k:r[section][k] for k,section in fields.items() if section!='errors'}})
        error_rows.append({**meta,'wrong_known_all_novel':r['errors']['all_novel_wrong_known_count'],'novel_gt_denominator':819,
            'memory_contamination_writes':r['errors']['memory_contamination_write_events'],'contaminated_states':r['errors']['contaminated_states'],**r['persistent']['outcome_counts']})
    combined={}
    for filename,new_rows in (('PER_ORDER_RESULTS.csv',order_rows),('ERROR_BREAKDOWN.csv',error_rows)):
        path=ROOT/'outputs/trackocd_core'/filename;old=list(csv.DictReader(io.StringIO(path.read_text())))
        if any(r['stage']=='T5' for r in old):raise ValueError('Do not duplicate completed formal stage in canonicalCSV')
        if set(old[0])!=set(new_rows[0]):raise ValueError('CanonicalCSV schema must preserve prior controlledrows')
        combined[filename]=[*old,*new_rows]
    csv_write(ROOT/'outputs/trackocd_core/LIMITED_MASA_BASELINE_AND_POLICY_COMPARISON.csv',aggregate)
    alias_counts={k:sum(r['execution_id']==k for r in plan['logical_cases']) for k in plan['jobs']};seen=set();alias_rows=[]
    for r in plan['logical_cases']:
        k=r['execution_id'];alias_rows.append({**{f:r[f] for f in ('method','seed','order','prefix','coverage_target')},
            'execution_id':k,'logical_alias_count':alias_counts[k],'first_logical_reference_not_independent_repetition':k not in seen});seen.add(k)
    csv_write(ROOT/'outputs/trackocd_core/LIMITED_MASA_CASE_ALIAS_MANIFEST.csv',alias_rows)
    csv_write(ROOT/'outputs/trackocd_core/LIMITED_MASA_PER_ORDER_RESULTS.csv',order_rows);csv_write(ROOT/'outputs/trackocd_core/LIMITED_MASA_ERROR_BREAKDOWN.csv',error_rows)
    for filename,rows in combined.items():csv_write(ROOT/'outputs/trackocd_core'/filename,rows)
    atomic_json(out/'full_evaluation_result.json',result);atomic_json(ROOT/'outputs/trackocd_core/LIMITED_MASA_EVALUATION_RESULT.json',result)
    print(json.dumps({'status':result['status'],'unique_executions':len(unique),'logical_cases':len(cases),'p16_main_point':[r for r in aggregate if r['prefix']==16 and r['coverage_target'] in (None,1.)]}),flush=True)


def supervise(commit):
    from scripts.trackocd_core.run_limited_masa_inference import inputs,ram,remaining_RAM_reservation,counterpart_reservation
    cfg,fc,manifest,identity=inputs();out=ROOT/cfg['output_directory'];started=time.monotonic();attempt=uuid.uuid4().hex
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=commit:raise ValueError('Exact remote evaluator registration required')
    for n in SOURCES:
        if subprocess.check_output(['git','show',f'{commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Evaluator source differs')
    if (out/'full_evaluation_result.json').exists():raise ValueError('Preserve completed full evaluation')
    plan=json.loads((out/'plan.json').read_text());assignment_path=out/f'evaluator_assignment_{attempt}.json'
    atomic_json(assignment_path,{'attempt':attempt,'preregistration_commit':commit,'source_sha256':{n:sha256_file(ROOT/n) for n in SOURCES}})
    pending=set(plan['jobs']);active={};children=[];logs=[];error=None
    try:
        while pending or active:
            m=ram()
            if m['MemAvailable']<m['MemTotal']*.25 or time.monotonic()-started>cfg['stage_wall_seconds']:raise RuntimeError('EvaluatorRAM/wall guard')
            for key,p in list(active.items()):
                code=p.poll()
                if code is not None:
                    if code!=0:raise RuntimeError(f'Owned evaluatorcase failed {key}; preserve actual files')
                    del active[key]
            for key in sorted(pending):
                directory=out/'cases'/key
                if (directory/'case_metrics.json').exists():metrics_record(out,key);pending.remove(key);continue
                own_remaining=remaining_RAM_reservation(active.values(),4*2**30)
                inference_remaining=counterpart_reservation(out,'inference_supervisor_progress.json')
                if len(active)>=4 or m['MemAvailable']-own_remaining-inference_remaining-4*2**30<m['MemTotal']*.25:break
                if not (directory/'sealed_complete.json').exists():continue
                log=(out/f'evaluator_{key}_{attempt}.log').open('x');logs.append(log)
                env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--case',key,'--assignment',str(assignment_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=p;children.append(p);pending.remove(key)
                m=ram()
            atomic_json(out/'evaluator_progress.json',{'complete_cases':sum((out/'cases'/k/'case_metrics.json').exists() for k in plan['jobs']),'pending':len(pending),'active':len(active),
                'remaining_RAM_reservation_bytes':remaining_RAM_reservation(active.values(),4*2**30),'seconds':time.monotonic()-started})
            time.sleep(5)
        inference_path=out/'full_inference_manifest.json'
        while not inference_path.exists():
            if time.monotonic()-started>cfg['stage_wall_seconds']:raise RuntimeError('Wait for actual full inference terminal receipt; no premature final result')
            time.sleep(5)
        inference=json.loads(inference_path.read_text())
        if inference['status']!='COMPLETE_FROZEN_FULL_LIMITED_MASA_SEMANTIC_PREDICTIONS_NOT_YET_METRICS' or inference['identity']!=identity or inference['unique_executions']!=840:raise ValueError('Actual full inference receipt required')
        collect(cfg,out,plan,commit,started)
    except Exception as exc:
        error=repr(exc)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        raise
    finally:
        for log in logs:log.close()
        atomic_json(out/'evaluator_progress.json',{'remaining_RAM_reservation_bytes':0,'active':0,'seconds':time.monotonic()-started,'error':error})
        atomic_json(out/f'evaluator_supervisor_{attempt}.json',{'error':error,'owned_returncodes':[p.returncode for p in children],'foreign_process_interference':False,'wall_seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit');p.add_argument('--case');p.add_argument('--assignment');a=p.parse_args()
    if a.case:worker(a.case,a.assignment)
    elif a.preregistration_commit:supervise(a.preregistration_commit)
    else:p.error('Choose exactremote parent or owned case')
