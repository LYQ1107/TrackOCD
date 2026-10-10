#!/usr/bin/env python3
"""Posthoc descriptive input/prototype coverage; no inference or tuning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file


def length_profile(lengths):
    if not lengths:raise ValueError('Nonempty observed prefixes required')
    return {'tracks':len(lengths),'observations':sum(lengths),'mean_available_observations':sum(lengths)/len(lengths),
        'singletons':sum(n==1 for n in lengths),'at_least16':sum(n>=16 for n in lengths),
        'total_observations_at_cap':{str(p):sum(min(p,n) for n in lengths) for p in (1,2,4,8,16)}}


def main(commit):
    import numpy as np
    import pyarrow.parquet as pq
    started=time.monotonic();out=ROOT/'outputs/trackocd_core';base=out/'core_training/train_first_v1'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=commit or subprocess.check_output(['git','show',f'{commit}:scripts/trackocd_core/report_input_shift.py'])!=Path(__file__).read_bytes():raise ValueError('Exact registered descriptive source required')
    # No GT opened before the predefined complete all-ID prediction seal.
    matrix=base/'limited_masa_evaluation';plan=json.loads((matrix/'plan.json').read_text())
    row=next(r for r in plan['logical_cases'] if (r['method'],r['order'],r['prefix'])==('B0_frame_snapshot_vote','main',1))
    directory=matrix/'cases'/row['execution_id'];seal=json.loads((directory/'sealed_complete.json').read_text())
    if seal['status']!='SEALED_COMPLETE_ALL_PREDICTED_IDS' or seal['physical_tracks']!=304561 or sha256_file(directory/seal['ledger']['filename'])!=seal['ledger']['sha256']:raise ValueError('Actually sealed full predictions required before GT')
    freeze=json.loads((base/'model_freeze.json').read_text())
    for name,sha in freeze['protected_sha256'].items():
        if sha256_file(ROOT/name)!=sha:raise ValueError('No model/protocol amendment allowed')
    def guard():
        m={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}
        rss=next(int(s.split()[1])*1024 for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:'))
        reserve=0
        for filename in ('inference_supervisor_progress.json','evaluator_progress.json'):
            p=matrix/filename
            if p.exists():reserve+=json.loads(p.read_text())['remaining_RAM_reservation_bytes']
        if m['MemAvailable']-reserve-max(0,256*2**20-rss)<m['MemTotal']*.25 or rss>256*2**20:raise RuntimeError('OneCPU256MiB actualRSS/jointsystem25% reserve; defer without foreign interference')
        return rss
    guard()
    train=out/'features/train_first_v1';tm=json.loads((train/'manifest.json').read_text());labels_path=train/'train_labels.parquet'
    for filename in ('train_labels.parquet','index.parquet'):
        if sha256_file(train/filename)!=tm['payloads'][filename]['sha256']:raise ValueError('Actual sealed Train payload identity differs')
    labels=pq.read_table(labels_path,columns=['partition','category_id'],use_threads=False).to_pylist()
    proto={r['category_id'] for r in labels if r['partition']=='prototype'}
    fit=set(json.loads((out/'TRAIN_FIRST_REPRESENTATION_RESULT.json').read_text())['fit_classes'])
    assert len(proto)==48 and len(fit)==15 and fit<=proto
    train_lengths=pq.read_table(train/'index.parquet',columns=['observation_count'],use_threads=False)['observation_count'].to_pylist()
    support=json.loads((out/'audit/physical_cross_video_support.json').read_text());gt_path=out/'audit/physical_cross_video_support_private/evaluator_only_geometry_join.json'
    if sha256_file(gt_path)!=support['private_join']['sha256']:raise ValueError('Existing posthoc geometry join changed')
    gt=json.loads(gt_path.read_text())['frontends']['MASA_NATIVE'];matches={};known_table=Counter();gt_counts=Counter()
    for target in gt:
        role=target['role'].lower();gt_counts[role]+=1;observed=target['reliable_predicted_local_id'] is not None
        if role=='known':known_table[('fit15' if target['category_id'] in fit else 'prototype_only33' if target['category_id'] in proto else 'no_prototype30',observed)]+=1
        if observed:
            key=(target['video_id'],int(target['reliable_predicted_local_id']))
            if key in matches:raise ValueError('One-to-one existing physical GT join required')
            matches[key]=role
    assert (gt_counts['known'],gt_counts['novel'])==(4413,819)
    feature_path=out/'features/masa_limited_prefix_v1/full_manifest.json';fm=json.loads(feature_path.read_text());groups={k:[] for k in ('known','novel','other_matched','unmatched_unknown')}
    matched_found=0;max_rss=guard()
    for record in fm['records']:
        p=feature_path.parent/'shards'/record['npz_filename']
        if sha256_file(p)!=record['npz_sha256']:raise ValueError('Existing common feature bytes differ')
        with np.load(p,allow_pickle=False) as a:
            ids=a['track_id'];lengths=np.diff(a['offsets'])
            for identity,n in zip(ids,lengths):
                role=matches.get((record['video_id'],int(identity)));group=role if role in ('known','novel') else 'unmatched_unknown' if role is None else 'other_matched'
                groups[group].append(int(n));matched_found+=role is not None
        max_rss=max(max_rss,guard())
    assert matched_found==len(matches) and sum(map(len,groups.values()))==304561
    assert (len(groups['known']),len(groups['novel']))==(1400,189)
    result={'status':'COMPLETE_POSTHOC_DESCRIPTIVE_INPUT_SHIFT_NOT_CAUSAL_INTERVENTION','preregistration_commit':commit,
        'source_sha256':sha256_file(Path(__file__).resolve()),'sealed_reference_sha256':seal['ledger']['sha256'],'existing_GT_join_sha256':sha256_file(gt_path),
        'Train_feature_manifest_sha256':sha256_file(train/'manifest.json'),'Native_feature_manifest_sha256':sha256_file(feature_path),'model_freeze_sha256':sha256_file(base/'model_freeze.json'),
        'Train_selected':length_profile(train_lengths),'Native_all':length_profile([n for values in groups.values() for n in values]),
        'Native_by_evaluator_reliable_match':{k:length_profile(v) if v else {'tracks':0} for k,v in groups.items()},
        'Known_GT_prototype_and_physical_coverage':[{'prototype_group':g,'reliable_physical_match':observed,'GT_tracks':known_table[g,observed]} for g in ('fit15','prototype_only33','no_prototype30') for observed in (True,False)],
        'GT_counts_including_any_other_roles':dict(gt_counts),'missing_prototypes_not_filled_from_Val':True,
        'unmatched_not_assumed_background_or_proven_pollution':True,'all_short_tracks_kept':True,
        'posthoc_description_after_first_Val_metrics_not_preregistered_causal_hypothesis':True,
        'models_or_operating_points_modified':False,'GT_to_live_inference':False,'Val_training_or_tuning':False,'TAO_Test_access':False,
        'resources':{'CPU_workers':1,'GPU_used':False,'planned_actual_RSS_ceiling_bytes':256*2**20,'max_observed_RSS_bytes':max_rss,
            'kernel_ru_maxrss_bytes_may_include_preexec_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.monotonic()-started}}
    atomic_json(out/'INPUT_SHIFT_DIAGNOSTIC_RESULT.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);a=p.parse_args();main(a.preregistration_commit)
