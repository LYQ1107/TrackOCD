#!/usr/bin/env python3
"""First4 metadata videos, all-ID frozen feature->baseline->policy->evaluator.

GT geometry/roles are read only after sealed predictions. This is integration,
not a full988-video result or a model/prototype/threshold-selection trial.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preregistration-commit',required=True);args=parser.parse_args()
    source='scripts/trackocd_core/integrate_limited_masa.py'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=args.preregistration_commit or subprocess.check_output(['git','show',f'{args.preregistration_commit}:{source}'])!=(ROOT/source).read_bytes():raise ValueError('Exact remote semantic integration registration required')
    from scripts.trackocd_core.extract_limited_masa_features import inputs
    cfg,common,metadata=inputs();out=ROOT/cfg['output_directory'];output=out/'integration_result.json'
    if output.exists():raise ValueError('Preserve actual integration; no favorable subset replacement')
    manifest_path=out/'integration_manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['videos']!=4 or manifest['config_sha256']!=sha256_file(ROOT/'configs/trackocd_core/limited_masa_features.json'):raise ValueError('Four fixed metadata video features must be complete')
    import numpy as np
    import torch
    from src.trackocd_core.limited_masa_features import completed_feature,LimitedMasaVideoCache
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,evidence_bank,prototypes
    from src.trackocd_core.train_first_replay import replay
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_core.evaluation import Target,TrackKey,join_evaluation,evaluate_standard,evaluate_persistent
    torch.set_num_threads(1);started=time.monotonic()
    train,labels,episode_known=load_train(ROOT);known_ids=json.loads((ROOT/'configs/trackocd_core/roles.json').read_text())['known_ids']
    prototype_ids=sorted({r['category_id'] for r in labels if r['partition']=='prototype'})
    if len(prototype_ids)!=48 or len(known_ids)!=78 or set(prototype_ids)-set(known_ids):raise ValueError('Only48legal Train prototypes,full78Known IDs retained')
    freeze_path=ROOT/cfg['model_freeze'];freeze=json.loads(freeze_path.read_text())
    policy=json.loads((ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    caches=[];routes=[];order=[v['video_id'] for v in metadata['videos'][:4]]
    for record in manifest['records']:
        done=completed_feature(out/'shards',record['video_id'],manifest['config_sha256'],record['input_npz_sha256'])
        cache=LimitedMasaVideoCache(out/'shards'/done['npz_filename'],record['video_id']);caches.append(cache);routes.extend(cache.rows)
    if [c.video_id for c in caches]!=order or len(routes)!=manifest['tracks']:raise ValueError('Every fixed physical ID required')
    # Eight methods,one registered representative seed,all5caps; no class-based selection.
    variants=[('B0_frame_snapshot_vote','A0_RAW',None),('B1_track_nearest','A0_RAW',None),('B2_track_dpmeans','A0_RAW',None),
        ('A1_SELECTED','A1_SELECTED',None),('A2_EVIDENCE','A2_EVIDENCE',None),
        ('D1_SIMPLE_MLP','A1_SELECTED','D1_SIMPLE_MLP'),('D2_RISK_AWARE','A1_SELECTED','D2_RISK_AWARE'),('FULL','A2_EVIDENCE','D2_RISK_AWARE')]
    cases=[];prediction_records=[];source_gt_sha=None;physical_reference=None
    physical_reference_path=ROOT/'outputs/trackocd_core/audit/masa_full_val_physical_result.json'
    for name,representation,policy_name in variants:
        model,lineage=frozen_model(ROOT,representation,1027);proto=prototypes(train,labels,prototype_ids,model)
        banks={cap:{} for cap in (1,2,4,8,16)}
        for cache in caches:
            bank=evidence_bank(cache,cache.rows,model)
            for cap,values in bank.items():banks[cap].update(values)
        backend=name if representation=='A0_RAW' else 'B1_track_nearest'
        calibration=next(c for c in freeze['representation_thresholds'] if c['representation']==representation and c['seed']==(None if representation=='A0_RAW' else 1027) and c['backend']==backend)
        thresholds=calibration['thresholds'];decision=None;wait_bias=0.;fit=None
        if policy_name:
            fit=next(f for f in policy['fits'] if (f['representation'],f['seed'],f['model'])==(representation,1027,policy_name))
            checkpoint=fit['selected_checkpoint']['checkpoint'];path=ROOT/checkpoint['path']
            if sha256_file(path)!=checkpoint['sha256']:raise ValueError('Frozen policy differs')
            decision=DecisionMLP().eval().requires_grad_(False);decision.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state_dict'],strict=True)
            wait_bias=fit['operating_points']['1.0']['wait_bias']
        for cap in (1,2,4,8,16):
            sealed,runtime=replay(routes,order,cap,proto,thresholds,banks[cap],name=backend,decision_model=decision,wait_bias=wait_bias,known_ids=known_ids)
            # The live inference above receives NO Val target/category/geometry fields.
            # Evaluator-only source first opened after irrevocable sealing.
            gt_path=ROOT/'outputs/trackocd_core/audit/physical_cross_video_support_private/evaluator_only_geometry_join.json'
            support=json.loads((ROOT/'outputs/trackocd_core/audit/physical_cross_video_support.json').read_text())
            if sha256_file(gt_path)!=support['private_join']['sha256'] or sha256_file(physical_reference_path)!=support['full_physical_receipt_sha256']:
                raise ValueError('Completed evaluator-only geometry/physical reference identity differs')
            gt=json.loads(gt_path.read_text())['frontends']['MASA_NATIVE'];source_gt_sha=sha256_file(gt_path)
            selected=[g for g in gt if g['video_id'] in order]
            targets=[Target(TrackKey(g['video_id'],str(g['gt_local_id'])),g['category_id'],g['role']) for g in selected]
            matches={TrackKey(r['video_id'],str(r['physical_track_id'])):None for r in routes}
            for g in selected:
                if g['reliable_predicted_local_id'] is not None:
                    key=TrackKey(g['video_id'],str(g['reliable_predicted_local_id']))
                    if key not in matches:raise ValueError('Reliable prediction was lost from all-ID features')
                    matches[key]=TrackKey(g['video_id'],str(g['gt_local_id']))
            join=join_evaluation(sealed,targets,matches);standard=evaluate_standard(join);standard.pop('global_anonymous_hungarian_mapping_evaluator_only')
            persistent=evaluate_persistent(join)
            if physical_reference is None:
                physical_reference=json.loads(physical_reference_path.read_text())['results']['MASA_NATIVE']['canonical_tracking']
            if len(sealed.events)!=len(routes):raise AssertionError('All predicted identities,not matched-only subset')
            cases.append({'method':name,'seed':None if representation=='A0_RAW' else 1027,'prefix':cap,'standard':standard,'persistent':persistent,
                'runtime':runtime,'unmatched_predicted_IDs_retained':sum(v is None for v in matches.values()),'model_lineage':lineage,
                'same_frozen_full_Val_physical_reference_not_subset_score':physical_reference})
            prediction_records.append({'method':name,'prefix':cap,'events':[{'sequence':e.sequence,'video_id':e.physical_key.video_id,'physical_id':e.physical_key.local_track_id,
                'observed':e.observed_prefix,'kind':e.kind,'token':e.token,'known_id':e.known_category_id} for e in sealed.events]})
        del banks,model,decision
    private=out/'integration_sealed_predictions.json';atomic_json(private,prediction_records)
    result={'status':'PASS_FROZEN_FEATURE_BASELINE_POLICY_EVALUATOR_INTEGRATION','scope':'ONLY4_FIXED_METADATA_VIDEOS_NOT_FULL_VAL_SCIENTIFIC_RESULT',
        'preregistration_commit':args.preregistration_commit,'source_sha256':sha256_file(ROOT/source),'feature_manifest_sha256':sha256_file(manifest_path),
        'model_freeze_sha256':sha256_file(freeze_path),'video_order':order,'physical_tracks':len(routes),'observations':manifest['observations'],'cases':cases,
        'prototype_coverage':{'available':48,'inherited_known':78,'missing':30,'Val_GT_supplement':False},
        'evaluator_only_geometry_join_sha256':source_gt_sha,'all_IDs_including_unmatched_present':True,'GT_in_policy_state':False,
        'frozen_full_Val_physical_reference_sha256':sha256_file(physical_reference_path),
        'final_expected_fixed_full_GT':{'known':4413,'novel':819,'reuse_opportunities':[527,529,547,531]},
        'subset_denominators_not_full_Val_claim':True,'scientific_PASS_claimed':False,'Val_tuning':False,'TAO_Test_access':False,
        'new_detector_or_tracker_inference':False,'new_optimization':False,
        'private_sealed_predictions':{'bytes':private.stat().st_size,'sha256':sha256_file(private)},
        'resources':{'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'cpu_workers':1,'GPU_used':False}}
    atomic_json(output,result);atomic_json(ROOT/'outputs/trackocd_core/LIMITED_MASA_INTEGRATION_RESULT.json',result)
    for cache in caches:cache.close()
    print(json.dumps({'status':result['status'],'cases':len(cases),'physical_tracks':len(routes),'resources':result['resources']}),flush=True)


if __name__=='__main__':main()
