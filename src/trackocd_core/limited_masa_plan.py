"""All preregistered methods/orders/prefixes, exact-input alias identification."""
import hashlib
import json
from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders


def make_plan(cfg,freeze,policy,video_ids):
    orders=registered_orders(video_ids);jobs={};logical=[]
    if set(orders)!=set(cfg['orders']):raise ValueError('Registered four orders required')
    def add(name,rep,seed,backend,policy_name,options,coverage,point):
        calibration=next(c for c in freeze['representation_thresholds'] if c['representation']==rep and c['seed']==(None if rep=='A0_RAW' else seed) and c['backend']==backend)
        for order in cfg['orders']:
            for cap in cfg['prefixes']:
                identity={'representation':rep,'seed':seed,'backend':backend,'policy':policy_name,'options':options,
                    'thresholds':calibration['thresholds'],'wait_bias':point['wait_bias'] if point else 0.,'order':order,'prefix':cap}
                key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
                jobs.setdefault(key,{'id':key,**identity})
                logical.append({'method':name,'seed':seed,'representation':rep,'order':order,'prefix':cap,
                    'coverage_target':coverage,'development_operating_point':point,'execution_id':key})
    for backend in ('B0_frame_snapshot_vote','B1_track_nearest','B2_track_dpmeans'):add(backend,'A0_RAW',None,backend,None,{},None,None)
    for rep in ('A1_SELECTED','A2_EVIDENCE','A2_CAPACITY_CONTROL'):
        for seed in cfg['seeds']:add(rep,rep,seed,'B1_track_nearest',None,{},None,None)
    variants=[('D1_SIMPLE_MLP','A1_SELECTED','D1_SIMPLE_MLP',{}),('D2_RISK_AWARE','A1_SELECTED','D2_RISK_AWARE',{}),
        ('WITHOUT_ADAPTER','A0_RAW','D2_RISK_AWARE',{}),('WITHOUT_TEMPORAL','A1_SELECTED','D2_RISK_AWARE',{}),
        ('WITHOUT_RELIABILITY','A2_EVIDENCE','D2_RISK_AWARE',{'no_reliability':True}),('WITHOUT_WAIT','A2_EVIDENCE','D2_RISK_AWARE',{'allow_wait':False}),
        ('WITHOUT_MEMORY','A2_EVIDENCE','D2_RISK_AWARE',{'no_memory':True}),('WITHOUT_RISK','A2_EVIDENCE','D1_SIMPLE_MLP',{}),
        ('RESET_PER_VIDEO','A2_EVIDENCE','D2_RISK_AWARE',{'reset_per_video':True}),('FULL','A2_EVIDENCE','D2_RISK_AWARE',{})]
    for name,rep,p,options in variants:
        for seed in cfg['seeds']:
            fit=next(f for f in policy['fits'] if (f['representation'],f['seed'],f['model'])==(rep,seed,p))
            for coverage in cfg['coverage_targets']:
                point=fit['operating_points'][str(coverage)];add(name,rep,seed,'B1_track_nearest',p,options,coverage,point)
    groups={}
    for job in jobs.values():
        # Raw has no projected RAMbank to share;split its three baseline
        # streams for bounded independent worker scheduling,not new trials.
        backend=job['backend'] if job['representation']=='A0_RAW' and job['policy'] is None else 'shared'
        group=f"{job['representation']}_{job['seed']}_{backend}_{bool(job['options'].get('no_reliability'))}"
        groups.setdefault(group,[]).append(job['id'])
    if (len(logical),len(jobs))!=(cfg['expected_logical_cases'],cfg['expected_unique_executions_under_current_frozen_biases']):raise ValueError('Frozen exact-input count changed; not outcome-selected pruning')
    return {'orders':orders,'jobs':jobs,'logical_cases':logical,'groups':groups,'deduplicated_only_exact_inference_identities':True,
        'raw_baseline_no_training_seed_repetition':True,'labels_or_Val_metrics_used_to_choose_jobs':False}
