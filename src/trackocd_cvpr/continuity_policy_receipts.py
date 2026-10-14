"""Public complete one-head/four-arm policy qualification, never partial gain."""
import json
import math
from src.trackocd_v2.io import sha256_file
from .continuity_policy_protocol import EXPECTED,PREFIX
from .raw_membership_formal_cohort import verified_cohort


def trial_sources(model):
    if model not in EXPECTED['models']:raise ValueError('Exact five fixed heads only')
    out=PREFIX+'_'+model
    return (out+'/launch.json',out+'/result.json',out+'_supervisor_attempt_1.json')


def same_macro_aggregate(a,b,max_ulps=1):
    # External class insertion order can reorder three rounded class means.
    # The ONE solver's legacy/primary prefix-16 insertion order differs by two
    # ulps; permit that exact bounded case only at its call site. Model
    # events/scores/history/state/source and physical counts stay exact.
    return (math.isfinite(a) and math.isfinite(b) and isinstance(max_ulps,int)
            and max_ulps>=0 and abs(a-b)<=max_ulps*math.ulp(max(abs(a),abs(b))))


def verified_trial(root,model):
    from scripts.trackocd_cvpr.pilot_continuity_reuse_policy import SOURCES
    paths=tuple(root/p for p in trial_sources(model))
    if any(not p.is_file() or p.is_symlink() for p in paths):raise ValueError('Complete original policy triplet required')
    l,r,t=(json.loads(p.read_text()) for p in paths)
    if (r['status']!='MEASURED_COMPLETE_ONE_HEAD_FOUR_ARM_POLICY_CLOSED_LOOPS_NOT_UNSEEN_DEV_JOINT_PRED_OR_GOAL'
            or r['model']!=model or l['model']!=model or not t['successful_owned_execution']
            or t['owned_worker_exit_code']!=0 or t['guard_stop_reason'] is not None or not t['complete_result_present']
            or t['CPU_compute_workers']!=1 or t['peak_sampled_worker_threads']>2
            or t['peak_identified_native_driver_helpers']>1 or t['training_GPU_hours']!=0
            or t['peak_sampled_owned_worker_RSS_bytes']>EXPECTED['rss_bytes_ceiling']
            or t['wall_seconds_including_worker_imports_network']>EXPECTED['wall_seconds_ceiling']
            or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours']!=0
            or not r['all13_snapshot_fingerprints_before_policy'] or r['optimizer_steps']!=0
            or r['GT_or_future_truth_to_memory_repair'] or r['dev_final_Val_Test_pixels_or_new_nonKnown_semantics_read']
            or r['policy_resubstitution_is_unseen_dev_or_Goal_gain']
            or l['live_remote_registration_proof']!=r['live_remote_registration_proof']
            or l['preregistration_commit']!=r['live_remote_registration_proof']['sha']
            or set(l['source_sha256'])!=set(SOURCES) or set(r['arms'])!=set(EXPECTED['arms'])):
        raise ValueError('Actual bounded normal exact-head full four-arm policy, no partial/expanded/selected gain')
    formal=verified_cohort(root)
    parameters={'initial_Raw':'3ca6d4d867ba1958e6c100b771d1d5bbd432feda40c5085d761818e0ac86582e'}
    parameters.update({('calendar'+str(x['calendar_seed']) if x['control']=='calendar_matched' else 'ONE_fullpolicy'):x['model_parameter_sha256'] for x in formal['trials']})
    if r['formal_parameter_sha256']!=parameters[model]:raise ValueError('Original frozen endpoint required')
    s=r['policy_scope']
    if ((s['videos'],s['physical_tracks'],s['observations'],s['metadata_frames'],s['empty_GT_frames'])!=(10,54,1268,374,1)
            or s['existing_Known_target_counts']!={'Known':34,'policy_pseudo':10} or s['masked_loss_tracks']!=10
            or s['physical_label_strata']!={'Known':34,'policy_pseudo':10,'no_retained_semantic_truth':10}
            or not s['GT_localization_oracle_control_not_qualified_Pred'] or not s['external_loss_only_not_observation_or_model_features']
            or any(s[k] for k in ('development_final_or_other_video_payload_bodies_opened',
                'development_final_or_other_video_semantic_values_decoded','masked_tracks_dropped_or_declared_background',
                'new_nonKnown_semantic_values_or_vocabulary_decoded'))):
        raise ValueError('Complete lawful policy/unknown scope, not dev/Pred/new semantics')
    schedules={}
    for mode,arm in r['arms'].items():
        if (set(arm['orders'])!={'1027','1028','1029'} or not arm['actual_full_ID_reverse_equal']
                or not arm['actual_full_Known_binding_shuffle_equal'] or arm['future_mutations']<=0
                or not arm['nonempty_prior_event_origin_history_state_equal']
                or not arm['actual_external_label_swap_numeric_writes_equal']):
            raise ValueError('Each complete arm/all3orders/full mandatory nonvacuous probes')
        for order,x in arm['orders'].items():
            m=x['measured'];c=m['full_clock'];a=m['origin_aware_audit'];legacy=a['legacy_identity_dirty_history_includes_continuation_not_primary_scored_match']
            if (m['mode']!=mode or m['order_seed']!=int(order) or m['optimizer_steps']!=0
                    or m['own_continuation_is_scored_reuse_or_semantic_certificate'] or m['GT_to_policy_or_memory_writes']
                    or (c['canonical_groups'],c['physical_observations'],c['metadata_frames'],c['empty_frames'])!=(1268,1268,374,1)
                    or sum(c.get(k,0) for k in ('KNOWN','NEW','EXISTING','WAIT'))!=1268
                    or m['actual_commits']!=c.get('NEW',0)+c.get('EXISTING',0) or m['allocated_tokens']!=c.get('NEW',0)
                    or mode=='full_raw' and x['actual_original_baseline_parity'] is not True
                    or mode!='full_raw' and x['actual_original_baseline_parity'] is not None):
                raise ValueError('Complete clock/commit/birth conservation and each original baseline parity')
            schedule=m['video_ordinal_schedule_sha256']
            if order in schedules and schedules[order]!=schedule:raise ValueError('Same physical schedule across arms')
            schedules[order]=schedule
            origins=a['physical_origin_observations']
            if (sum(origins.values())!=1268 or any(type(n) is not int or n<0 for n in origins.values())
                    or set(origins)-{'SCORED_CANDIDATE','ROUTE_CONTINUATION_NOT_CATEGORY_MATCH','CONFLICTING_ROUTE_CONTINUITY'}
                    or origins.get('ROUTE_CONTINUATION_NOT_CATEGORY_MATCH',0)>c.get('EXISTING',0)
                    or origins.get('CONFLICTING_ROUTE_CONTINUITY',0)>c.get('WAIT',0)
                    or mode in ('full_raw','exclude_query_raw') and origins.get('ROUTE_CONTINUATION_NOT_CATEGORY_MATCH',0)
                    or a['GT_or_origin_used_to_rewrite_memory_history'] or a['full_semantic_CT_Hscore_Pred_or_Goal_accepted']
                    or legacy['observations']!=1268 or legacy['canonical_groups']!=1268
                    or legacy['allocated_tokens']!=m['allocated_tokens'] or legacy['unknown_declared_background_or_Novel']):
                raise ValueError('Actual full origin counts/history, no relabeled semantic certificates')
            expected={'1','2','4','8','16','all_available'}
            if set(a['primary_scored_reuse_prefix_snapshots'])!=expected or set(legacy['prefix_snapshots'])!=expected:
                raise ValueError('All complete fixed prefix slices retained')
            for prefix,p in a['primary_scored_reuse_prefix_snapshots'].items():
                old=legacy['prefix_snapshots'][prefix]
                if ((p['physical_tracks'],p['Known_queries'],p['declared_pseudo_queries'])!=(54,34,10)
                        or not p['certified_reuse_excludes_routing_continuation']
                        or not 0<=p['routing_only_continuation_declared_pseudo_tracks']<=10
                        or not 0<=p['routing_only_continuation_all_physical_tracks']<=54):
                    raise ValueError('All physical/known/pseudo/masked prefix denominators and origin masking')
                for key in ('Known_micro_accuracy','Known_macro_accuracy','all_physical_commit_track_coverage',
                            'certified_cross_video_reuse_micro_lower_bound','certified_cross_video_reuse_macro_lower_bound',
                            'scored_Existing_claim_track_coverage_over_all_declared_pseudo'):
                    if not math.isfinite(p[key]) or not 0<=p[key]<=1:raise ValueError('Finite scoped rates, no forged coverage')
                macro_ulps=2 if model=='ONE_fullpolicy' else 1
                if (p['certified_cross_video_reuse_micro_lower_bound']>p['scored_Existing_claim_track_coverage_over_all_declared_pseudo']+1e-12
                        or not same_macro_aggregate(p['Known_macro_accuracy'],old['Known_macro_accuracy'],macro_ulps)
                        or any(p[k]!=old[k] for k in ('Known_micro_accuracy','all_physical_commit_track_coverage',
                            'declared_pseudo_absorbed_as_Known_queries','declared_pseudo_anonymous_commit_queries'))):
                    raise ValueError('Scored reuse cannot count continuation; other physical outcomes retained')
            for key,n in (('first_commit_latency',54),('first_scored_Existing_claim_latency_declared_pseudo',10),
                          ('first_certified_scored_cross_video_reuse_latency_declared_pseudo',10)):
                latency=a[key]
                if (latency['full_track_denominator']!=n or latency['events']+latency['unresolved_tracks_retained']!=n
                        or not 0<=latency['events']<=n or not 0<=latency['unresolved_tracks_retained']<=n
                        or latency['units']!='observed_updates_1_based_not_dense_frame_or_wall_time'):
                    raise ValueError('Full censored observed-update latency, not dropped unresolved tracks')
                values=[latency[k] for k in ('mean_conditional_on_event','p50_conditional_on_event','p95_conditional_on_event')]
                if (not latency['events'] and any(v is not None for v in values)
                        or latency['events'] and (any(v is None or not math.isfinite(v) or not 1<=v<=1268 for v in values) or values[1]>values[2])):
                    raise ValueError('Event-conditioned finite latency, no zero/unresolved imputation')
    for p,h in l['source_sha256'].items():
        if (root/p).is_symlink() or sha256_file(root/p)!=h:raise ValueError('Actual registered229source/public paths changed')
    return {'model':model,'launch_sha256':sha256_file(paths[0]),'result_sha256':sha256_file(paths[1]),
        'terminal_sha256':sha256_file(paths[2]),'all4arms_each_full3orders_original_baseline_and_full_probes_verified':True,
        'origin_masked_scored_reuse_all_physical_unknown_latency_denominators_verified':True,
        'private_GT_weights_or_prediction_rows_read':False,'unseen_dev_joint_Pred_final_or_Goal_accepted':False}


def complete_policy_comparison(root):
    """Build a public all-five aggregate after every head independently qualifies."""
    models=tuple(EXPECTED['models'])
    proof={model:verified_trial(root,model) for model in models}
    clock_keys=('canonical_groups','physical_observations','metadata_frames','empty_frames',
                'KNOWN','NEW','EXISTING')
    metric_keys=('new_births','Known_micro_accuracy','Known_macro_accuracy',
                 'declared_pseudo_absorbed_as_Known_queries',
                 'scored_Existing_claim_track_coverage_over_all_declared_pseudo',
                 'certified_cross_video_reuse_micro_lower_bound',
                 'routing_only_continuation_all_physical_tracks',
                 'routing_only_continuation_declared_pseudo_tracks',
                 'first_scored_Existing_events','first_scored_Existing_mean_latency',
                 'first_scored_Existing_unresolved_tracks')
    rows={}
    for model in models:
        launch,result,terminal=(json.loads((root/p).read_text()) for p in trial_sources(model))
        arms={}
        for mode,arm in result['arms'].items():
            orders={}
            for order,trial in arm['orders'].items():
                measured=trial['measured'];clock=measured['full_clock']
                snapshot=measured['origin_aware_audit']['primary_scored_reuse_prefix_snapshots']['all_available']
                latency=measured['origin_aware_audit']['first_scored_Existing_claim_latency_declared_pseudo']
                orders[order]={
                    'full_clock':{key:clock.get(key,0) for key in clock_keys},
                    'new_births':clock['NEW'],
                    'Known_micro_accuracy':snapshot['Known_micro_accuracy'],
                    'Known_macro_accuracy':snapshot['Known_macro_accuracy'],
                    'declared_pseudo_absorbed_as_Known_queries':snapshot['declared_pseudo_absorbed_as_Known_queries'],
                    'scored_Existing_claim_track_coverage_over_all_declared_pseudo':snapshot['scored_Existing_claim_track_coverage_over_all_declared_pseudo'],
                    'certified_cross_video_reuse_micro_lower_bound':snapshot['certified_cross_video_reuse_micro_lower_bound'],
                    'routing_only_continuation_all_physical_tracks':snapshot['routing_only_continuation_all_physical_tracks'],
                    'routing_only_continuation_declared_pseudo_tracks':snapshot['routing_only_continuation_declared_pseudo_tracks'],
                    'first_scored_Existing_events':latency['events'],
                    'first_scored_Existing_mean_latency':latency['mean_conditional_on_event'],
                    'first_scored_Existing_unresolved_tracks':latency['unresolved_tracks_retained'],
                }
            arms[mode]={'orders':orders,'mean_over_orders':{
                key:math.fsum(orders[order][key] for order in ('1027','1028','1029'))/3
                for key in metric_keys}}
        rows[model]={'triplet_sha256':{
                'launch':proof[model]['launch_sha256'],'result':proof[model]['result_sha256'],
                'normal_terminal':proof[model]['terminal_sha256']},
            'normal_wall_seconds':terminal['wall_seconds_including_worker_imports_network'],
            'normal_peak_RSS_bytes':terminal['peak_sampled_owned_worker_RSS_bytes'],'arms':arms}
    return {
        'status':'ALL5_ACTUAL_COMPLETE_PUBLIC_CONTINUITY_POLICY_COMPARISON_NOT_GOAL',
        'models':rows,
        'policy_denominators':{'videos':10,'physical_tracks':54,'observations':1268,
            'metadata_frames':374,'empty_frames':1,'Known_tracks':34,
            'declared_pseudoNovel_tracks':10,'masked_tracks':10,'orders':[1027,1028,1029]},
        'all_five_heads_independently_verified':True,
        'training_GPU_hours':0,'private_GT_weights_prediction_rows_loaded':False,
        'head_arm_order_threshold_selected':False,'certified_scored_cross_video_reuse_demonstrated':False,
        'joint_Novel_R_C_M_strong_Pred_dense_calibrated_controls_locked_final_or_Goal_accepted':False,
        'macro_qualifier_note':'Prior heads permit exactly1 binary64 ulp for insertion-order aggregate; ONE_fullpolicy permits exactly2 ulps only for its legacy/primary prefix aggregate. All other public event/count/origin/coverage/probe checks remain exact.'
    }
