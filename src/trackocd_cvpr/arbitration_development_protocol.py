"""Exact complete Train-dev observer protocol after real all5 policy pilot."""
import json
from src.trackocd_v2.io import sha256_file
from .arbitration_attribution_qualification import verify_actual_policy_pilot

CONFIG = 'configs/trackocd_cvpr/arbitration_full_development_v1.json'
PREFIX = 'outputs/trackocd_cvpr/cvpr3/arbitration_full_development_v1'
PILOT_PROOF = 'outputs/trackocd_cvpr/cvpr3/arbitration_policy_attribution_pilot_qualification_v1.json'
PILOT_HASHES = {
    'outputs/trackocd_cvpr/cvpr3/arbitration_policy_attribution_pilot_v1/launch.json':'6d7eeb296cfee5bc8bf8ec8fe173fa568d45b3603ba7a6d6903d9e3de457e786',
    'outputs/trackocd_cvpr/cvpr3/arbitration_policy_attribution_pilot_v1/result.json':'c2398581fc4b505f03781eee098cf889735b56e43d7ec6ccfd242759c7e417d4',
    'outputs/trackocd_cvpr/cvpr3/arbitration_policy_attribution_pilot_v1_supervisor_attempt_1.json':'1583d0dff156bfc59266634b99b60123a069e853eac8b09103fb69a0a515590e',
}
EXPECTED = {
    'status':'REGISTERED_READONLY_ALL5_COMPLETE_TRAIN_DEV_ARBITRATION_OBSERVER_ACTUAL_POLICY_PILOT_REQUIRED',
    'base_data_config':'configs/trackocd_cvpr/C_GT_causal_development_v1.json',
    'models':['initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy'],
    'order_seeds':[1027,1028,1029],'development_videos':72,'physical_tracks':472,'observations':8753,
    'metadata_frames':2562,'empty_frames':116,'Known_tracks':302,'declared_pseudo_tracks':119,'masked_tracks':51,
    'K':8,'optimizer_steps':0,'execution_device':'CPU','CPU_compute_workers':1,
    'rss_bytes_ceiling':3221225472,'wall_seconds_ceiling':3000,'output_bytes_ceiling':16777216,
    'output_prefix':PREFIX,'maximum_actual_attempts_per_model':1,'one_explicit_model_per_worker':True,
    'all13_private_snapshot_fingerprints_before_dev_input':True,
    'unmodified_qualified_policy_observer_numeric_source':True,
    'full_normal_reference_event_commit_state_parity_for_EACH_order':True,
    'full_ID_reverse_and_nonvacuous_future_prefix_probes':True,
    'external_dev_kind_bucket_remap_is_training_target_or_new_semantic_value':False,
    'shadow_exclusion_writes_or_retroactive_GT_history_edit':False,
    'conditional_shadow_is_full_intervention_stream_gain':False,
    'unknown_as_negative_BG_Novel_or_new_Novel_vocabulary':False,
    'final_Val_Test_original_pixels_or_new_nonKnown_GT_read':False,
    'dev_selected_head_seed_order_threshold_or_automatic_training':False,
    'all5_complete_or_causal_science_or_Goal_claimed_from_source':False,
}


def configuration(root,model):
    cfg = json.loads((root/CONFIG).read_text())
    if cfg != EXPECTED or model not in EXPECTED['models']:
        raise ValueError('Exact all5/full3orders/dev clock/unchanged guards, no subset or selected head')
    for p,h in PILOT_HASHES.items():
        if (root/p).is_symlink() or sha256_file(root/p) != h:raise ValueError('Exact actual complete policy pilot triplet')
    actual = verify_actual_policy_pilot(root)
    if json.loads((root/PILOT_PROOF).read_text()) != actual:raise ValueError('Genuine public policy prerequisite proof')
    base = json.loads((root/cfg['base_data_config']).read_text())
    keys = ('approval','frozen_protocol','private_assignment','private_Known_support','fit_cache_directory',
            'fit_cache_execution_commit','policy_dev_cache_directory','policy_dev_execution_commit')
    for key in ('physical_tracks','observations','metadata_frames','empty_frames','order_seeds','optimizer_steps',
                'execution_device','CPU_compute_workers','rss_bytes_ceiling','wall_seconds_ceiling','output_bytes_ceiling'):
        if base[key] != cfg[key]:raise ValueError('Same qualified uncapped dev and original safety protocol')
    return {**{k:base[k] for k in keys},**cfg,'model':model,'output_directory':PREFIX+'_'+model}
