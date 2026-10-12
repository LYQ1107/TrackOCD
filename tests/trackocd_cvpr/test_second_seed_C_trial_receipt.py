import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_Top8_conditioned_s1028'


def test_independent_seed1028_actual_C_fixed_endpoint_more_exposure_but_zero_reuse():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '27259d030b247d8780971078486d46863ee77708c1944765dc5997a6c9ae5353'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    first = json.loads((OUT.with_name('C_GT_window_v1_Top8_conditioned_s1027') / 'result.json').read_text())['summary']
    assert r['variant'] == 'Top8_conditioned' and r['seed'] == 1028
    assert s['optimizer_steps'] == 200 and s['trainable_parameters'] == 708
    assert s['model_parameter_sha256_before'] != first['model_parameter_sha256_before']
    assert s['actual_optimizer_family_groups'] == {'Known': 381, 'policy_pseudo': 97}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 19
    assert all(e['physical_actions'].get('EXISTING', 0) == 0 for e in s['episodes'])
    assert s['episodes'][0]['positive_other_video_groups_all_frames'] == 90
    assert s['episodes'][3]['positive_other_video_groups_all_frames'] == 80
    assert len(s['episodes']) == 4
    assert all((e['videos'], e['metadata_frames'], e['physical_observations']) == (10, 374, 1268) for e in s['episodes'])
    assert r['endpoint']['reopen_exact'] and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert not r['old_pilot_or_R_or_policy_warmstart'] and not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None


def test_seed1028_pool_control_same_exposure_and_zero_reuse_not_a_scoring_win():
    out = OUT.with_name('C_GT_window_v1_Top8_pool_context_s1028')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '696ebf8b3b6d58fc31f52cf5c86e9e5d1d74ba26e93042b202f32887ee3582be'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    paired = json.loads((OUT / 'result.json').read_text())['summary']
    assert r['seed'] == 1028 and r['variant'] == 'Top8_pool_context'
    assert s['model_parameter_sha256_before'] == paired['model_parameter_sha256_before']
    assert s['actual_optimizer_family_groups'] == paired['actual_optimizer_family_groups'] == {'Known': 381, 'policy_pseudo': 97}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2 and s['optimizer_steps'] == 200
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 19
    assert all(e['physical_actions'].get('EXISTING', 0) == 0 for e in s['episodes'])
    assert all((e['videos'], e['metadata_frames'], e['physical_observations']) == (10, 374, 1268) for e in s['episodes'])
    assert r['endpoint']['reopen_exact'] and not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
