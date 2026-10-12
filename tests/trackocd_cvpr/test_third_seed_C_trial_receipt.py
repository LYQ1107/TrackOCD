import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_Top8_conditioned_s1029'


def test_third_independent_C_seed_high_Existing_is_not_certified_category_reuse():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '0b53c928b5fc0f4001bffe5b475452e39162f00a8e244b360a73225d120350f6'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    assert r['variant'] == 'Top8_conditioned' and r['seed'] == 1029
    assert s['optimizer_steps'] == 200 and s['trainable_parameters'] == 708
    assert s['actual_optimizer_family_groups'] == {'Known': 266, 'policy_pseudo': 178}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 0
    assert s['episodes'][1]['physical_actions']['EXISTING'] == 1169
    assert s['episodes'][2]['physical_actions']['EXISTING'] == 1210
    assert s['episodes'][1]['positive_other_video_groups_all_frames'] == s['episodes'][2]['positive_other_video_groups_all_frames'] == 0
    assert s['episodes'][3]['physical_actions']['EXISTING'] == 1194
    assert s['episodes'][3]['positive_other_video_groups_all_frames'] == 60
    initial = {s['model_parameter_sha256_before']}
    for seed in (1027, 1028):
        other = json.loads((OUT.with_name('C_GT_window_v1_Top8_conditioned_s' + str(seed)) / 'result.json').read_text())
        initial.add(other['summary']['model_parameter_sha256_before'])
    assert len(initial) == 3
    assert all((e['videos'], e['metadata_frames'], e['physical_observations']) == (10, 374, 1268) for e in s['episodes'])
    assert r['endpoint']['reopen_exact'] and not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None


def test_third_seed_pool_control_preserves_action_variance_and_zero_optimizer_crossvideo_credit():
    out = OUT.with_name('C_GT_window_v1_Top8_pool_context_s1029')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == '3a8e3672f48e467ba86140493b51b7a5bc64329762b461a50fb3486e4e801bd4'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    paired = json.loads((OUT / 'result.json').read_text())['summary']
    assert r['seed'] == 1029 and r['variant'] == 'Top8_pool_context'
    assert s['model_parameter_sha256_before'] == paired['model_parameter_sha256_before']
    assert s['actual_optimizer_family_groups'] == {'Known': 266, 'policy_pseudo': 183}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2 and s['optimizer_steps'] == 200
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 0
    assert [e['physical_actions'].get('EXISTING', 0) for e in s['episodes']] == [0, 151, 163, 861]
    assert [e['positive_other_video_groups_all_frames'] for e in s['episodes']] == [0, 22, 44, 89]
    assert all((e['videos'], e['metadata_frames'], e['physical_observations']) == (10, 374, 1268) for e in s['episodes'])
    assert r['endpoint']['reopen_exact'] and not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None


def test_full_nine_C_matrix_actual_endpoints_and_third_Top1_exposure_not_a_win():
    out = OUT.with_name('C_GT_window_v1_Top1_conditioned_s1029')
    assert hashlib.sha256((out / 'result.json').read_bytes()).hexdigest() == 'eee833acf5576e67bde892bdd392ab7d217d2d46d2f019feaea9e73d0a967776'
    r = json.loads((out / 'result.json').read_text()); s = r['summary']
    assert s['actual_optimizer_family_groups'] == {'Known': 210, 'policy_pseudo': 185}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 0
    assert [e['physical_actions'].get('EXISTING', 0) for e in s['episodes']] == [0, 0, 70, 36]
    initial = {}
    for variant in ('Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context'):
        for seed in (1027, 1028, 1029):
            p = OUT.with_name('C_GT_window_v1_' + variant + '_s' + str(seed))
            x = json.loads((p / 'result.json').read_text()); q = x['summary']
            assert x['variant'] == variant and x['seed'] == seed
            assert q['optimizer_steps'] == 200 and q['trainable_parameters'] == 708
            assert len(q['episodes']) == 4
            assert all((e['videos'], e['metadata_frames'], e['physical_observations']) == (10, 374, 1268) for e in q['episodes'])
            assert x['endpoint']['reopen_exact'] and x['checkpoint_selection'] == 'Fixed200 endpoint'
            assert not x['dev_final_Val_Test_or_new_nonKnown_labels_used']
            assert not x['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
            initial.setdefault(seed, set()).add(q['model_parameter_sha256_before'])
            t = json.loads(p.with_name(p.name + '_supervisor_attempt_1.json').read_text())
            assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert all(len(v) == 1 for v in initial.values())
    assert len(set(next(iter(v)) for v in initial.values())) == 3
