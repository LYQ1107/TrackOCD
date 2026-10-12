"""Matched training controls, not an unmeasured development win."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_Top8_pool_context_s1027'


def test_actual_pool_context_control_matches_nominal_budget_and_preserves_opportunity_difference():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '5296266b2c617855c538659b5686ba0e7453a004d2e08c43fe1dac1f382b5db2'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    matched = json.loads((OUT.with_name('C_GT_window_v1_Top8_conditioned_s1027') / 'result.json').read_text())['summary']
    assert r['variant'] == 'Top8_pool_context' and r['seed'] == 1027
    assert s['optimizer_steps'] == matched['optimizer_steps'] == 200
    assert s['trainable_parameters'] == matched['trainable_parameters'] == 708
    assert s['model_parameter_sha256_before'] == matched['model_parameter_sha256_before']
    assert s['actual_optimizer_family_groups'] == {'Known': 538, 'policy_pseudo': 107}
    assert s['actual_optimizer_policy_pseudo_classes'] == 1 and s['actual_optimizer_semantic_classes'] == 4
    assert len(s['episodes']) == 4
    assert s['episodes'][1]['physical_actions'] == s['episodes'][2]['physical_actions'] == {'KNOWN': 1268}
    assert s['episodes'][3]['physical_actions'] == {'KNOWN': 1166, 'NEW': 75, 'EXISTING': 27}
    assert s['episodes'][3]['positive_other_video_groups_all_frames'] == 41
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 2
    for e in s['episodes']:
        assert (e['videos'], e['metadata_frames'], e['empty_frames'], e['physical_observations']) == (10, 374, 1, 1268)
        assert e['optimizer_steps'] == 50
    assert r['endpoint']['reopen_exact'] and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None
