import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_Top1_conditioned_s1027'


def test_Top1_complete_seed1027_controls_and_different_label_opportunities_not_hidden():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '67cdd2052717d364cd88bfea8b30d0f68b19ccb615dc74cfdc65dbd5cec50569'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    assert r['seed'] == 1027 and r['variant'] == 'Top1_conditioned'
    assert s['optimizer_steps'] == 200 and s['trainable_parameters'] == 708
    assert s['actual_optimizer_family_groups'] == {'Known': 359, 'policy_pseudo': 110}
    assert s['actual_optimizer_policy_pseudo_classes'] == 2 and s['actual_optimizer_semantic_classes'] == 5
    assert all(e['physical_actions'] == {'KNOWN': 1268} for e in s['episodes'][1:])
    assert all(e['positive_other_video_groups_all_frames'] == 0 for e in s['episodes'])
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 0
    assert all(e['loss_plan_counts_all_frames']['MASKED_UNCERTAIN_PSEUDO_OR_ABSENT_KNOWN_CANDIDATE'] == 249 for e in s['episodes'])
    for variant in ('Top8_conditioned', 'Top8_pool_context'):
        other = json.loads((OUT.with_name('C_GT_window_v1_' + variant + '_s1027') / 'result.json').read_text())['summary']
        assert s['model_parameter_sha256_before'] == other['model_parameter_sha256_before']
        assert other['actual_optimizer_policy_pseudo_classes'] == 1  # Not identical supervision exposure.
    assert len(s['episodes']) == 4
    assert all((e['videos'], e['metadata_frames'], e['physical_observations']) == (10, 374, 1268) for e in s['episodes'])
    assert r['endpoint']['reopen_exact'] and r['checkpoint_selection'] == 'Fixed200 endpoint'
    assert not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None
