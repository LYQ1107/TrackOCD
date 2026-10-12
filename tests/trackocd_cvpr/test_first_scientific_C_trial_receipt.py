"""First actual 200-update C trial, including poor exposure and mixed history."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_window_v1_Top8_conditioned_s1027'


def test_fixed_C_endpoint_full_episodes_actual_optimizer_exposure_not_pilot_or_dev_acceptance():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '31104366407bd5956aa3ed78ffa1794e444f6dd6d47ff53f855d3a1aa9b48077'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    assert r['live_remote_registration_proof']['sha'] == '1cadefc1a7d71e436e7528a4c11bfa36d9d01022'
    assert r['variant'] == 'Top8_conditioned' and r['seed'] == 1027
    assert s['optimizer_steps'] == 200 and s['trainable_parameters'] == 708
    assert [x['step'] for x in s['loss_curve']] == list(range(1, 201))
    assert s['actual_optimizer_family_groups'] == {'Known': 538, 'policy_pseudo': 106}
    assert s['actual_optimizer_semantic_classes'] == 4 and s['actual_optimizer_policy_pseudo_classes'] == 1
    assert s['both_Known_and_pseudo_received_optimizer_loss']
    assert len(s['episodes']) == 4
    for e in s['episodes']:
        assert (e['videos'], e['metadata_frames'], e['empty_frames'], e['physical_observations']) == (10, 374, 1, 1268)
        assert e['optimizer_steps'] == 50
    assert s['episodes'][1]['physical_actions'] == s['episodes'][2]['physical_actions'] == {'KNOWN': 1268}
    assert s['episodes'][3]['physical_actions'] == {'KNOWN': 1073, 'NEW': 87, 'EXISTING': 108}
    assert s['episodes'][3]['positive_other_video_groups_all_frames'] == 104
    assert sum(x['positive_other_video_groups_this_update'] for x in s['loss_curve']) == 2
    assert not s['GT_repair_or_deleted_wrong_history']
    assert not s['self_support_exclusion_or_Pred_or_M_or_Goal_acceptance']
    assert r['checkpoint_selection'] == 'Fixed200 endpoint' and r['endpoint']['reopen_exact']
    assert not r['old_pilot_or_R_or_policy_warmstart']
    assert not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['all3seeds_or_distinct_R_C_M_or_qualified_Pred_or_Goal_accepted']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and not r['resources']['CUDA_initialized']
