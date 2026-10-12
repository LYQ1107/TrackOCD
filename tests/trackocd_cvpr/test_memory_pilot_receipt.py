import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr4/memory_optimizer_pilot_v1'


def test_real_M_pilot_has_actual_updates_full_scope_and_honest_exposure_limits():
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '5bbe27d48f96959775dcb558f85255c4d3c139210591a05a74e78ad58fd21e59'
    r = json.loads((OUT / 'result.json').read_text()); s = r['summary']
    assert s['optimizer_steps'] == 10 and s['trainable_parameters'] == 161
    assert s['model_parameter_sha256_before'] != s['model_parameter_sha256_after']
    assert s['complete_rollout'] == {'videos': 10, 'metadata_frames': 374, 'empty_frames': 1,
        'physical_observations': 1268, 'physical_actions': {'EXISTING': 1227, 'NEW': 41}}
    assert s['actual_optimizer_family_groups'] == {'Known': 20} and s['actual_optimizer_semantic_classes'] == 1
    assert s['version_lifecycle_records'] == {'Tentative': 1410}
    assert s['actual_anonymous_commits_retained'] == 1268
    assert not s['GT_driven_memory_repair_or_deleted_wrong_history']
    assert not s['strict_historical_purity_or_CT_or_Novel_gain_measured']
    assert r['checkpoint_selection'] == 'Fixed10 endpoint' and r['endpoint']['reopen_exact']
    assert r['endpoint']['sha256'] == '0a12dafdfb80b3cbb51a2170c6814bce6e6e62963f4c00dd45c9dc6f94b337b7'
    assert not r['old_R_C_M_or_pilot_warmstart'] and not r['dev_final_Val_Test_or_new_nonKnown_labels_used']
    assert not r['three_seed_M_strict_CT_Pred_or_Goal_accepted']
    terminal = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert terminal['successful_owned_execution'] and terminal['owned_worker_exit_code'] == 0
    assert terminal['guard_stop_reason'] is None and terminal['last_persisted_completed_optimizer_steps'] == 10
