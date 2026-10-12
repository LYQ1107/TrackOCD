import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/fit_R_optimizer_smoke_v1'


def test_actual_ten_fit_Known_updates_current_support_gradients_checkpoint_and_owned_exit():
    p = OUT / 'result.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest() == '2e8b2dc57185be352bc9423df24ef19987e9d6bec761cafb98753837d186c851'
    r = json.loads(p.read_text())
    assert r['status'] == 'VERIFIED_TEN_REAL_FIT_KNOWN_CPU_R_OPTIMIZER_STEPS_NOT_SCIENTIFIC_ACCEPTANCE'
    assert r['live_remote_registration_proof']['sha'] == '2f05201a9ae667b9d7669d7b5490ea03c5cac610'
    assert r['completed_optimizer_steps'] == 10 and r['seed'] == 1027
    assert r['model_parameter_sha256_before'] != r['model_parameter_sha256_after']
    assert r['private_checkpoint']['reopen_exact'] and r['private_checkpoint']['bytes'] == 616115
    scope = r['label_and_physical_scope']
    assert (scope['classes_for_loss'], scope['eligible_supervised_tracks'], scope['eligible_supervised_observations']) == (5, 1158, 22279)
    assert scope['physical_tracks_kept'] == 1401 and scope['masked_no_fit5_target_tracks'] == 243
    assert not scope['other_role_semantic_values_decoded']
    assert not scope['unlabeled_or_other_Known_assigned_background_or_Novel']
    assert all(row['support_gradient_norm'] > 0 and row['query_gradient_norm'] > 0 for row in r['loss_curve'])
    assert all(row['support_observations'] > 16384 for row in r['loss_curve'])
    assert not r['numeric_model_inputs_contain_labels_roles_IDs_or_futurelength']
    assert not r['queryself_or_samevideo_in_offline_Known_support']
    assert not r['development_final_pixels_or_labels_originalGT_or_DINOweights_loaded']
    assert not r['trained_scientific_seeds_or_R_C_M_mechanisms_or_Goal_A_B_C_accepted']
    assert r['resources']['execution_device'] == 'CPU' and not r['resources']['CUDA_initialized']
    t = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert t['successful_owned_execution'] and t['owned_worker_exit_code'] == 0 and t['guard_stop_reason'] is None
    assert t['last_persisted_completed_optimizer_steps'] == 10 and t['only_new_owned_Popen_worker_managed']
