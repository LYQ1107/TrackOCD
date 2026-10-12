"""Real limited GT-control prefix fixture; not a trained/predicted result."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/selected_fit_crop_prefix_smoke_v1'


def test_actual_GT_control_crop_prefix_preserves_Raw_and_future_invariance():
    result = json.loads((OUT / 'result.json').read_text())
    assert result['status'] == 'VERIFIED_FIT_GT_CROP_PREFIX_FIXTURE_NOT_TRAINED_OR_PREDICTED'
    assert result['images_decoded'] == 2 and result['actual_forward_calls'] == 3
    assert result['GT_geometric_localization_used_as_explicit_control']
    prefix = result['limited_actual_GT_crop_prefix_check']
    assert prefix['actual_observation_prefixes_checked'] == [1, 2]
    assert prefix['R_identity_start_Raw_exact'] and prefix['future_tail_mutation_prefix_exact']
    assert prefix['R_parameters_untrained_unchanged']
    assert prefix['GT_control_quality_constant'] == 1 and prefix['GT_geometry_unclipped_amodal_normalized']
    assert prefix['not_a_frame_online_controller_or_trained_method']
    assert not result['development_or_final_pixels_read']
    assert not result['GT_class_role_text_or_ID_in_forward']
    assert not result['full_cache_or_training_started'] and not result['new_protocol_frozen']
    assert result['live_remote_registration_proof']['sha'] == 'f5c947b4bc6481209a6c3b527a6a0444edd4be7a'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['successful_owned_execution'] and receipt['owned_worker_exit_code'] == 0
    assert receipt['guard_stop_reason'] is None and receipt['peak_sampled_worker_threads'] == 1
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] <= 2147483648
