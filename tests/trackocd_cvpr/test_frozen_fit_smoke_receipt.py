"""Real frozen CPU smoke, not trained R/C/M or crop/trajectory evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/selected_frozen_encoder_smoke_v3_terminal'


def test_actual_two_fit_decodes_three_CPU_forwards_frozen_and_clean_supervised_exit():
    result = json.loads((OUT / 'result.json').read_text())
    assert result['images_decoded'] == 2 and result['actual_forward_calls'] == 3
    assert result['descriptor_shape'] == [2, 768] and result['CPU_repeat_exact']
    assert result['frozen_parameters_unchanged']
    assert result['frozen_parameter_sha256'] == '42c4c180dbaa825ee5bfecc6e6bf392af6439a1d2aa97f05095ac09b5e7097a9'
    assert result['verified_tracked_encoder_python_files'] == 156
    assert result['verified_actual_imported_encoder_modules'] == 15
    assert result['encoder_Git_tree_python_entries'] == 157
    assert result['sparse_absent_python_paths_not_imported'] == ['setup.py']
    assert not result['GT_class_role_text_or_ID_in_forward']
    assert not result['development_or_final_pixels_read']
    assert not result['GT_or_predicted_crop_features_verified']
    assert not result['full_cache_or_training_started'] and not result['new_protocol_frozen']
    assert result['resources']['training_GPU_hours'] == 0 and not result['resources']['CUDA_initialized']
    assert result['live_remote_registration_proof']['sha'] == 'db96482cfc33d95aee83fb88e6edfe8c3dd44a12'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['successful_owned_execution'] and receipt['owned_worker_exit_code'] == 0
    assert receipt['guard_stop_reason'] is None and receipt['peak_sampled_worker_threads'] == 1
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] <= 2147483648
    assert receipt['wall_seconds_including_imports_network_and_solver'] <= 180
