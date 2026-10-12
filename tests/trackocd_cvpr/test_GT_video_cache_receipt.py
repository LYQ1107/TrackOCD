"""Public receipt contract; no private cached features required by this test."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/pilot_GT_video_cache_v1'


def test_actual_uncapped_GT_control_pilot_owned_clean_exit_and_precision():
    result_path = OUT / 'result.json'
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == 'da4375c4502280fd3ec244812e79a258315e3197bd7f33f6edf5b85126d09fdd'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'VERIFIED_TWO_FIT_VIDEO_UNCAPPED_GT_CONTROL_CACHE_PILOT_NOT_PRED_OR_TRAINED'
    assert result['live_remote_registration_proof']['sha'] == '3072fb2474bbe63ae699c3bbc8ff6e792ed03c18'
    assert (result['pilot_videos'], result['tracks'], result['observations'], result['metadata_frames']) == (2, 16, 418, 75)
    assert sum(s['every_visible_prefix']['every_visible_prefix_checked'] for s in result['shards']) == 418
    assert sum(s['every_visible_prefix']['prefixes_with_actual_future_tail_mutation'] for s in result['shards']) == 402
    assert all(s['reopen_exact'] for s in result['shards'])
    for name in ('precision_GPU_batch_single_repeat', 'precision_CPU_GPU_firstcrop', 'precision_float16_reader_firstcrop'):
        assert result[name]['max_absolute_error'] <= 0.0005 and result[name]['cosine'] >= 0.99999
    assert result['frozen_parameters_unchanged']
    assert result['frozen_parameter_sha256'] == '42c4c180dbaa825ee5bfecc6e6bf392af6439a1d2aa97f05095ac09b5e7097a9'
    assert result['GPU_idle_before_owned_allocation']['chosen_GPU_compute_applications_before_launch'] == 0
    assert not result['semantic_GT_roles_IDs_futurelength_in_numeric_forward']
    assert not result['development_final_or_protected_pixels_read']
    assert not result['optimizer_or_formal_training_started']
    assert not result['full407_cache_automatic_launch']
    assert not result['qualified_Pred_or_Goal_A_B_C_achieved']
    assert result['resources']['training_GPU_hours'] == 0
    assert result['resources']['GPU_peak_reserved_bytes'] < 5 * 1024**3
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['successful_owned_execution'] and receipt['owned_worker_exit_code'] == 0
    assert receipt['guard_stop_reason'] is None and receipt['only_new_owned_Popen_worker_managed']
    assert receipt['peak_sampled_owned_worker_RSS_bytes'] < 3 * 1024**3
    # CUDA driver background threads make this four OS threads, not four CPU
    # compute workers/GPUs. Do not import the CPU-only fixture's one-thread claim.
    assert receipt['peak_sampled_worker_threads'] == 4
