"""Actual public structural receipt checks, not trained-method acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v4_compressed'
REGISTERED = 'ddc434c4bb2861cc2fdd9aa3c8dd381e1462c87a'


def load(path):
    return json.loads(path.read_text())


def test_actual_v4_full_census_retained_and_scoped_negative_not_global_proof():
    result = load(OUT / 'result.json')
    structural = result['structural_result']
    assert result['status'] == structural['status'] == 'SOLVER_SCOPED_INFEASIBLE'
    assert result['census_reconciliation'] == dict(observations=38680, tracks=1962,
                                                    videos=429, positive_Known_classes=48)
    assert structural['solver_executed'] and structural['solver_status'] == 2
    assert structural['all_original_videos_retained'] == 429
    assert structural['distinct_presence_patterns'] == 65
    assert (structural['variables'], structural['constraints'], structural['nonzeros']) == (1500, 3547, 13596)
    assert structural['exact_presence_pattern_count_compression']
    assert structural['solver_reported_scoped_infeasibility_not_formal_global_data_proof']
    assert not structural['feasible_joint_assignment_verified']
    assert not structural['new_protocol_frozen']
    assert not structural['complete_visual_category_absence_proven']
    assert not structural['model_scores_or_names_used']
    assert structural['integer_necessary_conditions']['union_support_forced_merges'] == 0


def test_actual_live_registration_and_original_committed_source_bytes():
    result = load(OUT / 'result.json'); launch = load(OUT / 'launch.json')
    assert result['preregistration_commit'] == launch['preregistration_commit'] == REGISTERED
    assert result['source_sha256'] == launch['source_sha256']
    assert result['live_remote_registration_proof'] == {
        'method': 'fresh_git_ls_remote', 'ref': 'refs/heads/research/cvpr-trackocd-autonomous-v2', 'sha': REGISTERED}
    for path, expected in result['source_sha256'].items():
        original = subprocess.check_output(['git', 'show', f'{REGISTERED}:{path}'], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == expected
    assert result['fresh_source_no_existing_bytecode_cache']
    assert result['optimization_training_steps'] == 0
    assert not result['original_annotation_old_target_tables_names_Novel_Val_Test_pixels_features_models_read']


def test_actual_owned_worker_terminal_under_registered_resource_limits():
    receipt = load(OUT.with_name(OUT.name + '_supervisor_attempt_1.json'))
    cfg = load(ROOT / 'configs/trackocd_cvpr/joint_known_protocol_v4_compressed.json')
    assert receipt['owned_worker_exit_code'] == 0 and receipt['guard_stop_reason'] is None
    assert receipt['successful_owned_execution'] and receipt['only_new_owned_Popen_worker_managed']
    assert receipt['peak_sampled_worker_threads'] == 1
    assert 0 < receipt['peak_sampled_owned_worker_RSS_bytes'] <= cfg['rss_bytes_ceiling']
    assert 0 < receipt['wall_seconds_including_imports_network_and_solver'] <= cfg['wall_seconds_ceiling']
    assert receipt['training_GPU_hours'] == 0
    # Private witness is deliberately NOT opened by these public receipt tests.
    assert load(OUT / 'result.json')['private_output']['bytes'] == 2186
